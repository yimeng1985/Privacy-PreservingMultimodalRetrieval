import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler

# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0


class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
        early_stop_patience=0,
        keep_last_n_checkpoints=3,
        eval_interval=0,
    ):
        self.model = model
        self.diffusion = diffusion
        self.data = data
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps
        self.early_stop_patience = early_stop_patience
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        # How often to evaluate the EMA loss for best-model / early-stopping.
        # Defaults to save_interval if not specified.
        self.eval_interval = eval_interval if eval_interval > 0 else save_interval

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size * dist.get_world_size()

        # Best-model & early stopping tracking
        self.best_loss = float("inf")
        self._loss_accum = 0.0
        self._loss_count = 0
        self._no_improve_count = 0  # consecutive eval windows without improvement

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        self._optimizer_loaded = False  # Track whether optimizer state was loaded
        self.warmup_steps = 0  # Set to >0 when we need warmup
        if self.resume_step:
            self._load_optimizer_state()
            if not self._optimizer_loaded:
                # Optimizer state not loaded — use lr warmup to let AdamW build
                # proper per-parameter variance estimates before applying full lr.
                # Without this, AdamW's adaptive lr can cause disproportionately
                # large updates for low-variance parameters, risking collapse.
                self.warmup_steps = 2000
                logger.log(f"Optimizer state not loaded. Using lr warmup for "
                           f"{self.warmup_steps} steps (0 → {self.lr}).")
            # Model was resumed, either due to a restart or a checkpoint
            # being specified at the command line.
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

        if th.cuda.is_available():
            self.use_ddp = True
            self.ddp_model = DDP(
                self.model,
                device_ids=[dist_util.dev()],
                output_device=dist_util.dev(),
                broadcast_buffers=False,
                bucket_cap_mb=128,
                find_unused_parameters=False,
            )
        else:
            if dist.get_world_size() > 1:
                logger.warn(
                    "Distributed training requires CUDA. "
                    "Gradients will not be synchronized properly!"
                )
            self.use_ddp = False
            self.ddp_model = self.model

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            if dist.get_rank() == 0:
                logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
                self.model.load_state_dict(
                    dist_util.load_state_dict(
                        resume_checkpoint, map_location=dist_util.dev()
                    )
                )

        dist_util.sync_params(self.model.parameters())

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            if dist.get_rank() == 0:
                logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
                state_dict = dist_util.load_state_dict(
                    ema_checkpoint, map_location=dist_util.dev()
                )
                ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        dist_util.sync_params(ema_params)
        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            try:
                self.opt.load_state_dict(state_dict)
                self._optimizer_loaded = True
                logger.log("Optimizer state loaded successfully.")
            except (ValueError, RuntimeError) as e:
                logger.log(
                    f"WARNING: Could not load optimizer state (likely format change "
                    f"from legacy FP16 to AMP): {e}. "
                    f"Optimizer will restart from scratch. Training will continue normally."
                )
                self._optimizer_loaded = False

    def run_loop(self):
        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0 and self.step > 0:
                self.save()
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", ""):
                    return
            # Best-model evaluation & early stopping check
            if self.step % self.eval_interval == 0 and self.step > 0:
                if self._check_best_and_early_stop():
                    logger.log(
                        f"Early stopping triggered: no improvement for "
                        f"{self.early_stop_patience} eval windows. "
                        f"Best loss: {self.best_loss:.6f}"
                    )
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.
        if (self.step - 1) % self.save_interval != 0:
            self.save()

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        for i in range(0, batch.shape[0], self.microbatch):
            micro = batch[i : i + self.microbatch].to(dist_util.dev())
            micro_cond = {
                k: v[i : i + self.microbatch].to(dist_util.dev())
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,
                t,
                model_kwargs=micro_cond,
            )

            with self.mp_trainer.autocast_ctx():
                if last_batch or not self.use_ddp:
                    losses = compute_losses()
                else:
                    with self.ddp_model.no_sync():
                        losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self._accumulate_loss(loss.item())
            self.mp_trainer.backward(loss)

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        # Apply warmup if optimizer was not loaded (fresh optimizer with pretrained model)
        if self.warmup_steps > 0 and self.step < self.warmup_steps:
            warmup_factor = self.step / self.warmup_steps
            lr = self.lr * warmup_factor
            for param_group in self.opt.param_groups:
                param_group["lr"] = lr
            return

        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def _accumulate_loss(self, loss_val):
        """Track running average loss for best-model evaluation."""
        self._loss_accum += loss_val
        self._loss_count += 1

    def _check_best_and_early_stop(self):
        """
        Evaluate the average loss over the last eval window.
        Save the model if it's the best so far.
        Return True if early stopping should trigger.
        """
        if self._loss_count == 0:
            return False

        avg_loss = self._loss_accum / self._loss_count
        self._loss_accum = 0.0
        self._loss_count = 0

        global_step = self.step + self.resume_step
        logger.log(
            f"[Eval @ step {global_step}] avg_loss={avg_loss:.6f}, "
            f"best_loss={self.best_loss:.6f}"
        )

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self._no_improve_count = 0
            self._save_best()
            logger.log(f"  → New best model saved (loss={avg_loss:.6f})")
        else:
            self._no_improve_count += 1
            logger.log(
                f"  → No improvement ({self._no_improve_count}/"
                f"{self.early_stop_patience if self.early_stop_patience > 0 else '∞'})"
            )

        if self.early_stop_patience > 0 and self._no_improve_count >= self.early_stop_patience:
            return True
        return False

    def _save_best(self):
        """Save best model and EMA weights (overwrite previous best)."""
        if dist.get_rank() != 0:
            return
        logdir = get_blob_logdir()
        # Save best model
        state_dict = self.mp_trainer.master_params_to_state_dict(
            self.mp_trainer.master_params
        )
        with bf.BlobFile(bf.join(logdir, "best_model.pt"), "wb") as f:
            th.save(state_dict, f)
        # Save best EMA
        for rate, params in zip(self.ema_rate, self.ema_params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            with bf.BlobFile(bf.join(logdir, f"best_ema_{rate}.pt"), "wb") as f:
                th.save(state_dict, f)
        logger.log(f"  → Best model checkpoint saved to {logdir}/best_model.pt")

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        # Rotate old checkpoints: keep only the last N
        if dist.get_rank() == 0 and self.keep_last_n_checkpoints > 0:
            self._rotate_checkpoints()

        dist.barrier()

    def _rotate_checkpoints(self):
        """Delete old periodic checkpoints, keeping only the last N sets."""
        logdir = get_blob_logdir()
        import re
        # Find all model checkpoint steps (exclude best_model.pt)
        steps = sorted(set(
            int(m.group(1))
            for f in bf.listdir(logdir)
            for m in [re.match(r"model(\d{6})\.pt$", f)]
            if m
        ))
        if len(steps) <= self.keep_last_n_checkpoints:
            return
        steps_to_delete = steps[:-self.keep_last_n_checkpoints]
        for s in steps_to_delete:
            for pattern in [
                f"model{s:06d}.pt",
                f"opt{s:06d}.pt",
            ]:
                path = bf.join(logdir, pattern)
                if bf.exists(path):
                    bf.remove(path)
            # Also remove EMA files for this step
            for rate in self.ema_rate:
                path = bf.join(logdir, f"ema_{rate}_{s:06d}.pt")
                if bf.exists(path):
                    bf.remove(path)
            logger.log(f"Rotated old checkpoint: step {s}")


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
