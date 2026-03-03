import os
import sys
from argparse import Namespace

import torchvision
import numpy as np
import torch
from torch.utils.data import DataLoader
import time

from tqdm import tqdm

# ensure project root on path for package imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from mapper.datasets.latents_dataset import EmbedDataset

from mapper.options.test_options import TestOptions
from mapper.styleclip_mapper import StyleCLIPMapper
from torchvision import transforms


def run(test_opts):
    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    os.makedirs(out_path_results, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts = Namespace(**opts)
    # print(opts)
    net = StyleCLIPMapper(opts)
    net.eval()
    net.cuda()

    dataset = EmbedDataset(embed_dir=opts.latents_test_path,
                                  opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=True)

    if opts.n_images is None:
        opts.n_images = len(dataset)
    
    global_i = 0
    global_time = []
    for input_batch in tqdm(dataloader):
        if global_i >= opts.n_images:
            break
        with torch.no_grad():
            input_cuda = input_batch
            

            tic = time.time()
            result_batch = run_on_batch(input_cuda, net)
            toc = time.time()
            global_time.append(toc - tic)

        for i in range(opts.test_batch_size):
            im_path = os.path.basename(dataset.embed_paths[global_i]).split('.')[0]
            # print('saving ', im_path)  # noisy with tqdm; keep commented
            save_path = os.path.join(out_path_results, f"{im_path}.png")

            if opts.img_root is not None:
                # load corresponding original image for side-by-side comparison
                from pathlib import Path
                from PIL import Image
                tfm = transforms.Compose([transforms.Resize((256,256)),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5]*3, [0.5]*3)])
                rel = Path(dataset.embed_paths[global_i]).relative_to(Path(opts.latents_test_path))
                img_path = None
                for ext in [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"]:
                    cand = Path(opts.img_root) / rel.with_suffix(ext)
                    if cand.exists():
                        img_path = cand
                        break
                if img_path is not None:
                    orig = tfm(Image.open(img_path).convert("RGB")).to(result_batch[0][i].device)
                    gen = result_batch[0][i]
                    if gen.shape[-1] != orig.shape[-1]:
                        gen = torch.nn.functional.interpolate(gen.unsqueeze(0),
                                                              size=orig.shape[-2:],
                                                              mode='bilinear',
                                                              align_corners=False).squeeze(0)
                    grid = torch.cat([orig.unsqueeze(0), gen.unsqueeze(0)], dim=0)
                    torchvision.utils.save_image(grid, save_path,
                                                 normalize=True, value_range=(-1, 1))
                else:
                    torchvision.utils.save_image(result_batch[0][i], save_path,
                                                 normalize=True, value_range=(-1, 1))
            else:
                torchvision.utils.save_image(result_batch[0][i], save_path,
                                             normalize=True, value_range=(-1, 1))
            #torch.save(result_batch[1][i].detach().cpu(), os.path.join(out_path_results, f"latent_{im_path}.pt"))

            global_i += 1

    stats_path = os.path.join(opts.exp_dir, 'stats.txt')
    result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
    print(result_str)

    with open(stats_path, 'w') as f:
        f.write(result_str)

def run_on_batch(inputs, net):
    embeds = inputs.to('cuda')
    with torch.no_grad():
        # keep the same forward path as training:
        # w = avg_latent + mapper(embed)
        avg_latent = net.latent_avg.detach()
        if avg_latent.ndim == 2:                # [n_styles, 512]
            avg_latent = avg_latent.unsqueeze(0)
        if avg_latent.shape[0] != embeds.shape[0]:
            avg_latent = avg_latent.repeat(embeds.shape[0], 1, 1)

        w_hat = avg_latent + net.mapper(embeds)

        x_hat, w_hat, _ = net.decoder([w_hat], input_is_latent=True, return_latents=True,
                                      randomize_noise=False, truncation=1)
        x_hat = net.face_pool(x_hat)  # 256x256, matches training logs

        result_batch = (x_hat, w_hat)
    return result_batch


if __name__ == '__main__':
	test_opts = TestOptions().parse()
	run(test_opts)
