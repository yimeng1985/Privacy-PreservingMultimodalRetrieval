"""
Helpers for distributed training.
"""

import io
import os
import socket

import blobfile as bf
import torch as th
import torch.distributed as dist

# Change this to reflect your cluster layout.
# The GPU for a given rank is (rank % GPUS_PER_NODE).
GPUS_PER_NODE = 8

SETUP_RETRY_COUNT = 3


def _has_mpi():
    """Check if MPI is available and functional."""
    # Skip MPI if user explicitly wants single-GPU mode
    if os.environ.get("NO_MPI", ""):
        return False
    try:
        # Check if the MPI runtime is available before importing mpi4py
        import subprocess
        result = subprocess.run(
            ["ompi_info"], capture_output=True, timeout=5
        )
        if result.returncode != 0:
            return False
        from mpi4py import MPI
        _ = MPI.COMM_WORLD.Get_rank()
        return True
    except Exception:
        return False


def setup_dist():
    """
    Setup a distributed process group.
    Supports both MPI-based and single-GPU (torchrun/env-based) setups.
    """
    if dist.is_initialized():
        return

    if _has_mpi():
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{comm.Get_rank() % GPUS_PER_NODE}"

        backend = "gloo" if not th.cuda.is_available() else "nccl"

        if backend == "gloo":
            hostname = "localhost"
        else:
            hostname = socket.gethostbyname(socket.getfqdn())
        os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
        os.environ["RANK"] = str(comm.rank)
        os.environ["WORLD_SIZE"] = str(comm.size)

        port = comm.bcast(_find_free_port(), root=0)
        os.environ["MASTER_PORT"] = str(port)
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        # Single-GPU or torchrun-based setup
        backend = "gloo" if not th.cuda.is_available() else "nccl"
        if "RANK" not in os.environ:
            os.environ["RANK"] = "0"
        if "WORLD_SIZE" not in os.environ:
            os.environ["WORLD_SIZE"] = "1"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "localhost"
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(_find_free_port())
        dist.init_process_group(backend=backend, init_method="env://")


def dev():
    """
    Get the device to use for torch.distributed.
    """
    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, **kwargs):
    """
    Load a PyTorch file without redundant fetches across MPI ranks.
    """
    if _has_mpi():
        from mpi4py import MPI
        chunk_size = 2 ** 30  # MPI has a relatively small size limit
        if MPI.COMM_WORLD.Get_rank() == 0:
            with bf.BlobFile(path, "rb") as f:
                data = f.read()
            num_chunks = len(data) // chunk_size
            if len(data) % chunk_size:
                num_chunks += 1
            MPI.COMM_WORLD.bcast(num_chunks)
            for i in range(0, len(data), chunk_size):
                MPI.COMM_WORLD.bcast(data[i : i + chunk_size])
        else:
            num_chunks = MPI.COMM_WORLD.bcast(None)
            data = bytes()
            for _ in range(num_chunks):
                data += MPI.COMM_WORLD.bcast(None)
    else:
        with bf.BlobFile(path, "rb") as f:
            data = f.read()

    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
