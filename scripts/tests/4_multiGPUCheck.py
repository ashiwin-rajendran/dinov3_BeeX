import torch
import torch.distributed as dist

print(f"PyTorch distributed backend available: {torch.distributed.is_available()}")
print(f"NCCL available: {torch.distributed.is_nccl_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")

# Check FSDP (used by DINOv3 for large model sharding)
from torch.distributed.fsdp import FullyShardedDataParallel
print("FSDP import OK")