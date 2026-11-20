#!/usr/bin/env python3
"""PRO 5000 ×2 · TP=2 通信延迟探针
Qwen3.6-35B-A3B 的 MoE 在 TP=2 下：专家权重按 TP 切分，combine 阶段是 all-reduce 求和，
dispatch 阶段是 all-gather（token 分发）。token 级消息很小（单 token 2048 hidden × bf16 = 4 KB，
top_k=8 ⇒ 32 KB 量级），所以这里测 4 KB / 32 KB / 128 KB / 512 KB 四档。
用法： python -m torch.distributed.run --nproc_per_node=2 allreduce_lat.py
"""
import os, time, torch, torch.distributed as dist

SIZES = [4 * 1024, 32 * 1024, 128 * 1024, 512 * 1024]
WARM, ITER = 20, 200


def bench(op, nbytes):
    elems = nbytes // 2  # bf16
    x = torch.ones(elems, dtype=torch.bfloat16, device="cuda")
    for _ in range(WARM):
        (dist.all_reduce(x) if op == "all_reduce" else dist.all_gather_into_tensor(torch.empty(2 * elems, dtype=torch.bfloat16, device="cuda"), x))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(ITER):
        (dist.all_reduce(x) if op == "all_reduce" else dist.all_gather_into_tensor(torch.empty(2 * elems, dtype=torch.bfloat16, device="cuda"), x))
    torch.cuda.synchronize()
    us = (time.perf_counter() - t0) / ITER * 1e6
    return us


def main():
    rank = int(os.environ["RANK"]) if "RANK" in os.environ else 0
    world = int(os.environ.get("WORLD_SIZE", 2))
    dist.init_process_group(backend="nccl", world_size=world, rank=rank)
    torch.cuda.set_device(rank)
    if rank == 0:
        print("world=%d  dev=%s  nccl=%s" % (world, torch.cuda.get_device_name(rank), torch.cuda.nccl.version()), flush=True)
        print("%-8s %-12s %10s %10s" % ("size", "op", "latency_us", "GB/s"), flush=True)
    for n in SIZES:
        for op in ("all_reduce", "all_gather"):
            dist.barrier()
            us = bench(op, n) if True else 0
            if rank == 0:
                # all_reduce 搬 2(n-1)/n ≈ n bytes；all_gather 收 n bytes
                moved = n if op == "all_gather" else n * 2 * (world - 1) / world
                print("%-8d %-12s %10.1f %10.2f" % (n, op, us, moved / (us * 1e-6) / 1e9), flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
