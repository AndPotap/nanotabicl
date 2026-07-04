import time

import numpy as np
import torch

from prior import rand_dataset_filtered, rand_dataset_plain

n_repeats = 5
n_datasets = 10
n_samples = 1024
n_feats = 100
n_classes = 0
bsz = 512
# bsz = 512 // 8
n_skip = 2
seed = 21
n_steps = 100_000
# fn = [rand_dataset_plain, rand_dataset_filtered][0]
fn = [rand_dataset_plain, rand_dataset_filtered][-1]

np.random.seed(seed)
torch.manual_seed(seed)
all_times = np.empty(shape=(n_datasets, n_repeats))

for idx in range(n_repeats):
    for jdx in range(n_datasets):
        tic = time.perf_counter()

        tensors = fn(x_cat_sizes=[0] * n_feats, y_cat_sizes=[n_classes], n_samples=n_samples)
        x = torch.cat([tensors[f"x_{col}"] for col in range(n_feats)], dim=-1)
        y = tensors["y_0"].squeeze(-1)

        all_times[jdx, idx] = time.perf_counter() - tic

per_dataset_time = np.median(all_times[:, n_skip])
proj_time = bsz * n_steps * per_dataset_time
print(f"\nmedian: {np.median(np.sum(all_times[:, n_skip:], axis=0)):.3e} sec for {n_datasets=:,d}\n")
print(f"mean:   {np.mean(all_times[:, n_skip:]):.3e} sec")
print(f"std:    {np.std(all_times[:, n_skip:]):.3e} sec")

days, rem = divmod(int(proj_time), 3600 * 24)
hours, rem = divmod(rem, 3600)
mins, secs = divmod(rem, 60)
print(f"\n{proj_time=:.3e} sec")
print(f"{bsz=:,d} | {n_steps:,d} | {n_samples=:,d} | {n_feats=:,d}")
print(f"{days:,d}d {hours:,d}h {mins:,d}m {secs:.2f}s")

bytes_used = bsz * n_steps * n_samples * n_feats * (32 // 8)
gb = bytes_used / 1e9
tb = bytes_used / 1e12

print(f"{gb:.2f} GB ({tb:.3f} TB)")
