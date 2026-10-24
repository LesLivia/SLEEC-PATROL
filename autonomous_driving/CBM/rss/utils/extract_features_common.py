# Common feature extraction functions
"""Dataset-independent parts of the feature extraction, used by extract_features_bdd_oia.py and
extract_features_sdd_oia.py: the timm backbone, the token pooling
    [global token | mean of each grid cell | max of each grid cell],
the extraction loop that writes one feature file per image, the guard of the output folder (a finished
extraction is never written again; an interrupted one is resumed only with the same settings), the copy
of the label files, the reproducibility check and the meta.json fields shared by both datasets.
"""

import datetime
import json
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def add_common_args(parser):
    """Add the options shared by every dataset to an argument parser"""
    parser.add_argument("--model", default="vit_large_patch16_dinov3.lvd1689m", help="timm model name")
    parser.add_argument("--grid", default="2x3", help="pooling grid ROWSxCOLS (rows: far/near, cols: left/centre/right)")
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--workers", type=int, default=2, help="image loading workers")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N images of each split (trial run)")


class FrameDataset(torch.utils.data.Dataset):
    """Loads full frames, resizes them without cropping and normalises them"""

    def __init__(self, names, image_dir, height, width, mean, std):
        self.names = names
        self.image_dir = image_dir
        self.size = (width, height)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.image_dir, self.names[idx])).convert("RGB")
        img = img.resize(self.size, Image.BICUBIC)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return (x - self.mean) / self.std, idx


def load_model(name):
    """Create the pretrained timm model (on the GPU) and its normalisation config

    Args:
        name: timm model name

    Returns:
        model, data config, timm version
    """
    import timm
    from timm.data import resolve_model_data_config

    model = timm.create_model(name, pretrained=True, dynamic_img_size=True, num_classes=0).cuda().eval()
    return model, resolve_model_data_config(model), timm.__version__


def pool_tokens(tokens, n_prefix, token_grid, cells):
    """Pool the output tokens into [global | mean per grid cell | max per grid cell]

    Args:
        tokens: (B, n_prefix + gh * gw, D) output of forward_features
        n_prefix: number of leading non-patch tokens (class token first, then registers)
        token_grid: (gh, gw) patch-token grid
        cells: (rows, cols) pooling grid

    Returns:
        features: (B, (1 + 2 * rows * cols) * D) float32
    """
    B, _, D = tokens.shape
    gh, gw = token_grid
    patches = tokens[:, n_prefix:].float()
    assert patches.shape[1] == gh * gw, (tuple(tokens.shape), token_grid, n_prefix)
    glob = tokens[:, 0].float()
    fmap = patches.reshape(B, gh, gw, D).permute(0, 3, 1, 2)  # (B, D, gh, gw)
    avg = F.adaptive_avg_pool2d(fmap, cells).flatten(2).transpose(1, 2)  # (B, cells, D), row-major
    mx = F.adaptive_max_pool2d(fmap, cells).flatten(2).transpose(1, 2)
    return torch.cat([glob, avg.reshape(B, -1), mx.reshape(B, -1)], dim=1)


@torch.no_grad()
def extract(model, x, n_prefix, token_grid, cells):
    """Features of one batch of normalised images"""
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        tokens = model.forward_features(x.cuda(non_blocking=True))
    return pool_tokens(tokens, n_prefix, token_grid, cells).cpu()


def check_settings(out, settings):
    """Stop if the output folder holds a finished extraction, otherwise record the settings

    A folder with meta.json holds a finished extraction and is never written again, so the features of
    different runs are never mixed. A folder without meta.json holds an interrupted run, which is resumed
    only with exactly the same settings.
    """
    if os.path.exists(os.path.join(out, "meta.json")):
        raise SystemExit(f"{out} already contains a finished extraction (meta.json).\n"
                         "Use a new --out folder, or delete this one to start again.")
    os.makedirs(out, exist_ok=True)
    settings_path = os.path.join(out, "extraction_settings.json")
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            previous = json.load(f)
        if previous != settings:
            raise SystemExit(f"{out} holds an interrupted extraction with different settings:\n  {previous}\n"
                             f"now:\n  {settings}\nUse a new --out folder, or delete this one to start again.")
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)


def copy_file(src, dst):
    """Copy a label file; write then rename, so an interrupted run never leaves a partial copy behind"""
    shutil.copyfile(src, dst + ".tmp")
    os.replace(dst + ".tmp", dst)


def backbone_info(model, data_cfg, height, width, cells):
    """Token grid and feature size of the backbone at the given input size (printed for the log)

    Returns:
        dict with patch_size, token_grid, prefix_tokens, embed_dim, cells, dim, mean and std
    """
    ph, pw = model.patch_embed.patch_size
    assert height % ph == 0 and width % pw == 0, f"--height/--width must be multiples of the patch size {ph}x{pw}"
    rows, cols = cells
    info = {
        "patch_size": (ph, pw),
        "token_grid": (height // ph, width // pw),
        "prefix_tokens": model.num_prefix_tokens,
        "embed_dim": model.num_features,
        "cells": (rows, cols),
        "dim": (1 + 2 * rows * cols) * model.num_features,
        "mean": data_cfg["mean"],
        "std": data_cfg["std"],
    }
    print(f"input {height}x{width} | patch {ph}x{pw} -> token grid {info['token_grid'][0]}x{info['token_grid'][1]} | "
          f"prefix tokens {info['prefix_tokens']} | embed dim {info['embed_dim']} | feature dim {info['dim']} | "
          f"mean {info['mean']} std {info['std']}", flush=True)
    return info


def extract_to_files(model, info, args, split, names, image_dir, dst_of, keep_batch_dim):
    """Extract the features of the given images and write one file per image

    Images whose file already exists are skipped, so an interrupted run can be resumed.

    Args:
        names: image file names, in image_dir
        dst_of: function mapping an image name to its feature file path
        keep_batch_dim: save each vector as (1, D) instead of (D,)
    """
    todo = [n for n in names if not os.path.exists(dst_of(n))]
    print(f"{split}: {len(names)} frames, {len(names) - len(todo)} already done, {len(todo)} to extract", flush=True)
    if not todo:
        return
    ds = FrameDataset(todo, image_dir, args.height, args.width, info["mean"], info["std"])
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers, pin_memory=True)
    start, done = time.time(), 0
    for b, (x, idx) in enumerate(loader):
        feats = extract(model, x, info["prefix_tokens"], info["token_grid"], info["cells"])
        if not torch.isfinite(feats).all():
            raise RuntimeError(f"non-finite features in {split} batch {b}")
        for f, j in zip(feats, idx.tolist()):
            # clone: saving a view would store the whole batch tensor in every file;
            # write then rename, so an interrupted run never leaves a partial file behind
            dst = dst_of(todo[j])
            torch.save((f.unsqueeze(0) if keep_batch_dim else f).clone(), dst + ".tmp")
            os.replace(dst + ".tmp", dst)
        done += len(idx)
        if b % 50 == 0 or done == len(todo):
            rate = done / (time.time() - start)
            print(f"  {split}: {done}/{len(todo)} | {rate:.1f} img/s | ETA {(len(todo) - done) / rate / 60:.1f} min", flush=True)


def check_tensors(paths, shape):
    """Shape, dtype and finiteness of 20 randomly chosen feature files

    Returns:
        problems (list of str)
    """
    problems = []
    for path in random.Random(0).sample(paths, min(20, len(paths))):
        t = torch.load(path)
        if tuple(t.shape) != shape or t.dtype != torch.float32 or not torch.isfinite(t).all():
            problems.append(f"bad tensor {path}: shape {tuple(t.shape)}, dtype {t.dtype}")
    return problems


def recompute_max_diff(model, info, args, names, image_dir, paths):
    """Recompute the features of a few images and compare them with their stored files

    Returns:
        max abs difference between recomputed and stored features
    """
    ds = FrameDataset(names, image_dir, args.height, args.width, info["mean"], info["std"])
    again = extract(model, torch.stack([ds[i][0] for i in range(len(names))]),
                    info["prefix_tokens"], info["token_grid"], info["cells"])
    saved = torch.stack([torch.load(p).reshape(-1) for p in paths])
    return (again - saved).abs().max().item()


def base_meta(args, info, timm_version):
    """meta.json fields shared by every dataset: the backbone, the input size and the feature layout"""
    rows, cols = info["cells"]
    embed_dim = info["embed_dim"]
    return {
        "model": args.model,
        "timm_version": timm_version,
        "torch_version": torch.__version__,
        "height": args.height,
        "width": args.width,
        "resize": "PIL bicubic, full frame, no crop",
        "patch_size": list(info["patch_size"]),
        "token_grid": list(info["token_grid"]),
        "prefix_tokens": info["prefix_tokens"],
        "embed_dim": embed_dim,
        "grid": [rows, cols],
        "dim": info["dim"],
        "layout": f"global token ({embed_dim}) | mean of each {rows}x{cols} grid cell (row-major: far to near, "
                  f"left to right; {rows * cols}x{embed_dim}) | max of each grid cell ({rows * cols}x{embed_dim})",
        "mean": list(info["mean"]),
        "std": list(info["std"]),
    }


def write_meta(out, meta):
    """Write meta.json with its creation time"""
    meta["created"] = datetime.datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
