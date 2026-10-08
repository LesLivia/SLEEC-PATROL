# Feature extraction module
"""Extract frozen vision-backbone features for BDD-OIA, in the same layout as data/bdd2048.

Run from the rss/ folder, for example:
    python -m utils.extract_features --model vit_large_patch16_dinov3.lvd1689m --out data/dinov3_l_576 --height 576 --width 1024

For every frame listed in the rsbench split files, the full 1280x720 image (no crop) is resized to
--height x --width and passed through the timm model. The output tokens are pooled into
    [global token | mean of each grid cell | max of each grid cell]
and saved as <out>/<split>/inputs/<frame>.pt, a (1, D) float32 tensor like the original
data/bdd2048 inputs. labels/, concepts/ and the split .pkl files are copied from --src, and the
settings are written to <out>/meta.json, whose "dim" sets the concept extractor's input size
(datasets/boia.py). Existing outputs are skipped, so an interrupted run can be resumed.
"""

import argparse
import datetime
import json
import os
import pickle
import random
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SPLITS = ["train", "val", "test"]


def parse_args():
    """Parse the command line arguments

    Returns:
        args: parsed arguments
    """
    parser = argparse.ArgumentParser(description="Extract BOIA image features with a timm backbone.")
    parser.add_argument("--model", default="vit_large_patch16_dinov3.lvd1689m", help="timm model name")
    parser.add_argument("--out", required=True, help="output folder, e.g. data/dinov3_l_576")
    parser.add_argument("--src", default="data/bdd2048", help="folder with the split .pkl files, labels/ and concepts/")
    parser.add_argument("--images", default="data/BDD-OIA-lastframe/lastframe/data", help="folder with the original .jpg frames")
    parser.add_argument("--height", type=int, default=576, help="input height, a multiple of the patch size")
    parser.add_argument("--width", type=int, default=1024, help="input width, a multiple of the patch size")
    parser.add_argument("--grid", default="2x3", help="pooling grid ROWSxCOLS (rows: far/near, cols: left/centre/right)")
    parser.add_argument("--batch", type=int, default=8, help="batch size")
    parser.add_argument("--workers", type=int, default=2, help="image loading workers")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N frames of each split (trial run)")
    parser.add_argument("--overwrite", action="store_true", help="recompute features that already exist")
    return parser.parse_args()


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


def frame_path(root, split, sub, name):
    """Path of a per-frame .pt file, named like the rsbench files (<frame>.pt)"""
    return os.path.join(root, split, sub, name[:-4] + ".pt")


def mirror_split(args, split):
    """Write the split .pkl and copy labels/ and concepts/ from --src

    Returns:
        names: frame file names of the split (first --limit if given)
    """
    entries = pickle.load(open(os.path.join(args.src, f"{split}_BDD_OIA.pkl"), "rb"))
    if args.limit:
        entries = entries[: args.limit]
    names = [e["img_path"] for e in entries]
    for sub in ["inputs", "labels", "concepts"]:
        os.makedirs(os.path.join(args.out, split, sub), exist_ok=True)
    with open(os.path.join(args.out, f"{split}_BDD_OIA.pkl"), "wb") as f:
        pickle.dump(entries, f)
    for name in names:
        for sub in ["labels", "concepts"]:
            dst = frame_path(args.out, split, sub, name)
            if not os.path.exists(dst):
                shutil.copyfile(frame_path(args.src, split, sub, name), dst)
    return names


@torch.no_grad()
def extract(model, x, n_prefix, token_grid, cells):
    """Features of one batch of normalised images"""
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        tokens = model.forward_features(x.cuda(non_blocking=True))
    return pool_tokens(tokens, n_prefix, token_grid, cells).cpu()


def verify(args, split_names, dim, model, n_prefix, token_grid, cells, data_cfg):
    """Check counts, shapes, finiteness and reproducibility of the written features

    Returns:
        problems (list of str), max abs difference of recomputed features
    """
    problems = []
    for split, names in split_names.items():
        for sub in ["inputs", "labels", "concepts"]:
            missing = sum(not os.path.exists(frame_path(args.out, split, sub, n)) for n in names)
            if missing:
                problems.append(f"{split}/{sub}: {missing} of {len(names)} files missing")
    all_inputs = [frame_path(args.out, s, "inputs", n) for s, ns in split_names.items() for n in ns]
    for path in random.Random(0).sample(all_inputs, min(20, len(all_inputs))):
        t = torch.load(path)
        if tuple(t.shape) != (1, dim) or t.dtype != torch.float32 or not torch.isfinite(t).all():
            problems.append(f"bad tensor {path}: shape {tuple(t.shape)}, dtype {t.dtype}")
    names = split_names["train"][:4]
    ds = FrameDataset(names, args.images, args.height, args.width, data_cfg["mean"], data_cfg["std"])
    again = extract(model, torch.stack([ds[i][0] for i in range(len(names))]), n_prefix, token_grid, cells)
    saved = torch.cat([torch.load(frame_path(args.out, "train", "inputs", n)) for n in names])
    return problems, (again - saved).abs().max().item()


def main():
    args = parse_args()
    rows, cols = (int(v) for v in args.grid.lower().split("x"))

    # a resumed run must use the same settings, otherwise old and new features would be mixed
    os.makedirs(args.out, exist_ok=True)
    settings = {"model": args.model, "height": args.height, "width": args.width,
                "grid": [rows, cols], "src": args.src, "images": args.images}
    settings_path = os.path.join(args.out, "extraction_settings.json")
    if os.path.exists(settings_path) and not args.overwrite:
        with open(settings_path) as f:
            previous = json.load(f)
        if previous != settings:
            raise SystemExit(f"{args.out} was extracted with different settings:\n  {previous}\nnow:\n  {settings}\n"
                             "Use another --out folder, or --overwrite to recompute everything.")
    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)

    print(f"Loading {args.model} (the first run downloads the weights) ...", flush=True)
    model, data_cfg, timm_version = load_model(args.model)
    ph, pw = model.patch_embed.patch_size
    assert args.height % ph == 0 and args.width % pw == 0, f"--height/--width must be multiples of the patch size {ph}x{pw}"
    token_grid = (args.height // ph, args.width // pw)
    n_prefix = model.num_prefix_tokens
    embed_dim = model.num_features
    dim = (1 + 2 * rows * cols) * embed_dim
    print(f"input {args.height}x{args.width} | patch {ph}x{pw} -> token grid {token_grid[0]}x{token_grid[1]} | "
          f"prefix tokens {n_prefix} | embed dim {embed_dim} | feature dim {dim} | "
          f"mean {data_cfg['mean']} std {data_cfg['std']}", flush=True)

    split_names = {}
    for split in SPLITS:
        names = mirror_split(args, split)
        split_names[split] = names
        todo = [n for n in names if args.overwrite or not os.path.exists(frame_path(args.out, split, "inputs", n))]
        print(f"{split}: {len(names)} frames, {len(names) - len(todo)} already done, {len(todo)} to extract", flush=True)
        if not todo:
            continue
        ds = FrameDataset(todo, args.images, args.height, args.width, data_cfg["mean"], data_cfg["std"])
        loader = torch.utils.data.DataLoader(ds, batch_size=args.batch, num_workers=args.workers, pin_memory=True)
        start, done = time.time(), 0
        for b, (x, idx) in enumerate(loader):
            feats = extract(model, x, n_prefix, token_grid, (rows, cols))
            if not torch.isfinite(feats).all():
                raise RuntimeError(f"non-finite features in {split} batch {b}")
            for f, j in zip(feats, idx.tolist()):
                # clone: saving a view would store the whole batch tensor in every file;
                # write then rename, so an interrupted run never leaves a partial file behind
                dst = frame_path(args.out, split, "inputs", todo[j])
                torch.save(f.unsqueeze(0).clone(), dst + ".tmp")
                os.replace(dst + ".tmp", dst)
            done += len(idx)
            if b % 50 == 0 or done == len(todo):
                rate = done / (time.time() - start)
                print(f"  {split}: {done}/{len(todo)} | {rate:.1f} img/s | ETA {(len(todo) - done) / rate / 60:.1f} min", flush=True)

    problems, max_diff = verify(args, split_names, dim, model, n_prefix, token_grid, (rows, cols), data_cfg)
    complete = not problems and args.limit is None
    meta = {
        "model": args.model,
        "timm_version": timm_version,
        "torch_version": torch.__version__,
        "height": args.height,
        "width": args.width,
        "resize": "PIL bicubic, full frame, no crop",
        "patch_size": [ph, pw],
        "token_grid": list(token_grid),
        "prefix_tokens": n_prefix,
        "embed_dim": embed_dim,
        "grid": [rows, cols],
        "dim": dim,
        "layout": f"global token ({embed_dim}) | mean of each {rows}x{cols} grid cell (row-major: far to near, "
                  f"left to right; {rows * cols}x{embed_dim}) | max of each grid cell ({rows * cols}x{embed_dim})",
        "mean": list(data_cfg["mean"]),
        "std": list(data_cfg["std"]),
        "source_images": args.images,
        "source_labels": args.src,
        "limit": args.limit,
        "counts": {s: len(n) for s, n in split_names.items()},
        "recompute_max_abs_diff": max_diff,
        "complete": complete,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"recomputed features max abs difference: {max_diff:.2e}")
    for p in problems:
        print("PROBLEM:", p)
    print(f"{'complete' if complete else 'NOT complete'}: {args.out} (meta.json written)")


if __name__ == "__main__":
    main()
