"""
Anima transformer key diagnostic.

Compares the keys in a checkpoint against the model architecture built by
build_anima(), reporting missing keys, unexpected keys, and shape mismatches.

Usage:
    python testing/test_anima_transformer.py --ckpt /path/to/anima.safetensors
"""

import argparse
import os
import sys

import torch
from safetensors.torch import load_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import only what we need - avoids the heavy extension chain
from extensions_built_in.diffusion_models.anima.model import build_anima


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to anima transformer .safetensors")
    args = parser.parse_args()

    print(f"\n=== Building Anima architecture ===")
    model = build_anima(dtype=torch.float32, device="cpu")
    model_sd = model.state_dict()
    model_keys = set(model_sd.keys())
    print(f"  Architecture parameters: {len(model_keys)}")
    print(f"  Total params: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\n=== Loading checkpoint: {args.ckpt} ===")
    raw_sd = load_file(args.ckpt, device="cpu")
    print(f"  Checkpoint keys (raw): {len(raw_sd)}")

    # Strip the 'net.' prefix that Anima checkpoints use
    stripped = {}
    n_stripped = 0
    for k, v in raw_sd.items():
        if k.startswith("net."):
            stripped[k[4:]] = v
            n_stripped += 1
        else:
            stripped[k] = v
    print(f"  Keys after stripping 'net.' prefix: {len(stripped)} ({n_stripped} had prefix)")

    ckpt_keys = set(stripped.keys())

    missing    = model_keys - ckpt_keys      # model expects but ckpt doesn't have
    unexpected = ckpt_keys  - model_keys     # ckpt has but model doesn't expect

    print(f"\n=== Key comparison ===")
    print(f"  Missing   (in model, not in ckpt): {len(missing)}")
    print(f"  Unexpected (in ckpt, not in model): {len(unexpected)}")

    if missing:
        print(f"\n  --- MISSING KEYS (first 30) ---")
        for k in sorted(missing)[:30]:
            print(f"    {k}  [model shape: {tuple(model_sd[k].shape)}]")
        if len(missing) > 30:
            print(f"    ... and {len(missing) - 30} more")

    if unexpected:
        print(f"\n  --- UNEXPECTED KEYS (first 30) ---")
        for k in sorted(unexpected)[:30]:
            print(f"    {k}  [ckpt shape: {tuple(stripped[k].shape)}]")
        if len(unexpected) > 30:
            print(f"    ... and {len(unexpected) - 30} more")

    # Check shape mismatches on keys present in both
    shared = model_keys & ckpt_keys
    shape_mismatches = []
    for k in shared:
        ms = tuple(model_sd[k].shape)
        cs = tuple(stripped[k].shape)
        if ms != cs:
            shape_mismatches.append((k, ms, cs))

    print(f"\n  Shape mismatches (same key, different shape): {len(shape_mismatches)}")
    if shape_mismatches:
        print(f"\n  --- SHAPE MISMATCHES ---")
        for k, ms, cs in shape_mismatches[:30]:
            print(f"    {k}: model={ms}  ckpt={cs}")

    print(f"\n=== Summary ===")
    if not missing and not unexpected and not shape_mismatches:
        print("  PASS: Architecture matches checkpoint exactly.")
    else:
        if missing:
            print(f"  FAIL: {len(missing)} keys missing from checkpoint -- those layers use random weights.")
        if unexpected:
            print(f"  WARN: {len(unexpected)} unexpected keys in checkpoint (ignored on load).")
        if shape_mismatches:
            print(f"  FAIL: {len(shape_mismatches)} shape mismatches -- those layers load wrong weights.")

    # Show which top-level modules have issues
    if missing or unexpected:
        print(f"\n=== Affected top-level modules ===")
        all_bad = missing | unexpected
        modules = {}
        for k in all_bad:
            top = k.split(".")[0]
            modules[top] = modules.get(top, 0) + 1
        for mod, count in sorted(modules.items(), key=lambda x: -x[1]):
            tag = "MISSING" if any(k.startswith(mod+".") for k in missing) else "UNEXPECTED"
            print(f"  {mod}: {count} keys ({tag})")


if __name__ == "__main__":
    main()
