#!/usr/bin/env python3
"""
Bump the learning rate in a training checkpoint to escape plateaus.
"""

import torch
import argparse
from pathlib import Path


def bump_lr(checkpoint_path: str, new_lr: float, reset_scheduler: bool = True):
    """
    Modify the learning rate in a checkpoint file.

    Args:
        checkpoint_path: Path to checkpoint.pth
        new_lr: New learning rate to set
        reset_scheduler: If True, also reset scheduler state
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        return False

    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Show current state
    print(f"\nCurrent state:")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Best Val Acc: {checkpoint.get('best_val_acc', 'unknown'):.2f}%")

    # Get current LR from optimizer
    if 'optimizer_state_dict' in checkpoint:
        for i, pg in enumerate(checkpoint['optimizer_state_dict']['param_groups']):
            print(f"  Optimizer LR (group {i}): {pg['lr']:.6f}")

    # Update optimizer LR
    print(f"\nUpdating LR to: {new_lr:.6f}")
    if 'optimizer_state_dict' in checkpoint:
        for pg in checkpoint['optimizer_state_dict']['param_groups']:
            pg['lr'] = new_lr
            pg['initial_lr'] = new_lr  # Also update initial_lr for scheduler

    # Optionally reset scheduler
    if reset_scheduler and 'scheduler_state_dict' in checkpoint:
        print("Resetting scheduler state (will restart cosine annealing)")
        # For SequentialLR with warmup + cosine, we need to reset carefully
        # Just remove the scheduler state so it reinitializes
        del checkpoint['scheduler_state_dict']

    # Save modified checkpoint
    backup_path = checkpoint_path.with_suffix('.pth.bak')
    print(f"\nBacking up original to: {backup_path}")
    import shutil
    shutil.copy2(checkpoint_path, backup_path)

    print(f"Saving modified checkpoint to: {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)

    print("\nDone! Resume training to continue with new LR.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Bump learning rate in checkpoint")
    parser.add_argument("checkpoint", help="Path to checkpoint.pth")
    parser.add_argument("--lr", type=float, default=0.0005,
                        help="New learning rate (default: 0.0005)")
    parser.add_argument("--keep-scheduler", action="store_true",
                        help="Keep scheduler state (don't reset)")

    args = parser.parse_args()

    bump_lr(args.checkpoint, args.lr, reset_scheduler=not args.keep_scheduler)


if __name__ == "__main__":
    main()
