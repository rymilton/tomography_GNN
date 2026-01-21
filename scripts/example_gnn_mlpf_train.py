#!/usr/bin/env python3
"""Small end-to-end example using the repository's MLPF GNN (gnn_lsh) with synthetic data.

Runs a short training loop on CPU (or GPU if available) and saves a checkpoint.
"""
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torch.optim as optim

from mlpf.model.mlpf import MLPF
from mlpf.model.PFDataset import PFBatch
from mlpf.model.utils import unpack_predictions, unpack_target, save_checkpoint
from mlpf.model.losses import mlpf_loss

import numpy as np
def make_synthetic_batch(batch_size=4, num_muons=16, num_voxels=10, num_input_features=10, device="cpu"):
    # The input to the model is going to be a batch of muons (padded to num_muons) and their features
    # X feature tensor: (batch, num_muons, num_input_features)
    X = torch.zeros((batch_size, num_muons, num_input_features), dtype=torch.float32, device=device)
    # mark elements as present
    X[..., 0] = 1.0
    # set pt to 1.0 (feature idx 1)
    X[..., 1] = 1.0
    # sin_phi (idx 3) = 0, cos_phi (idx 4) = 1
    X[..., 3] = 0.0
    X[..., 4] = 1.0
    # energy (idx 5)
    X[..., 5] = 1.0

    # The target will be the voxel densities for all the muons and voxels.
    # y feature tensor: (batch, num_voxels)
    y = torch.zeros((batch_size, num_voxels), dtype=torch.float32, device=device)
    y[..., 0] = 1
    y[..., 1] = 0
    y[..., 2] = 0.0
    y[..., 3] = 0.0
    y[..., 4] = 0.0
    y[..., 5] = 1.0
    y[..., 6] = 0.0
    y[..., 7] = 0.0
    y[..., 8] = 0.0

    muon_mask = torch.ones((batch_size, num_muons,), device=device, dtype=torch.bool)
    voxel_mask = torch.ones((batch_size, num_muons, num_voxels,), device=device, dtype=torch.bool)
    path_length = torch.ones((batch_size, num_muons, num_voxels,), device=device, dtype=torch.float)

    return X, y, muon_mask, voxel_mask, path_length


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    MAX_NUMBER_OF_VOXELS = 10  # adjust as needed
    NUMBER_OF_INPUT_FEATURES = 10  # adjust as needed
    # small model config for quick runs
    model = MLPF(
        input_dim=NUMBER_OF_INPUT_FEATURES,
        embedding_dim=32,
        width=32,
        max_num_voxels=MAX_NUMBER_OF_VOXELS,
        num_convs=1,
        conv_type="gnn_lsh",
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_steps = 20
    batch_size = 4
    seq_len = 16

    for step in range(1, num_steps + 1):
        model.train()
        X, y, muon_mask, voxel_mask, path_length = make_synthetic_batch(batch_size=batch_size, num_muons=seq_len, num_voxels=MAX_NUMBER_OF_VOXELS, num_input_features=NUMBER_OF_INPUT_FEATURES, device=device)
        # forward
        print("X shape:", X.shape)
        print(voxel_mask.shape)
        preds = model(X, muon_mask, path_length=path_length)

        loss_opt = mlpf_loss(y, preds)

        optimizer.zero_grad()
        loss_opt.backward()
        optimizer.step()

        if step % 5 == 0 or step == 1:
            print(f"Step {step}/{num_steps}: Total loss = {loss_opt.item():.6f}")

    # run a short eval pass
    model.eval()
    with torch.no_grad():
        X, y, muon_mask, voxel_mask, path_length = make_synthetic_batch(batch_size=2, num_muons=seq_len, num_voxels=MAX_NUMBER_OF_VOXELS, num_input_features=NUMBER_OF_INPUT_FEATURES, device=device)
        preds = model(X, muon_mask, path_length=path_length)
        val_losses = mlpf_loss(y, preds)
        print("Validation Total loss:", val_losses.item())

    # save checkpoint
    outdir = Path("experiments/example_gnn_run")
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_path = outdir / "checkpoint.pth"
    save_checkpoint(str(ckpt_path), model, optimizer, extra_state={"step": num_steps})
    print("Saved checkpoint to", ckpt_path)


if __name__ == "__main__":
    main()
