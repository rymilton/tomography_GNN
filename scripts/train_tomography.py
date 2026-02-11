import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torch.optim as optim

import pickle
import argparse
from mlpf.model.mlpf import MLPF
from mlpf.model.losses import mlpf_loss
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import math, time, yaml, pickle, argparse
from torch.utils.data import Dataset
import pandas as pd


# Class to read each row in the pandas dataframe and get the voxel densities
class MuonDataset(Dataset):
    def __init__(self, muon_dataframe, feature_columns):
        self.df = muon_dataframe.reset_index(drop=True)
        self.feature_columns = feature_columns

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        muon = self.df.iloc[idx]

        # Features
        features = torch.tensor(
            [muon[col] for col in self.feature_columns], dtype=torch.float
        )

        # Voxel indices and path lengths
        voxel_indices = torch.tensor(muon["voxel_indices_1d"], dtype=torch.long)
        path_lengths = torch.tensor(muon["lengths_in_voxels"], dtype=torch.float)
        object_id = int(muon["object_ID"])

        return features, voxel_indices, path_lengths, object_id


# Function to group together a list of muons into groups of muons
def muon_group_collate_fn(
    batch,
    voxel_densities_object,
    voxel_densities_noobject,
    num_voxels,
    device,
    num_muons_per_group,
):
    """
    Groups a flat list of muons into groups of size num_muons_per_group
    """
    outputs = []

    def process_subbatch(subbatch, voxel_densities):
        if len(subbatch) == 0 or voxel_densities is None:
            return None

        features_list, voxel_indices_list, path_lengths_list, _ = zip(*subbatch)
        # features_list is a list of length 1418. Each entry is a tuple of 8 features

        # Total number of muons in the subbatch
        total_muons = len(subbatch)
        num_groups = max(1, math.ceil(total_muons / num_muons_per_group))

        # The number of muons that should be in the group if each group were full
        total_batch_len = num_groups * num_muons_per_group
        feature_dim = features_list[0].shape[0]

        # Adding extra muons with all zeros to make each subbatch len total_batch_len
        pad_feature = torch.zeros(feature_dim)
        features_list = list(features_list)
        features_list.extend([pad_feature] * (total_batch_len - total_muons))

        X = (
            torch.stack(features_list)
            .view(num_groups, num_muons_per_group, feature_dim)
            .to(device)
        )

        muon_mask = torch.zeros(total_batch_len, dtype=torch.bool, device=device)
        muon_mask[:total_muons] = True
        muon_mask = muon_mask.view(num_groups, num_muons_per_group)

        voxel_mask = torch.zeros(
            num_groups, num_muons_per_group, num_voxels, dtype=torch.bool, device=device
        )
        path_length = torch.zeros(
            num_groups,
            num_muons_per_group,
            num_voxels,
            dtype=torch.float,
            device=device,
        )

        for g in range(num_groups):
            for m in range(num_muons_per_group):
                i = g * num_muons_per_group + m
                if i >= total_muons:
                    break
                vox = voxel_indices_list[i]
                lengths = path_lengths_list[i]
                voxel_mask[g, m, vox] = True
                path_length[g, m, vox] = lengths.to(device)

        y = torch.stack([torch.tensor(voxel_densities, device=device)] * num_groups)

        return X, y, muon_mask, voxel_mask, path_length

    # --- Always process object muons ---
    batch_object = [b for b in batch if b[3] == 1]
    # process_subpatch returns a tuple of muon features, true densities, muon mask, voxel mask, path length
    out_obj = process_subbatch(batch_object, voxel_densities_object)
    if out_obj is not None:
        outputs.append(out_obj)

    batch_noobject = [b for b in batch if b[3] == 0]
    out_noobj = process_subbatch(batch_noobject, voxel_densities_noobject)
    if out_noobj is not None:
        outputs.append(out_noobj)

    return outputs


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_object_file",
        default="/home/ryan/tomography_GNN/preprocessed_data_train.pkl",
        help="Input pickle file with object and reconstructed theta, phi angles, and the voxel interesections",
        type=str,
    )
    parser.add_argument(
        "--input_free_file",
        default="/home/ryan/tomography_GNN/preprocessed_data_train.pkl",
        help="Input pickle file without object and reconstructed theta, phi angles, and the voxel interesections",
        type=str,
    )
    parser.add_argument(
        "--config",
        default="/home/ryan/tomography_GNN/scripts/training_config.yaml",
        help="Path of config file",
        type=str,
    )
    parser.add_argument(
        "--num_muons",
        default=None,
        help="Number of muons to use during training",
        type=int,
    )
    parser.add_argument(
        "--free_and_object",
        default=False,
        help="Enable if you want to use both data with object and without object",
        action="store_true",
    )
    return parser.parse_args()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    flags = parse_arguments()

    # Opening config file and storing some commonly used options
    config = yaml.safe_load(open(flags.config))
    muon_feature_names = config["MUON_FEATURES"]
    num_muon_features = len(muon_feature_names)
    num_muon_groups_per_batch = config.get("NUM_MUON_GROUPS_PER_BATCH", 64)
    num_muons_per_group = config.get("NUM_MUONS_PER_GROUP", 64)

    # Open input data file
    with open(flags.input_object_file, "rb") as f:
        object_data = pickle.load(f)

    if flags.free_and_object:

        with open(flags.input_free_file, "rb") as f:
            free_data = pickle.load(f)

        # Balancing free and object events
        free_df = free_data["dataframe"]
        object_df = object_data["dataframe"]

        # Find the minimum count
        min_count = min(len(free_df), len(object_df))

        # Sample that many from each class
        free_df_balanced = free_df.sample(n=min_count, random_state=42)
        object_df_balanced = object_df.sample(n=min_count, random_state=42)

        # Combine them back into one DataFrame
        muon_dataframe_balanced = pd.concat(
            [free_df_balanced, object_df_balanced], ignore_index=True
        )

        max_num_muons = (
            flags.num_muons
            if flags.num_muons is not None
            else len(muon_dataframe_balanced)
        )

        # Shuffle rows so it's not ordered by object_id
        muon_dataframe = muon_dataframe_balanced.sample(
            frac=1, random_state=42, ignore_index=True
        )[:max_num_muons]

        voxel_densities_object = object_data["voxel_densities"]
        voxel_densities_noobject = free_data["voxel_densities"]
    else:
        max_num_muons = (
            flags.num_muons
            if flags.num_muons is not None
            else len(object_data["dataframe"])
        )
        muon_dataframe = object_data["dataframe"][:max_num_muons]
        voxel_densities_object = object_data["voxel_densities"]
        voxel_densities_noobject = None

    # max_val = muon_dataframe_balanced["voxel_indices_1d"].apply(max).max()
    dataset = MuonDataset(muon_dataframe, muon_feature_names)
    print(f"Have {len(muon_dataframe)} number of events!")

    # In each batch, we will have multiple groups of muons
    # Each group of muons will be formed into a graph
    num_muons_per_batch = (
        num_muons_per_group * num_muon_groups_per_batch
    )  # total muons per DataLoader batch

    dataset_train, dataset_validation = train_test_split(
        dataset, test_size=config["VALIDATION_SPLIT"]
    )

    # The dataloader will form a list of randomly selected muons
    # The size will be batch_size
    # Then when the dataloader is called, it will call the muon_group_collate_fn
    # muon_group_collate_fn will then group the muons in the batches into num_muons_per_group
    # Then the model will receive num_groups_per_batch of muon groups per batch that will each be formed into a graph
    dataloader_train = DataLoader(
        dataset_train,
        batch_size=num_muons_per_batch,
        shuffle=True,
        collate_fn=lambda b: muon_group_collate_fn(
            b,
            voxel_densities_object,
            voxel_densities_noobject=voxel_densities_noobject,
            num_voxels=MAX_NUMBER_OF_VOXELS,
            device=device,
            num_muons_per_group=num_muons_per_group,
        ),
    )
    dataloader_validation = DataLoader(
        dataset_validation,
        batch_size=num_muons_per_batch,
        shuffle=True,
        collate_fn=lambda b: muon_group_collate_fn(
            b,
            voxel_densities_object,
            voxel_densities_noobject=voxel_densities_noobject,
            num_voxels=MAX_NUMBER_OF_VOXELS,
            device=device,
            num_muons_per_group=num_muons_per_group,
        ),
    )

    MAX_NUMBER_OF_VOXELS = len(voxel_densities_object)

    # small model config for quick runs
    model = MLPF(
        input_dim=num_muon_features,
        embedding_dim=32,
        width=32,
        max_num_voxels=MAX_NUMBER_OF_VOXELS,
        num_convs=1,
        conv_type="gnn_lsh",
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=float(config.get("LEARNING_RATE", 1e-3))
    )

    num_epochs = config.get("NUM_EPOCHS", 100)
    all_epoch_metrics = []
    training_time = time.time()
    for epoch in range(num_epochs):
        print(f"Starting epoch {epoch}")
        epoch_start_time = time.time()

        model.train()
        train_loss = 0.0
        for batch_outputs in dataloader_train:
            optimizer.zero_grad()

            train_loss_per_batch = 0.0

            # batch_outputs is a LIST of (X, y, muon_mask, voxel_mask, path_length)
            for X, y, muon_mask, voxel_mask, path_length in batch_outputs:
                y_pred = model(X, muon_mask, voxel_mask, path_length)

                # Checking if each voxel has at least one muonpassing through it
                voxel_mask_event = voxel_mask.any(dim=1)  # [B, V]
                train_loss_per_batch += mlpf_loss(y_pred, y, voxel_mask_event)
            train_loss_per_batch.backward()
            optimizer.step()

            train_loss += train_loss_per_batch.item()

        avg_train_loss = train_loss / len(dataloader_train)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_outputs in dataloader_validation:
                val_loss_per_batch = 0.0

                for X, y, muon_mask, voxel_mask, path_length in batch_outputs:
                    y_pred = model(X, muon_mask, voxel_mask, path_length)
                    voxel_mask_event = voxel_mask.any(dim=1)
                    val_loss_per_batch += mlpf_loss(y_pred, y, voxel_mask_event)

                val_loss += val_loss_per_batch.item()

        avg_val_loss = val_loss / len(dataloader_validation)

        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Epoch time: {time.time() - epoch_start_time:.2f}s")

        all_epoch_metrics.append(
            {"epoch": epoch + 1, "train_loss": avg_train_loss, "val_loss": avg_val_loss}
        )

        # --- Save checkpoint every epoch ---
        model_path = f"{config['MODEL_SAVE_DIRECTORY']}/{config['MODEL_NAME']}"
        checkpoint_dir = f"{model_path}/checkpoints/"
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    # --- Save final model ---
    final_model_path = os.path.join(model_path, "final_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved to {final_model_path}")

    # --- Save loss history ---
    loss_file = os.path.join(model_path, "training_losses.pt")
    torch.save(all_epoch_metrics, loss_file)
    print(f"Training losses saved to {loss_file}")
    print(f"Training took {time.time() - training_time} s")


if __name__ == "__main__":
    main()
