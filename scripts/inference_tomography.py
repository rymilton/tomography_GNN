import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import pickle
import argparse
from torch.utils.data import Dataset, DataLoader
import math, yaml, numpy as np
from mlpf.model.mlpf import MLPF

import pandas as pd


# Dataset class, same as training
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
        voxel_indices = torch.tensor(muon["voxel_indices_1d"], dtype=torch.long)
        path_lengths = torch.tensor(muon["lengths_in_voxels"], dtype=torch.float)
        object_id = int(muon["object_ID"])

        return features, voxel_indices, path_lengths, object_id


# Collate function supporting multiple object types like training
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

    def process_subbatch(subbatch, voxel_densities, object_id):
        if len(subbatch) == 0 or voxel_densities is None:
            return None

        features_list, voxel_indices_list, path_lengths_list, _ = zip(*subbatch)
        total_muons = len(subbatch)
        num_groups = max(1, math.ceil(total_muons / num_muons_per_group))
        total_batch_len = num_groups * num_muons_per_group

        feature_dim = features_list[0].shape[0]
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

        return X, y, muon_mask, voxel_mask, path_length, object_id

    # --- Always process object muons ---
    batch_object = [b for b in batch if b[3] == 1]
    # prcess_subpatch returns a tuple of muon features, true densities, muon mask, voxel mask, path length
    out_obj = process_subbatch(batch_object, voxel_densities_object, 1)
    if out_obj is not None:
        outputs.append(out_obj)

    batch_noobject = [b for b in batch if b[3] == 0]
    out_noobj = process_subbatch(batch_noobject, voxel_densities_noobject, 0)
    if out_noobj is not None:
        outputs.append(out_noobj)

    return outputs


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_object_file",
        default="/home/ryan/tomography_GNN/preprocessed_data_test.pkl",
        help="Input pickle file with object and reconstructed theta, phi angles, and the voxel interesections",
        type=str,
    )
    parser.add_argument(
        "--input_free_file",
        default="/home/ryan/tomography_GNN/preprocessed_data_test.pkl",
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
    config = yaml.safe_load(open(flags.config))
    muon_feature_names = config["MUON_FEATURES"]
    num_muon_features = len(muon_feature_names)
    num_muon_groups_per_batch = config.get("NUM_MUON_GROUPS_PER_BATCH", 64)
    num_muons_per_group = config.get("NUM_MUONS_PER_GROUP", 64)

    # Load input data
    with open(flags.input_object_file, "rb") as f:
        object_data = pickle.load(f)

    object_df = object_data["dataframe"]
    # Prepare voxel densities per object type
    voxel_densities_dict = {}
    voxel_densities_object = object_data["voxel_densities"]
    voxel_densities_dict[1] = voxel_densities_object

    if flags.free_and_object:
        with open(flags.input_free_file, "rb") as f:
            free_data = pickle.load(f)
        free_df = free_data["dataframe"]
        muon_dataframe = pd.concat([free_df, object_df], ignore_index=True)
        max_num_muons = (
            flags.num_muons if flags.num_muons is not None else len(muon_dataframe)
        )
        # Shuffle rows so it's not ordered by object_id
        muon_dataframe = muon_dataframe.sample(
            frac=1, random_state=42, ignore_index=True
        )[:max_num_muons]
        voxel_densities_noobject = free_data["voxel_densities"]
        voxel_densities_dict[0] = voxel_densities_noobject
    else:
        muon_dataframe = object_df
        max_num_muons = (
            flags.num_muons if flags.num_muons is not None else len(muon_dataframe)
        )
        muon_dataframe = muon_dataframe.sample(
            frac=1, random_state=42, ignore_index=True
        )[:max_num_muons]
        voxel_densities_noobject = None

    dataset = MuonDataset(muon_dataframe, muon_feature_names)
    num_muons_per_batch = num_muon_groups_per_batch * num_muons_per_group
    MAX_NUMBER_OF_VOXELS = max(len(v) for v in voxel_densities_dict.values())

    dataloader = DataLoader(
        dataset,
        batch_size=num_muons_per_batch,
        shuffle=True,
        collate_fn=lambda b: muon_group_collate_fn(
            b,
            voxel_densities_object,
            voxel_densities_noobject,
            num_voxels=MAX_NUMBER_OF_VOXELS,
            device=device,
            num_muons_per_group=num_muons_per_group,
        ),
    )

    # Load model
    model_path = f"{config['MODEL_SAVE_DIRECTORY']}/{config['MODEL_NAME']}"
    model = MLPF(
        input_dim=num_muon_features,
        embedding_dim=32,
        width=32,
        max_num_voxels=MAX_NUMBER_OF_VOXELS,
        num_convs=1,
        conv_type="gnn_lsh",
    ).to(device)
    model.load_state_dict(
        torch.load(model_path + "/final_model.pth", map_location=device)
    )
    model.eval()

    # Initialize sums, counts, and predictions per object typconfige
    voxel_sum_dict = {
        obj_id: np.zeros(len(vox), dtype=np.float64)
        for obj_id, vox in voxel_densities_dict.items()
    }
    voxel_count_dict = {
        obj_id: np.zeros(len(vox), dtype=np.int64)
        for obj_id, vox in voxel_densities_dict.items()
    }
    voxel_predictions_dict = {
        obj_id: np.full(len(vox), -1.0, dtype=np.float64)
        for obj_id, vox in voxel_densities_dict.items()
    }

    # Inference loop
    with torch.no_grad():
        for batch_outputs in dataloader:
            # batch_outputs is a list of (X, y, muon_mask, voxel_mask, path_length, object_ID) per object type
            for X, y, muon_mask, voxel_mask, path_length, object_ID in batch_outputs:
                y_pred = model(X, muon_mask, voxel_mask, path_length)

                # X's shape is [num_groups, num_muons_per_group, num_features]
                # Every muon in each muon group has the same object_ID
                obj_id = object_ID

                voxel_mask_event = voxel_mask.any(dim=1)

                # Looping over the predictions for each group of muons
                for pred, mask, true in zip(y_pred, voxel_mask_event, y):
                    pred = pred.detach().cpu().numpy()
                    true = true.detach().cpu().numpy()
                    mask = mask.detach().cpu().numpy()
                    voxel_sum_dict[obj_id][mask] += pred[mask]
                    voxel_count_dict[obj_id][mask] += 1

    # Average per object type
    for obj_id in voxel_densities_dict.keys():
        touched = voxel_count_dict[obj_id] > 0
        voxel_predictions_dict[obj_id][touched] = (
            voxel_sum_dict[obj_id][touched] / voxel_count_dict[obj_id][touched]
        )
    print(voxel_predictions_dict[1])
    # Save predictions
    output_file = os.path.join(model_path, "test_predictions.pt")
    print(output_file)
    torch.save(
        {
            "voxel_predictions": voxel_predictions_dict,
            "voxel_true_densities": voxel_densities_dict,
        },
        output_file,
    )

    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    main()
