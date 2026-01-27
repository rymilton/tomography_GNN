import numpy as np
import argparse
import pickle
import time
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input_data_file",
        default="/home/ryan/tomography_GNN/target_merge_global_with_voxels.pkl",
        help="Input pickle file with with reconstructed theta, phi angles, and the voxel interesections",
        type=str,
    )
    parser.add_argument(
        "--input_voxel_densities_file",
        default="/home/ryan/tomography_GNN/soil_target_voxels.pkl",
        help="Input pickle file with voxel densities",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        default="/home/ryan/tomography_GNN/preprocessed_data.pkl",
        help="Output pickle file to save preprocessed data to",
        type=str,
    )
    return parser.parse_args()

def scale_data(data, min_val, max_val):
    scaled = (data - min_val) / (max_val - min_val)
    return scaled

def main():
    flags = parse_arguments()
    with open(flags.input_data_file, "rb") as f:
        data = pickle.load(f)
    with open(flags.input_voxel_densities_file, "rb") as f:
        voxel_densities = pickle.load(f)
    
    flattened_densities = voxel_densities.flatten()
    rows = []
    for idx, row in data.iterrows():
        voxels = np.array(row['voxels_hit'], dtype=int)
        
        voxel_indices_1d = np.ravel_multi_index(
            (voxels[:, 0], voxels[:, 1], voxels[:, 2]),
            voxel_densities.shape
        )
        lengths_in_voxels = np.array(row['lengths_in_voxels'])
        theta_reco = scale_data(row['theta_reco'], 0, 90)
        phi_reco = scale_data(row['phi_reco'], -180, 180)

        z_above_object = scale_data(row["z_global_above"], -20, 20)
        z_below_object = scale_data(row["z_global_below"], -20, 20)
        x_above_object = scale_data(row["x_global_above"], -50, 50)
        x_below_object = scale_data(row["x_global_below"], -50, 50)
        y_above_object = scale_data(row["y_global_above"], -50, 50)
        y_below_object = scale_data(row["y_global_below"], -50, 50)

        rows.append({
            "voxel_indices_1d": voxel_indices_1d,
            "lengths_in_voxels": lengths_in_voxels,
            "theta_reco": theta_reco,
            "phi_reco": phi_reco,
            "z_global_above": z_above_object,
            "z_global_below": z_below_object,
            "x_global_above": x_above_object,
            "x_global_below": x_below_object,
            "y_global_above": y_above_object,
            "y_global_below": y_below_object,
        })
    output_df = pd.DataFrame(rows)

    # Save the output dataframe to a pickle file, as well as the flattened voxel densities. Just want a single copy of the voxel densities, so we make a file with the dataframe and the voxel densities array
    with open(flags.output_file, "wb") as f:
        pickle.dump({'dataframe': output_df, 'voxel_densities': flattened_densities}, f)
if __name__ == "__main__":
    main()