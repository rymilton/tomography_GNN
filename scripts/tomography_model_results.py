import torch
import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import mplhep as hep

hep.style.use("CMS")
import os

import argparse, pickle
from matplotlib import cm
from matplotlib.colors import Normalize


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_path",
        default="/home/ryan/tomography_GNN/tomography_models/testing_training/",
        help="Directory containing the trained model",
        type=str,
    )
    parser.add_argument(
        "--input_voxel_densities_file",
        default="/home/ryan/tomography_GNN/input_files/soil_target_voxels.pkl",
        help="Input pickle file with voxel densities",
        type=str,
    )
    parser.add_argument(
        "--input_voxel_positions_file",
        default="/home/ryan/tomography_GNN/input_files/soil_target_voxel_coords.pkl",
        help="Input pickle file with with positions of the voxels",
        type=str,
    )
    parser.add_argument(
        "--plot_directory",
        default="/home/ryan/tomography_GNN/tomography_models/testing_training/plots/",
        help="Directory to store the plots",
        type=str,
    )
    return parser.parse_args()


def main():
    flags = parse_arguments()

    model_dir = flags.model_path
    plot_directory = flags.plot_directory
    os.makedirs(plot_directory, exist_ok=True)

    # Opening the files
    losses = torch.load(model_dir + "/training_losses.pt")
    loss_epochs = [losses[i]["epoch"] for i in range(len(losses))]
    train_loss_per_epoch = [losses[i]["train_loss"] for i in range(len(losses))]
    validation_loss_per_epoch = [losses[i]["val_loss"] for i in range(len(losses))]

    model_predictions = torch.load(
        model_dir + "/test_predictions.pt", weights_only=False
    )
    density_predictions_noobject = model_predictions["voxel_predictions"][0]
    density_predictions_object = model_predictions["voxel_predictions"][1]
    density_true_noobject = model_predictions["voxel_true_densities"][0]
    density_true_object = model_predictions["voxel_true_densities"][1]

    with open(flags.input_voxel_positions_file, "rb") as f:
        voxel_positions = pickle.load(f)
    voxel_positions = np.array(voxel_positions) / 1000  # convert from mm to m

    density_predictions_noobject_3d = density_predictions_noobject.reshape(
        density_true_noobject.shape
    )
    density_predictions_object_3d = density_predictions_object.reshape(
        density_true_object.shape
    )

    # Loss curve
    figure_loss = plt.figure(figsize=(12, 8))
    plt.plot(loss_epochs, train_loss_per_epoch, label="Training")
    plt.plot(loss_epochs, validation_loss_per_epoch, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.xlim(0, 105)
    plt.ylim(0, 40)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_directory + "losses.png")

    from mpl_toolkits.mplot3d import Axes3D

    # Flatten densities and positions for plotting
    voxel_positions_flat = voxel_positions.reshape(-1, 3)
    pred_densities_noobject_flat = np.array(density_predictions_noobject_3d).flatten()
    pred_densities_object_flat = np.array(density_predictions_object_3d).flatten()

    # Optionally ignore voxels with -1 (untouched)
    mask_valid_noobject = pred_densities_noobject_flat >= 0
    voxel_positions_noobject_flat = voxel_positions_flat
    density_true_noobject = density_true_noobject
    pred_densities_noobject_flat = pred_densities_noobject_flat[mask_valid_noobject]

    mask_valid_object = pred_densities_object_flat >= 0
    voxel_positions_object_flat = voxel_positions_flat
    density_true_object = density_true_object
    pred_densities_object_flat = pred_densities_object_flat[mask_valid_object]

    x_unique_object = np.unique(voxel_positions_object_flat[:, 0])
    y_unique_object = np.unique(voxel_positions_object_flat[:, 1])
    z_unique_object = np.unique(voxel_positions_object_flat[:, 2])

    dx_object = np.min(np.diff(x_unique_object)) / 2
    dy_object = np.min(np.diff(y_unique_object)) / 2
    dz_object = np.min(np.diff(z_unique_object)) / 2

    x_unique_noobject = np.unique(voxel_positions_noobject_flat[:, 0])
    y_unique_noobject = np.unique(voxel_positions_noobject_flat[:, 1])
    z_unique_noobject = np.unique(voxel_positions_noobject_flat[:, 2])

    dx_noobject = np.min(np.diff(x_unique_noobject)) / 2
    dy_noobject = np.min(np.diff(y_unique_noobject)) / 2
    dz_noobject = np.min(np.diff(z_unique_noobject)) / 2

    offset = 0.0

    # 3D scatter plot for true densities
    fig_true = plt.figure(figsize=(12, 10))
    ax_true = fig_true.add_subplot(111, projection="3d")
    norm = Normalize(vmin=np.min(0), vmax=np.max(10))
    colors = cm.hot_r(norm(density_true_object))
    colors[:, 3] = 0.2

    ax_true.bar3d(
        voxel_positions_object_flat[:, 0] + offset,
        voxel_positions_object_flat[:, 1] + offset,
        voxel_positions_object_flat[:, 2] + offset,
        dx_object,
        dy_object,
        dz_object,
        color=colors,
        shade=True,
        linewidth=0,
    )

    mappable = cm.ScalarMappable(norm=norm, cmap="hot_r")
    mappable.set_array([])
    fig_true.colorbar(mappable, ax=ax_true, label="True density")
    ax_true.view_init(elev=25, azim=50)
    ax_true.set_xlabel("X [m]", labelpad=15)
    ax_true.set_ylabel("Y [m]", labelpad=15)
    ax_true.set_zlabel("Z [m]", labelpad=15)
    ax_true.set_title("True Voxel Densities\n With object")
    plt.tight_layout()
    plt.savefig(plot_directory + "true_densities_3d_object.png")

    fig_true = plt.figure(figsize=(12, 10))
    ax_true = fig_true.add_subplot(111, projection="3d")
    norm = Normalize(vmin=np.min(0), vmax=np.max(10))
    colors = cm.hot_r(norm(density_true_noobject))
    colors[:, 3] = 0.2

    ax_true.bar3d(
        voxel_positions_noobject_flat[:, 0] + offset,
        voxel_positions_noobject_flat[:, 1] + offset,
        voxel_positions_noobject_flat[:, 2] + offset,
        dx_noobject,
        dy_noobject,
        dz_noobject,
        color=colors,
        shade=True,
        linewidth=0,
    )
    mappable = cm.ScalarMappable(norm=norm, cmap="hot_r")
    mappable.set_array([])
    fig_true.colorbar(mappable, ax=ax_true, label="True density")
    ax_true.view_init(elev=25, azim=50)
    ax_true.set_xlabel("X [m]", labelpad=15)
    ax_true.set_ylabel("Y [m]", labelpad=15)
    ax_true.set_zlabel("Z [m]", labelpad=15)
    ax_true.set_title("True Voxel Densities\n No object")
    plt.tight_layout()
    plt.savefig(plot_directory + "true_densities_3d_noobject.png")

    # 3D scatter plot for predicted densities
    fig_pred = plt.figure(figsize=(12, 10))
    ax_pred = fig_pred.add_subplot(111, projection="3d")

    norm = Normalize(vmin=np.min(0), vmax=np.max(10))
    colors = cm.hot_r(norm(pred_densities_object_flat))
    colors[:, 3] = 0.2

    ax_pred.bar3d(
        voxel_positions_object_flat[:, 0][mask_valid_object] + offset,
        voxel_positions_object_flat[:, 1][mask_valid_object] + offset,
        voxel_positions_object_flat[:, 2][mask_valid_object] + offset,
        dx_object,
        dy_object,
        dz_object,
        color=colors,
        shade=True,
        linewidth=0,
    )
    mappable = cm.ScalarMappable(norm=norm, cmap="hot_r")
    mappable.set_array([])
    fig_pred.colorbar(mappable, ax=ax_pred, label="Predicted density")
    ax_pred.view_init(elev=25, azim=50)
    ax_pred.set_xlabel("X [m]", labelpad=15)
    ax_pred.set_ylabel("Y [m]", labelpad=15)
    ax_pred.set_zlabel("Z [m]", labelpad=15)
    ax_pred.set_title("Predicted Voxel Densities\n With object")
    plt.tight_layout()
    plt.savefig(plot_directory + "predicted_densities_3d_object.png")

    # 3D scatter plot for predicted densities
    fig_pred = plt.figure(figsize=(12, 10))
    ax_pred = fig_pred.add_subplot(111, projection="3d")
    norm = Normalize(vmin=np.min(0), vmax=np.max(10))
    colors = cm.hot_r(norm(pred_densities_noobject_flat))
    colors[:, 3] = 0.2
    ax_pred.bar3d(
        voxel_positions_noobject_flat[:, 0][mask_valid_noobject] + offset,
        voxel_positions_noobject_flat[:, 1][mask_valid_noobject] + offset,
        voxel_positions_noobject_flat[:, 2][mask_valid_noobject] + offset,
        dx_noobject,
        dy_noobject,
        dz_noobject,
        color=colors,
        shade=True,
        linewidth=0,
    )
    mappable = cm.ScalarMappable(norm=norm, cmap="hot_r")
    mappable.set_array([])
    fig_pred.colorbar(mappable, ax=ax_pred, label="Predicted density")
    ax_pred.view_init(elev=25, azim=50)
    ax_pred.set_xlabel("X [m]", labelpad=15)
    ax_pred.set_ylabel("Y [m]", labelpad=15)
    ax_pred.set_zlabel("Z [m]", labelpad=15)
    ax_pred.set_title("Predicted Voxel Densities\n No object")
    plt.tight_layout()
    plt.savefig(plot_directory + "predicted_densities_3d_noobject.png")


if __name__ == "__main__":
    main()
