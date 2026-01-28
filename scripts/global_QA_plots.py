import numpy as np
import pickle
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        default="/home/ryan/tomography_GNN/target_merge_global.pkl",
        help="Input pickle file with reconstructed local theta, phi angles and global coordinates",
        type=str,
    )
    return parser.parse_args()


def main():

    flags = parse_arguments()

    # Detector locations in global frame (meters)
    detector_locations = [
        [0, 0, 0.5],
        [-5, -5, 0.5],
        [5, 5, 0.5],
    ]

    # Define QA viewing angles: (name, elev, azim)
    views = [
        ("default", 30, -60),
        ("top", 90, 0),
        ("xz", 0, 0),
        ("yz", 0, 90),
        ("iso", 30, 45),
        ("iso_back", 30, 225),
    ]

    # Load data
    with open(flags.input_file, "rb") as f:
        detector_data = pickle.load(f)

    df = pd.DataFrame(detector_data)

    plotted = 0
    max_plots = 10

    for index, row in df.iterrows():
        if plotted >= max_plots:
            break

        detector_id = int(row["detector"])

        # Skip invalid events
        if row["theta_reco"] == 90:
            continue

        theta_local = row["theta_reco"]
        phi_local = row["phi_reco"]

        x_above = row["x_global_above"]
        y_above = row["y_global_above"]
        z_above = row["z_global_above"]

        x_below = row["x_global_below"]
        y_below = row["y_global_below"]
        z_below = row["z_global_below"]

        theta_global = row["theta_global"]
        phi_global = row["phi_global"]

        x0, y0, z0 = detector_locations[detector_id]

        # Ray points
        x_line = [x_below, x0, x_above]
        y_line = [y_below, y0, y_above]
        z_line = [z_below, z0, z_above]

        # Create one figure per event
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot ray
        ax.plot(x_line, y_line, z_line, linewidth=2)

        # Plot detector
        ax.scatter([x0], [y0], [z0], s=60)

        # Plot intersections
        ax.scatter(
            [x_above, x_below],
            [y_above, y_below],
            [z_above, z_below],
            s=40,
        )

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]", labelpad=0.05)
        ax.set_xlim(-25, 25)
        ax.set_ylim(-25, 25)
        ax.set_zlim(-25, 25)
        # ax.set_box_aspect([2, 2, 2])
        plt.tight_layout()

        # Save one image per view
        for name, elev, azim in views:
            ax.view_init(elev=elev, azim=azim)
            ax.set_title(
                f"Detector {detector_id}, "
                f"$\\theta={theta_global:.1f}^{{\circ}}, \phi={phi_global:.1f}^{{\circ}}$"
            )
            plt.savefig(
                f"global_QA_detector_{detector_id}_event_{index}_{name}.png",
                dpi=150,
                bbox_inches="tight",
            )

        plt.close(fig)
        print(f"Detector ID: {detector_id}")
        print(f"Detector position: x0={x0:.2f}, y0={y0:.2f}, z0={z0:.2f}")
        print(f"Above plane: x={x_above:.2f}, y={y_above:.2f}, z={z_above}")
        print(f"Below plane: x={x_below:.2f}, y={y_below:.2f}, z={z_below}")
        print(f"Global angles: theta={theta_global:.2f}, phi={phi_global:.2f}")
        print("-----")

        plotted += 1


if __name__ == "__main__":
    main()
