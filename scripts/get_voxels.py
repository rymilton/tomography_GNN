import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import argparse
import pickle
from raytrace.raytracers import raytrace
import time
import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_voxel_file",
        default="/home/ryan/tomography_GNN/soil_target_voxels.pkl",
        help="Input pickle file with with reconstructed local theta, phi angles",
        type=str,
    )
    parser.add_argument(
        "--input_voxel_positions_file",
        default="/home/ryan/tomography_GNN/soil_target_voxel_coords.pkl",
        help="Input pickle file with with positions of the voxels",
        type=str,
    )
    parser.add_argument(
        "--input_muon_file",
        default="/home/ryan/tomography_GNN/target_merge_global.pkl",
        help="Input pickle file with with reconstructed local theta, phi angles",
        type=str,
    )
    parser.add_argument(
        "--num_muons",
        default=None,
        help="Number of muons to process",
        type=int,
    )
    parser.add_argument(
        "--draw_plots",
        action="store_true",
        default=False,
        help="Whether to draw plots of the muon rays and voxel intersections",
    )
    parser.add_argument(
        "--output_file",
        default="/home/ryan/tomography_GNN/target_merge_global_with_voxels.pkl",
        help="Output pickle file to save transformed global coordinates",
        type=str,
    )
    return parser.parse_args()


def draw_voxel_cube(ax, idx, start, spacing, color="gray", alpha=0.1, lw=0.5):
    idx = np.array(idx)
    start = np.array(start)
    spacing = np.array(spacing)

    first_plane = start - spacing / 2
    corner = first_plane + idx * spacing

    x0, y0, z0 = corner
    dx, dy, dz = spacing

    x = [x0, x0 + dx]
    y = [y0, y0 + dy]
    z = [z0, z0 + dz]

    edges = [
        [(x[0], y[0], z[0]), (x[1], y[0], z[0])],
        [(x[0], y[1], z[0]), (x[1], y[1], z[0])],
        [(x[0], y[0], z[1]), (x[1], y[0], z[1])],
        [(x[0], y[1], z[1]), (x[1], y[1], z[1])],
        [(x[0], y[0], z[0]), (x[0], y[1], z[0])],
        [(x[1], y[0], z[0]), (x[1], y[1], z[0])],
        [(x[0], y[0], z[1]), (x[0], y[1], z[1])],
        [(x[1], y[0], z[1]), (x[1], y[1], z[1])],
        [(x[0], y[0], z[0]), (x[0], y[0], z[1])],
        [(x[1], y[0], z[0]), (x[1], y[0], z[1])],
        [(x[0], y[1], z[0]), (x[0], y[1], z[1])],
        [(x[1], y[1], z[0]), (x[1], y[1], z[1])],
    ]

    # collect artist handles so we can remove them later
    artists = []
    for (x1, y1, z1), (x2, y2, z2) in edges:
        (ln,) = ax.plot([x1, x2], [y1, y2], [z1, z2], color=color, alpha=alpha, lw=lw)
        artists.append(ln)

    return artists


def main():
    flags = parse_arguments()

    with open(flags.input_voxel_file, "rb") as f:
        vol = pickle.load(f)
    with open(flags.input_voxel_positions_file, "rb") as f:
        voxel_positions = pickle.load(f)
    voxel_positions = np.array(voxel_positions) / 1000  # convert from mm to m

    voxel_start_pos = voxel_positions[0, 0, 0]
    dz = voxel_positions[0, 0, 1, 2] - voxel_positions[0, 0, 0, 2]
    dy = voxel_positions[0, 1, 0, 1] - voxel_positions[0, 0, 0, 1]
    dx = voxel_positions[1, 0, 0, 0] - voxel_positions[0, 0, 0, 0]
    voxel_spacing = np.array([dx, dy, dz])
    print("Voxel start position (x, y, z) in meters:", voxel_start_pos)
    print("Voxel spacing (x, y, z) in meters:", voxel_spacing)

    with open(flags.input_muon_file, "rb") as f:
        muons = pickle.load(f)

    muon_detectors = muons["detector"].to_numpy()
    detector_locations = [
        [0, 0, 0.5],
        [-5, -5, 0.5],
        [5, 5, 0.5],
    ]
    max_num_muons = len(muons) if flags.num_muons is None else flags.num_muons
    print(f"Processing {max_num_muons} muons...")
    muons_initial = np.column_stack(
        [
            muons["x_global_above"].to_numpy(),
            muons["y_global_above"].to_numpy(),
            muons["z_global_above"].to_numpy(),
        ]
    )[:max_num_muons]
    muons_final = np.column_stack(
        [
            muons["x_global_below"].to_numpy(),
            muons["y_global_below"].to_numpy(),
            muons["z_global_below"].to_numpy(),
        ]
    )[:max_num_muons]

    # run raytrace
    # For each muon, we get the voxel indices it passes through and the lengths in each voxel
    siddon_start_time = time.time()
    all_muon_voxels, all_muon_lengths_in_voxels = raytrace(
        muons_final,
        muons_initial,
        vol,
        vol_start=voxel_start_pos,
        vol_spacing=voxel_spacing,
    )
    print(
        f"Siddon raytrace time for {len(muons_initial)} muons: %.3f seconds"
        % (time.time() - siddon_start_time)
    )

    output_muons = muons[:max_num_muons].copy().reset_index(drop=True)

    output_muons["voxels_hit"] = pd.Series(all_muon_voxels)
    output_muons["lengths_in_voxels"] = pd.Series(all_muon_lengths_in_voxels)

    # --- Save updated muons DataFrame ---
    output_file = flags.output_file
    with open(output_file, "wb") as f:
        pickle.dump(output_muons, f)

    print(f"Saved muon DataFrame with voxel info to {output_file}")

    if flags.draw_plots:
        fig = plt.figure(figsize=(9, 9))
        ax = fig.add_subplot(111, projection="3d")
        start = voxel_start_pos
        spacing = voxel_spacing
        nz, ny, nx = vol.shape

        print("Drawing static voxel grid...")
        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    draw_voxel_cube(
                        ax, (i, j, k), start, spacing, color="gray", alpha=0.25, lw=0.4
                    )

        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")

        # Equal aspect (critical)
        lims = np.array(
            [
                ax.get_xlim3d(),
                ax.get_ylim3d(),
                ax.get_zlim3d(),
            ]
        )
        center = lims.mean(axis=1)
        radius = (lims[:, 1] - lims[:, 0]).max() / 2
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

        n_muons_to_draw = 10

        for m in range(n_muons_to_draw):

            muon_above = muons_final[m]
            muon_below = muons_initial[m]
            voxels = all_muon_voxels[m]

            dynamic_artists = []

            x_line = [
                muon_above[0],
                detector_locations[muon_detectors[m]][0],
                muon_below[0],
            ]
            y_line = [
                muon_above[1],
                detector_locations[muon_detectors[m]][1],
                muon_below[1],
            ]
            z_line = [
                muon_above[2],
                detector_locations[muon_detectors[m]][2],
                muon_below[2],
            ]

            # ---- draw ray ----
            (ray_line,) = ax.plot(x_line, y_line, z_line, color="black", lw=2.5)
            dynamic_artists.append(ray_line)

            ax.set_title(
                f"Muon in detector {muon_detectors[m]}: $\\theta_{{reco}}={muons['theta_reco'].to_numpy()[m]:.1f}^{{\\circ}}, \\phi_{{reco}}={muons['phi_reco'].to_numpy()[m]:.1f}^{{\\circ}}$"
            )

            # ---- highlight hit voxels ----
            for v in voxels:
                arts = draw_voxel_cube(
                    ax, v, start, spacing, color="red", alpha=0.9, lw=2.0
                )
                dynamic_artists.extend(arts)

            # ---- save figure ----
            outname = f"muon_ray_{m:03d}.png"
            plt.savefig(outname, dpi=150)
            print(f"Saved {outname}")

            # ---- remove dynamic artists ----
            for art in dynamic_artists:
                art.remove()


if __name__ == "__main__":
    main()
