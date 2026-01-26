import numpy as np
import pickle
import pandas as pd
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--input_file",
        default="/home/ryan/tomography_GNN/target_merge.pkl",
        help="Input pickle file with with reconstructed local theta, phi angles",
        type=str,
    )
    parser.add_argument(
        "--output_file",
        default="/home/ryan/tomography_GNN/target_merge_global.pkl",
        help="Output pickle file to save transformed global coordinates",
        type=str,
    )
    return parser.parse_args()
def Rx(alpha):
    """Rotation about x-axis by angle alpha (rad)."""
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0,  c,  -s ],
        [0.0,  s,   c ]
    ])

def Ry(beta):
    """Rotation about y-axis by angle beta (rad)."""
    c, s = np.cos(beta), np.sin(beta)
    return np.array([
        [ c,  0.0,  s ],
        [0.0, 1.0, 0.0],
        [-s,  0.0,  c ]
    ])

def Rz(psi):
    """Rotation about z-axis by angle psi (rad)."""
    c, s = np.cos(psi), np.sin(psi)
    return np.array([
        [ c, -s, 0.0],
        [ s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])

def rotation_matrix_local_to_global(alpha, beta, psi):
    """
    Build R_local_to_global using Z–Y–X (yaw–pitch–roll) convention.

    alpha : roll  (x-axis rotation, rad)
    beta  : pitch (y-axis rotation, rad)
    psi   : yaw   (z-axis rotation, rad)
    """
    return Rz(psi) @ Ry(beta) @ Rx(alpha)

def xy_theta_phi_at_height_in_global(
    theta_local,
    phi_local,
    origin_local_in_global, # [x0, y0, z0]
    R_local_to_global,
    z_target_global
):
    """
    Compute (x, y) position and (theta, phi) direction in the global frame
    where a ray defined in the local frame crosses z_global = z_target_global.
    """

    # Direction in local frame
    unitvector_local = np.array([
        np.sin(theta_local) * np.cos(phi_local),
        np.sin(theta_local) * np.sin(phi_local),
        np.cos(theta_local)
    ])

    # Rotate direction into global frame
    unitvector_global = R_local_to_global @ unitvector_local

    # Normalize (safety)
    unitvector_global /= np.linalg.norm(unitvector_global)

    # Ray origin in global frame
    r0_global = np.asarray(origin_local_in_global)
    x0_global, y0_global, z0_global = r0_global[0], r0_global[1], r0_global[2]
    
    # Solve for intersection with z_global plane
    if np.abs(unitvector_global[2]) < 1e-12:
        return None, None, None, None  # Ray is parallel to the plane

    # The ray equation is:
    # r_global = r0_global + t * unitvector_global
    t = (z_target_global - z0_global) / unitvector_global[2]

    # Intersection point
    x_global = x0_global + t * unitvector_global[0]
    y_global = y0_global + t * unitvector_global[1]

    # Direction angles in global frame
    theta_global = np.arccos(np.clip(unitvector_global[2], -1.0, 1.0))
    phi_global = np.arctan2(unitvector_global[1], unitvector_global[0])

    return x_global, y_global, theta_global, phi_global

def main():

    flags = parse_arguments()
    # Detector locations in global frame in meters
    detector_locations = [[0, 0, 0.5], [-5, -5, 0.5], [5, 5, 0.5]]
    # Detector pitch, yaw, roll in radians
    detector_alpha_beta_psi = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]

    desired_z_location_above = 20 # The top of the geometry is 20 meters. We will get x,y at this z height.
    desired_z_location_below = -20

    # Open pickle file and load data

    with open(flags.input_file, "rb") as f:
        detector_data = pickle.load(f)

    df = pd.DataFrame(detector_data)
    for index, row in df.iterrows():
        detector_id = int(row['detector'])
        
        # Jay said that the theta = 90 degree events are invalid, so we skip them.
        if row['theta_reco'] == 90:
            continue
        theta_local = np.deg2rad(row['theta_reco'])
        phi_local = np.deg2rad(row['phi_reco'])

        origin_local_in_global = detector_locations[detector_id]
        alpha, beta, psi = detector_alpha_beta_psi[detector_id]

        R_local_to_global = rotation_matrix_local_to_global(alpha, beta, psi)

        x_global_above, y_global_above, theta_global, phi_global = xy_theta_phi_at_height_in_global(
            theta_local,
            phi_local,
            origin_local_in_global,
            R_local_to_global,
            desired_z_location_above
        )
        x_global_below, y_global_below, _, _ = xy_theta_phi_at_height_in_global(
            theta_local,
            phi_local,
            origin_local_in_global,
            R_local_to_global,
            desired_z_location_below
        )

        if x_global_above is None or x_global_below is None:
            print(f"Detector ID: {detector_id} - Ray is parallel to the target plane.")
            continue

        # Save the resulting global coordinates and angles back to the dataframe
        df.at[index, 'x_global_above'] = x_global_above
        df.at[index, 'y_global_above'] = y_global_above
        df.at[index, 'x_global_below'] = x_global_below
        df.at[index, 'y_global_below'] = y_global_below
        df.at[index, 'theta_global'] = np.rad2deg(theta_global)
        df.at[index, 'phi_global'] = np.rad2deg(phi_global)

        # Print the first 10 entries for verification
        if index < 10:
            print(f"Detector ID: {detector_id}")
            print(f"Global Coordinates at z={desired_z_location_above} m: x = {x_global_above:.2f} m, y = {y_global_above:.2f} m")
            print(f"Global Coordinates at z={desired_z_location_below} m: x = {x_global_below:.2f} m, y = {y_global_below:.2f} m")
            print(f"Global coordinates at detector: x0 = {origin_local_in_global[0]:.2f} m, y0 = {origin_local_in_global[1]:.2f} m, z0 = {origin_local_in_global[2]:.2f} m")
            print(f"Global Angles: theta = {np.rad2deg(theta_global):.2f} deg, phi = {np.rad2deg(phi_global):.2f} deg")
            print(f"Local Angles: theta = {np.rad2deg(theta_local):.2f} deg, phi = {np.rad2deg(phi_local):.2f} deg")
            print("-----")
    df.to_pickle(flags.output_file)
if __name__ == "__main__":
    main()