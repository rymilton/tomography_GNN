import numpy as np
import pickle
import pandas as pd
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
    v_local = np.array([
        np.sin(theta_local) * np.cos(phi_local),
        np.sin(theta_local) * np.sin(phi_local),
        np.cos(theta_local)
    ])

    # Rotate direction into global frame
    v_global = R_local_to_global @ v_local

    # Normalize (safety)
    v_global /= np.linalg.norm(v_global)

    # Ray origin in global frame
    r0_global = np.asarray(origin_local_in_global)

    # Solve for intersection with z_global plane
    if np.abs(v_global[2]) < 1e-12:
        return None, None, None, None  # Ray is parallel to the plane

    t = (z_target_global - r0_global[2]) / v_global[2]

    # Intersection point
    x_global = r0_global[0] + t * v_global[0]
    y_global = r0_global[1] + t * v_global[1]

    # Direction angles in global frame
    theta_global = np.arccos(np.clip(v_global[2], -1.0, 1.0))
    phi_global = np.arctan2(v_global[1], v_global[0])

    return x_global, y_global, theta_global, phi_global

def main():

    # Detector locations in global frame in meters
    detector_locations = [[0, 0, 0.5], [-5, -5, 0.5], [5, 5, 0.5]]
    # Detector pitch, yaw, roll in radians
    detector_alpha_beta_psi = [(0, 0, 0), (0, 0, 0), (0, 0, 0)]

    desired_z_location = 40 # The top of the geometry is 40 meters. We will get x,y at this z height.

    # Open pickle file and load data
    
    with open("/home/ryan/tomography_GNN/target_merge.pkl", "rb") as f:
        detector_data = pickle.load(f)

    df = pd.DataFrame(detector_data)
    for index, row in df.iterrows():
        detector_id = int(row['detector'])
        theta_local = np.deg2rad(row['theta_reco'])
        phi_local = np.deg2rad(row['phi_reco'])
        print(theta_local, phi_local)

        origin_local_in_global = detector_locations[detector_id]
        alpha, beta, psi = detector_alpha_beta_psi[detector_id]

        R_local_to_global = rotation_matrix_local_to_global(alpha, beta, psi)

        x_global, y_global, theta_global, phi_global = xy_theta_phi_at_height_in_global(
            theta_local,
            phi_local,
            origin_local_in_global,
            R_local_to_global,
            desired_z_location
        )
        if x_global is None:
            print(f"Detector ID: {detector_id} - Ray is parallel to the target plane.")
            continue

        # Save the resulting global coordinates and angles back to the dataframe
        df.at[index, 'x_global'] = x_global
        df.at[index, 'y_global'] = y_global
        df.at[index, 'theta_global'] = np.rad2deg(theta_global)
        df.at[index, 'phi_global'] = np.rad2deg(phi_global)

        if index < 10:
            print(f"Detector ID: {detector_id}")
            print(f"Global Coordinates at z={desired_z_location} m: x = {x_global:.2f} m, y = {y_global:.2f} m")
            print(f"Global coordinates at detector: x0 = {origin_local_in_global[0]:.2f} m, y0 = {origin_local_in_global[1]:.2f} m, z0 = {origin_local_in_global[2]:.2f} m")
            print(f"Global Angles: theta = {np.rad2deg(theta_global):.2f} deg, phi = {np.rad2deg(phi_global):.2f} deg")
            print(f"Local Angles: theta = {np.rad2deg(theta_local):.2f} deg, phi = {np.rad2deg(phi_local):.2f} deg")
            print("-----")
    df.to_pickle("/home/ryan/tomography_GNN/target_global.pkl")
if __name__ == "__main__":
    main()