import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from feature_match import sift, matching_features, Draw_matches
from fundamental_matrix import get_fundamental_matrix, draw_epilines
# Measona_calib
K1=K2=np.asarray([[1421.9, 0.5, 509.2],
                 [0,   1421.9, 380.2],
                 [0,        0,     1]])

# Starue_calib
# Camera A:
K1=np.array([[5426.566895, 0.678017, 330.096680],
             [0.000000, 5423.133301, 648.950012],
             [0.000000,    0.000000,   1.000000]])
# Camera B:
K2=np.array([[5426.566895, 0.678017, 387.430023],
             [0.000000, 5423.133301, 620.616699],
             [0.000000,    0.000000,   1.000000]])


# def ndarray2matlab(x):
#     return matlab.double(x.tolist())

def compute_essential_matrix(F, K1, K2):
    """
    Compute the Essential Matrix from the Fundamental Matrix.
    Args:
        F: Fundamental Matrix.
        K1, K2: Intrinsic matrices of the cameras.
    Returns:
        Essential Matrix E.
    """
    E = K2.T @ F @ K1
    # Enforce rank-2 constraint
    U, S, Vt = np.linalg.svd(E)
    S[0] = S[1] = (S[0] + S[1]) / 2
    S[2] = 0
    E = U @ np.diag(S) @ Vt
    return E

def decompose_essential_matrix(E):
    """
    Decompose the Essential Matrix into possible camera poses.
    Args:
        E: Essential Matrix.
    Returns:
        List of 4 possible [R|t] combinations.
    """
    # SVD of E
    U, _, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    # Ensure proper rotation matrices
    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    # Two possible rotations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt

    # Translation vector
    t = U[:, 2]

    # Four combinations of [R|t]
    poses = [
        (R1, t),
        (R1, -t),
        (R2, t),
        (R2, -t)
    ]

    return poses

def triangulate_points(P1, P2, pts1, pts2):
    """
    Triangulate 3D points from two camera matrices and point correspondences.
    Args:
        P1: Projection matrix for camera 1 (3x4).
        P2: Projection matrix for camera 2 (3x4).
        pts1: Points in the first image (Nx2).
        pts2: Corresponding points in the second image (Nx2).
    Returns:
        Reconstructed 3D points in homogeneous coordinates (Nx4).
    """
    points_4d = []
    for pt1, pt2 in zip(pts1, pts2):
        A = np.zeros((4, 4))
        A[0] = pt1[0] * P1[2] - P1[0]
        A[1] = pt1[1] * P1[2] - P1[1]
        A[2] = pt2[0] * P2[2] - P2[0]
        A[3] = pt2[1] * P2[2] - P2[1]

        # Solve for the point using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        points_4d.append(X / X[3])  # Normalize to homogeneous coordinates

    return np.array(points_4d)

def evaluate_poses(K1, K2, poses, pts1, pts2):
    """
    Evaluate the four poses and determine the most appropriate solution.
    Args:
        K1, K2: Intrinsic matrices for the two cameras.
        poses: List of 4 possible [R|t] combinations.
        pts1, pts2: Matched points from the two images.
    Returns:
        Best pose and corresponding 3D points.
    """
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Camera 1 is canonical
    best_pose = None
    max_in_front = 0
    best_3d_points = None

    for R, t in poses:
        # Construct the projection matrix for the second camera
        P2 = K2 @ np.hstack((R, t.reshape(-1, 1)))

        # Triangulate points
        points_3d = triangulate_points(P1, P2, pts1, pts2)

        # Check the cheirality condition
        in_front = 0
        for X in points_3d:
            # Check if the point is in front of both cameras
            if X[2] > 0 and (R @ X[:3] + t)[2] > 0:
                in_front += 1

        # Update the best pose if more points are in front
        if in_front > max_in_front:
            max_in_front = in_front
            best_pose = (R, t)
            best_3d_points = points_3d

    return best_pose, best_3d_points

def export_to_obj(points_3d, output_file):
    """
    Export 3D points to an OBJ file.
    Args:
        points_3d: Reconstructed 3D points (Nx3 or Nx4).
        output_file: Path to save the OBJ file.
    """
    with open(output_file, 'w') as f:
        for point in points_3d:
            # Write vertex coordinates (drop homogeneous coordinate if present)
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")
    print(f"Exported 3D points to {output_file}")

def plot_3d_points(points_3d):
    """
    Plot 3D points using matplotlib.
    Args:
        points_3d: Reconstructed 3D points (Nx3).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], s=1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

if __name__=='__main__':
    img1_path=os.path.join('data/Statue1.bmp') 
    img2_path=os.path.join('data/Statue2.bmp') 

    img1=cv2.imread(img1_path)
    img2=cv2.imread(img2_path)

    # 1.Feature Matching (SIFT)
    keys1,desc1=sift(img1)
    keys2,desc2=sift(img2)
    matches=matching_features(desc1,desc2)
    Draw_matches(matches,img1,img2,keys1,keys2)
    
    # 2. get the Fundamental matrix by correspondence
    pts1 = np.array([(int(keys1[i][0]), int(keys1[i][1])) for i, j in matches])  # 影像1的匹配點
    pts2 = np.array([(int(keys2[j][0]), int(keys2[j][1])) for i, j in matches])  # 影像2的匹配點

    F,inlier_idxs = get_fundamental_matrix(pts1,pts2,threshold=3)
    inlier1 = pts1[inlier_idxs]
    inlier2 = pts2[inlier_idxs]

    img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    # 3. draw epipolar lines
    print(f'# correspondence: {len(pts1)}')
    print(f'# inliers: {len(inlier_idxs)}')
    print("Fundamental Matrix F:\n", F)
    draw_epilines(img1, img2, inlier1, inlier2, F, 'epilines.png')

    # 4. four possible solutions of essential matrix
    # Compute the Essential Matrix
    E = compute_essential_matrix(F, K1, K2)

    # Decompose into possible poses
    poses = decompose_essential_matrix(E)

    # Print the results
    for i, (R, t) in enumerate(poses):
        print(f"Pose {i + 1}:")
        print(f"Rotation:\n{R}")
        print(f"Translation:\n{t}\n")

    # 5. find out the most appropriate solution of essential matrix
    # 6. get the 3D points
    # Evaluate the four poses to find the most appropriate one
    best_pose, best_3d_points = evaluate_poses(K1, K2, poses, pts1, pts2)

    # Print the best pose
    R_best, t_best = best_pose
    print("Best Rotation Matrix:")
    print(R_best)
    print("\nBest Translation Vector:")
    print(t_best)

    # Visualize the 3D points (optional)
    print("Reconstructed 3D Points:")
    print(best_3d_points[:, :3])  # Drop the homogeneous coordinate for visualization

    # 7. get 3D model

    export_to_obj(best_3d_points[:, :3], output_file='output/reconstructed_scene.obj')

    # Plot the best 3D points
    plot_3d_points(best_3d_points[:, :3])