import os
import cv2
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from mpl_toolkits.mplot3d import Axes3D
from feature_match import sift, matching_features, Draw_matches
from fundamental_matrix import get_fundamental_matrix, draw_epilines

# Measona_calib
K1=K2=np.asarray([[1421.9, 0.5, 509.2],
                 [0,   1421.9, 380.2],
                 [0,        0,     1]])

# # Starue_calib
# # Camera A:
# K1=np.array([[5426.566895, 0.678017, 330.096680],
#              [0.000000, 5423.133301, 648.950012],
#              [0.000000,    0.000000,   1.000000]])
# # Camera B:
# K2=np.array([[5426.566895, 0.678017, 387.430023],
#              [0.000000, 5423.133301, 620.616699],
#              [0.000000,    0.000000,   1.000000]])


# def ndarray2matlab(x):
#     return matlab.double(x.tolist())

def compute_essential_matrix(F, K1, K2):
    """
    Compute the essential matrix from the fundamental matrix and intrinsic matrices.
    """
    return K2.T @ F @ K1

def decompose_essential_matrix(E):
    """
    Decompose the essential matrix into four possible [R|t] solutions.
    """
    # SVD of E
    U, S, Vt = svd(E)
    
    # Ensure singular values are [1, 1, 0]
    S_corrected = np.array([1, 1, 0])
    E = U @ np.diag(S_corrected) @ Vt

    # Define W matrix for decomposition
    W = np.array([[0, -1, 0],
                  [1,  0, 0],
                  [0,  0, 1]])
    
    # Compute possible rotations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    
    # Ensure valid rotations (det(R) = +1)
    if np.linalg.det(R1) < 0:
        R1 = -R1
    if np.linalg.det(R2) < 0:
        R2 = -R2
    
    # Translation vector (up to scale)
    t = U[:, 2]
    
    # Return four combinations
    return [(R1, t), (R1, -t), (R2, t), (R2, -t)]

def triangulate_points(P1, P2, pts1, pts2):
    print("Triangulating points...")
    print(f"Projection Matrix P1:\n{P1}")
    print(f"Projection Matrix P2:\n{P2}")
    print(f"Number of points: {pts1.shape[0]}")

    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    
    # Check if points_4d is empty or invalid
    print(f"points_4d shape: {points_4d.shape}")

    try:
        print("Accessing points_4d[3]...")
        w = points_4d[3]
        # Print only the first 10 elements to avoid flooding the console
        print(f"First 10 elements of w: {w[:10]}")
        # Print summary statistics
        print(f"w stats - min: {w.min()}, max: {w.max()}, mean: {w.mean()}, std: {w.std()}")
    except IndexError as e:
        print(f"IndexError: {e}")
        return np.array([])  # Return empty array or handle accordingly
    except Exception as e:
        print(f"Unexpected exception: {e}")
        return np.array([])

    # Continue processing
    print("Processing homogeneous coordinates...")
    print(f"Number of points: {points_4d.shape[1]}")

    # Filter invalid points
    valid_mask = np.abs(w) > 1e-6
    print(f"Number of valid points: {np.sum(valid_mask)}")

    points_4d = points_4d[:, valid_mask]  # Keep only valid points

    if points_4d.size == 0:
        print("No valid points after filtering.")
        return np.array([])

    # Convert from homogeneous to Euclidean coordinates
    points_3d = points_4d[:3] / points_4d[3]
    print("Triangulation completed.")
    return points_3d.T  # Transpose for easier handling

def count_positive_depth(P1, P2, points_3d):
    """
    Count the number of points with positive depth in both camera views.
    """
    # Compute depths in the first camera
    points_cam1 = P1 @ np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T
    z1 = points_cam1[2]

    # Compute depths in the second camera
    points_cam2 = P2 @ np.hstack((points_3d, np.ones((points_3d.shape[0], 1)))).T
    z2 = points_cam2[2]

    # Points are valid if depth is positive in both cameras
    return np.sum((z1 > 0) & (z2 > 0))

def find_best_pose(K1, K2, solutions, pts1, pts2):
    """
    Find the most appropriate [R|t] solution by checking positive depth.
    """
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))  # First camera: P1 = K1[I|0]
    best_pose = None
    max_valid_points = 0

    print("Starting pose selection...")
    for idx, (R, t) in enumerate(solutions):
        print(f"Testing solution {idx + 1} with R:\n{R}\nand t:\n{t}")

        # Projection matrix for the second camera
        P2 = K2 @ np.hstack((R, t.reshape(-1, 1)))  # P2 = K2[R|t]
        
        # Triangulate 3D points
        try:
            points_3d = triangulate_points(P1, P2, pts1, pts2)
            print(f"Triangulated {len(points_3d)} points for solution {idx + 1}")
        except Exception as e:
            print(f"Error during triangulation for solution {idx + 1}: {e}")
            continue

        # Count valid points with positive depth
        valid_points = count_positive_depth(P1, P2, points_3d)
        print(f"Valid points for solution {idx + 1}: {valid_points}")

        # Update the best pose if this solution has more valid points
        if valid_points > max_valid_points:
            max_valid_points = valid_points
            best_pose = (R, t)

    if best_pose is None:
        print("No valid pose found!")
    else:
        print(f"Best pose selected with {max_valid_points} valid points:\n{best_pose}")

    return best_pose

def save_points_to_ply(points_3d, filename):
    """
    Save 3D points to a PLY file.
    """
    with open(filename, 'w') as file:
        # Write PLY header
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points_3d)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("end_header\n")
        
        # Write points
        for point in points_3d:
            file.write(f"{point[0]} {point[1]} {point[2]}\n")
    print(f"3D points saved to {filename}")

def create_mesh(points_3d, filename):
    """
    Create a mesh from 3D points and save as OBJ.
    """
    # Example: Simple convex hull
    mesh = trimesh.convex.convex_hull(points_3d)
    mesh.export(filename)
    print(f"3D mesh saved to {filename}")

def visualize_point_cloud(points_3d):
    """
    Visualize 3D points using matplotlib.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='blue', s=1)
    plt.show()

if __name__=='__main__':
    img1_path = os.path.join('data/Statue1.bmp') 
    img2_path = os.path.join('data/Statue2.bmp') 

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # 1.Feature Matching (SIFT)
    keys1,desc1 = sift(img1)
    keys2,desc2 = sift(img2)
    matches = matching_features(desc1, desc2)
    Draw_matches(matches, img1, img2, keys1, keys2)
    
    # 2. get the Fundamental matrix by correspondence
    pts1 = np.array([(int(keys1[i][0]), int(keys1[i][1])) for i, j in matches])  # 影像1的匹配點
    pts2 = np.array([(int(keys2[j][0]), int(keys2[j][1])) for i, j in matches])  # 影像2的匹配點

    F, inlier_idxs = get_fundamental_matrix(pts1,pts2,threshold=3)
    inlier1 = pts1[inlier_idxs]
    inlier2 = pts2[inlier_idxs]

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 3. draw epipolar lines
    print(f'# correspondence: {len(pts1)}')
    print(f'# inliers: {len(inlier_idxs)}')
    print("Fundamental Matrix F:\n", F)
    draw_epilines(img1, img2, inlier1, inlier2, F, 'epilines.png')

    # 4. four possible solutions of essential matrix
    E = compute_essential_matrix(F, K1, K2)
    print("Essential Matrix (E):\n", E)

    # find the solutions of Essential Matrix
    solutions = decompose_essential_matrix(E)
    for i, (R, t) in enumerate(solutions):
        print(f"Solution {i+1}:")
        print("Rotation (R):\n", R)
        print("Translation (t):\n", t)

    print('afdsssssssssssss')
    # 5. find out the most appropriate solution of essential matrix
    best_pose = find_best_pose(K1, K2, solutions, pts1, pts2)
    print("Best Pose (R, t):", best_pose)

    # 6. get the 3D 
    R, t = best_pose  # Decompose best_pose into R (rotation) and t (translation)

    # Compute projection matrices
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))  # First camera: P1 = K1[I|0]
    P2 = K2 @ np.hstack((R, t.reshape(-1, 1)))          # Second camera: P2 = K2[R|t]

    # Triangulate 3D points using the correct camera pose
    points_3d = triangulate_points(P1, P2, pts1, pts2)

    # Output the 3D points
    print("3D Points:\n", points_3d)

    # 7. get 3D model
    points_3d_path = "output/points.ply"
    save_points_to_ply(points_3d, points_3d_path)
    print(f"3D points已存到 {points_3d_path}")

    mesh_path = "output/mesh.obj"
    create_mesh(points_3d, mesh_path)
    visualize_point_cloud(points_3d)