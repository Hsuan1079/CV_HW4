import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

## Draw epipolar lines
def norm_line(lines):
    a = lines[0,:]
    b = lines[1,:]
    length = np.sqrt(a**2 + b**2)
    return lines / length

def draw_epilines(gray1, gray2, inlier1, inlier2, F, filename):
    # inliner2 transfter by F will be the epipolar lines in the first image
    lines1_unnorm= F @ np.hstack((inlier2,np.ones((inlier2.shape[0],1)))).T
    lines1 = norm_line(lines1_unnorm)
    img1, img2 = drawlines(gray1, gray2, lines1.T, inlier1.astype(int), inlier2.astype(int))

    lines2_unnorm = F.T @ np.hstack((inlier1,np.ones((inlier1.shape[0],1)))).T
    lines2 = norm_line(lines2_unnorm)
    img3, img4 = drawlines(gray2, gray1, lines2.T, inlier2.astype(int), inlier1.astype(int))

    plt.subplot(221), plt.imshow(img1)
    plt.subplot(222), plt.imshow(img2)
    plt.subplot(223), plt.imshow(img4)
    plt.subplot(224), plt.imshow(img3)

    # save it to the output folder
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath)
    print(f"圖像已保存到 {filepath}")

def drawlines(img1, img2, lines, pts1, pts2):
    r,c = img1.shape
    # change the gray image to RGB image
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)

    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1] ])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

## Find the best Fundamental matrix
def normalize_coordinate(points):
    x = points[0]
    y = points[1]
    center = points.mean(axis=1)  # mean of each row
    cx = x - center[0] # center the points
    cy = y - center[1]
    dist = np.sqrt(np.power(cx, 2) + np.power(cy, 2))
    scale = np.sqrt(2) / dist.mean()
    T = np.array([
        [scale, 0, -scale * center[0]],
        [0, scale, -scale * center[1]],
        [0,     0,                  1]
        ])
    return T, T@points

def compute_fundamental_matrix(x,x_):
    # X'^T * F * X = 0
    # X' means the point in the second image, X means the point in the first image
    # A = [x'*x, x'*y, x', y'*x, y'*y, y', x, y, 1]
    A = np.zeros((8,9))
    for i in range(8):
        A[i]=[ x_[0, i]*x[0, i], 
               x_[0, i]*x[1, i], 
               x_[0, i], 
               x_[1, i]*x[0, i], 
               x_[1, i]*x[1, i], 
               x_[1, i],
               x[0, i], 
               x[1, i], 
               1 ]
    
    # Af=0 SVD decomposition
    U, S, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # det(F)=0 constrain
    U, S, V = np.linalg.svd(F)
    S[-1] = 0
    F = U @ np.diag(S) @ V
    return F

def compute_fundamental_matrix_normalized(p1,p2):
    # p1 and P2 are coordinates of corresponding points in two images
    # preprocess image coordinates
    T1,p1_normalized = normalize_coordinate(p1.T)
    T2,p2_normalized = normalize_coordinate(p2.T)

    F = compute_fundamental_matrix(p1_normalized,p2_normalized)

    # denormalize
    F = T2.T @ F @ T1
    return F/F[-1,-1]

def get_fundamental_matrix(X1,X2,threshold):
    rs = np.random.RandomState(seed = 0)
    N=len(X1)
    # 轉換成齊次座標
    X1=np.hstack((X1,np.ones((N,1))))
    X2=np.hstack((X2,np.ones((N,1))))

    best_cost=1e9
    best_F=None
    best_inlier_idxs=None
    # find best F with RANSAC
    for _ in range(2000):
        choose_idx=rs.choice(N, 8, replace=False)  # sample 8 correspondence feature points
        # get F
        F=compute_fundamental_matrix_normalized(X1[choose_idx,:],X2[choose_idx,:])

        # select indices with accepted points, Sampson distance as error.
        Fx1=(X1@F).T
        Fx2=(X2@F).T
        # calculate the Sampson distance
        denom = Fx1[0] ** 2 + Fx1[1] ** 2 + Fx2[0] ** 2 + Fx2[1] ** 2
        errors = np.diag( X2 @ F @ X1.T) ** 2 / denom
        inlier_idxs=np.where(errors<threshold)[0]

        cost = np.sum(errors[errors<threshold]) + (N-len(inlier_idxs))*threshold
        if cost < best_cost:
            best_cost=cost
            best_F=F
            best_inlier_idxs=inlier_idxs

    best_F = best_F.T

    return best_F, best_inlier_idxs