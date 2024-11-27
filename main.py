import os
import numpy as np
import cv2
from feature_match import sift, matching_features, Draw_matches
from fundamentak_matrix import get_fundamental_matrix, draw_epilines
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
    # 5. find out the most appropriate solution of essential matrix
    # 6. get the 3D points
    # 7. get 3D moel