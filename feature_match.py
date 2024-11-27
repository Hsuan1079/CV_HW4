import cv2
import numpy as np
import os

def sift(img):
    # Get the key points and descriptors
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(img, None)
    kp = np.float32([i.pt for i in kp])
    return kp, des

def matching_features(des1, des2):
    # find the first and second best matches
    ratio_dis = 0.75
    matches = []
    for i, d1 in enumerate(des1):
        dis = []
        for j, d2 in enumerate(des2):
            distance = np.linalg.norm(d1 - d2) # Euclidean distance
            dis.append((distance, j))

        # Sort the distances by the first element of the tuple    
        dis.sort(key=lambda x: x[0])

        # Check if the first distance is less than the second distance
        if (dis[0][0] / dis[1][0]) < ratio_dis:
            matches.append((i, dis[0][1]))
    
    return matches

def Draw_matches(matches, img1, img2, kp1, kp2):
    # Draw the matches linking the two images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    img_matches = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
    img_matches[:h1, :w1] = img1
    img_matches[:h2, w1:] = img2
    for i, j in matches:
        pt1 = (int(kp1[i][0]), int(kp1[i][1]))
        pt2 = (int(kp2[j][0])+w1, int(kp2[j][1]))

        # draw a circle in the keypoints
        cv2.circle(img_matches, pt1, 5, (0, 0, 255), 1)
        cv2.circle(img_matches, pt2, 5, (0, 255, 0), 1)

        # draw a line between the keypoints
        cv2.line(img_matches, pt1, pt2, (255, 0, 0), 1)

    # Save the image
    cv2.imwrite(os.path.join('output', 'matches.jpg'), img_matches)