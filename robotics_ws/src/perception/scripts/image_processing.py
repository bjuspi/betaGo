import cv2
import numpy as np

def get_thresh(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)[1]

def find_contours(thresh):
    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
    return sorted(contours, key=cv2.contourArea, reverse=True)[0]

def find_approx_corners(cnt):
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, epsilon, True)

def dist(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def get_destination_corners(corners):
    # corners - A -> B -> C -> D
    w1 = dist(corners[0], corners[3])
    w2 = dist(corners[1], corners[2])
    w = max(int(w1), int(w2))

    h1 = dist(corners[0], corners[1])
    h2 = dist(corners[2], corners[3])
    h = max(int(h1), int(h2))

    destination_corners = np.float32([(0, h - 1), (0, 0), (w - 1, 0), (w - 1, h - 1)])
    
    # print('\nThe destination points are: \n')
    # for index, c in enumerate(destination_corners):
    #     character = chr(65 + index) + "'"
        # print(character, ':', c)
        
    # print('\nThe approximated height and width of the original image is: \n', (h, w))
    return destination_corners, h, w

def unwarp(img, src, dst, w, h):
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    un_warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)
    return un_warped

def get_perspective_tf(img, thresh):
    canvas = img.copy()
    
    cnt = find_contours(thresh)
    approx_corners = find_approx_corners(cnt)

    H, W = thresh.shape

    if len(approx_corners) == 4:
        cv2.drawContours(canvas, cnt, -1, (0, 255, 0), 3)
        cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)

        approx_corners = np.concatenate(approx_corners).tolist()
        ref_corners = [[0, H], [0, 0], [W, 0], [W, H]]
        sorted_corners = []

        for ref_corner in ref_corners:
            x = [dist(ref_corner, corner) for corner in approx_corners]
            min_position = x.index(min(x))
            sorted_corners.append(approx_corners[min_position])

        destination_corners, h, w = get_destination_corners(sorted_corners)
        un_warped = unwarp(img, np.float32(sorted_corners), destination_corners, w, h)
        cropped = un_warped[0:h, 0:w]
        cropped = cv2.resize(cropped, (300, 300))
        cropped = cropped[15:-15, 15:-15]
        return canvas, cropped

    cv2.drawContours(canvas, cnt, -1, (0, 255, 0), 3)
    return canvas, np.zeros((H, W, 3), np.uint8)
