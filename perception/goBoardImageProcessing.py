import uuid
import cv2
import numpy as np
import scipy.spatial as spatial
import scipy.cluster as cluster
from collections import defaultdict
from statistics import mean
# import tensorflow as tf
# from keras.preprocessing import image

def findContours(thresh):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return sorted(contours, key=cv2.contourArea, reverse=True)[0]

def findApproxCorners(cnt):
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, epsilon, True)

def getDestinationCorners(corners):
    """
        corners - A -> B -> C -> D
    """
    w1 = np.sqrt((corners[0][0] - corners[3][0]) ** 2 + (corners[0][1] - corners[3][1]) ** 2)
    w2 = np.sqrt((corners[1][0] - corners[2][0]) ** 2 + (corners[1][1] - corners[2][1]) ** 2)
    w = max(int(w1), int(w2))

    h1 = np.sqrt((corners[0][0] - corners[1][0]) ** 2 + (corners[0][1] - corners[1][1]) ** 2)
    h2 = np.sqrt((corners[2][0] - corners[3][0]) ** 2 + (corners[2][1] - corners[3][1]) ** 2)
    h = max(int(h1), int(h2))

    destination_corners = np.float32([(0, h - 1), (0, 0), (w - 1, 0), (w - 1, h - 1)])
    
    # print('\nThe destination points are: \n')
    for index, c in enumerate(destination_corners):
        character = chr(65 + index) + "'"
        # print(character, ':', c)
        
    # print('\nThe approximated height and width of the original image is: \n', (h, w))
    return destination_corners, h, w

def unwarp(img, src, dst, w, h):
    # print(img.shape)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    # print('\nThe homography matrix is: \n', H)
    un_warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)
    return un_warped

# Canny edge detection
def cannyEdge(img, sigma=0.33):
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edges = cv2.Canny(img, lower, upper)
    return edges

# Hough line detection
def houghLine(edges, min_line_length=100, max_line_gap=10):
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 125, min_line_length, max_line_gap)
    return lines

# Separate line into horizontal and vertical
def horizontalVerticalLines(lines):
    h_lines, v_lines = [], []
    for line in lines:
        rho, theta = line[0]
        if theta < np.pi / 4 or theta > np.pi - np.pi / 4:
            v_lines.append([rho, theta])
        else:
            h_lines.append([rho, theta])
    return h_lines, v_lines

# Find the intersections of the lines
def lineIntersections(h_lines, v_lines):
    points = []
    for r_h, t_h in h_lines:
        for r_v, t_v in v_lines:
            a = np.array([[np.cos(t_h), np.sin(t_h)], [np.cos(t_v), np.sin(t_v)]])
            b = np.array([r_h, r_v])
            inter_point = np.linalg.solve(a, b)
            points.append(inter_point)
    return np.array(points)

# Hierarchical cluster (by euclidean distance) intersection points
def clusterPoints(points):
    dists = spatial.distance.pdist(points)
    single_linkage = cluster.hierarchy.single(dists)
    flat_clusters = cluster.hierarchy.fcluster(single_linkage, 15, 'distance')
    cluster_dict = defaultdict(list)
    for i in range(len(flat_clusters)):
        cluster_dict[flat_clusters[i]].append(points[i])
    cluster_values = cluster_dict.values()
    clusters = map(lambda arr: (np.mean(np.array(arr)[:, 0]), np.mean(np.array(arr)[:, 1])), cluster_values)
    return sorted(list(clusters), key=lambda k: [k[1], k[0]])

# Average the y value in each row and augment original points
def augmentPoints(points):
    points_shape = list(np.shape(points))
    augmented_points = []
    for row in range(int(points_shape[0] / 10)):
        start = row * 10
        end = (row * 10) + 9
        rw_points = points[start:end + 1]
        rw_y = []
        rw_x = []
        for point in rw_points:
            x, y = point
            rw_y.append(y)
            rw_x.append(x)
        y_mean = mean(rw_y)
        for i in range(len(rw_x)):
            point = (rw_x[i], y_mean)
            augmented_points.append(point)
    augmented_points = sorted(augmented_points, key=lambda k: [k[1], k[0]])
    return augmented_points

# Get Stone color
def getStoneColor(img, x, y, extract_size=5, color="empty"):
    analyse_area = img[x - extract_size : x + extract_size, 
                        y - extract_size : y + extract_size]
    
    average_color_per_row = np.average(analyse_area, axis=0)
    average_color = np.average(average_color_per_row, axis=0)

    print(average_color)

    # cv2.imshow('analyse_area', analyse_area)

    # if color == "empty":
    #     cv2.imwrite(f"image/sample/training/empty/{uuid.uuid1()}.jpg", analyse_area)
    # elif color == "white":
    #     cv2.imwrite(f"image/sample/training/white/{uuid.uuid1()}.jpg", analyse_area)
    # elif color == "black":
    #     cv2.imwrite(f"image/sample/training/black/{uuid.uuid1()}.jpg", analyse_area)

    constraints = np.load('constraints.npy')
    if average_color[0] < constraints[0]: # Black stones.
        cv2.circle(img, (y, x), radius=5, color=(153, 255, 51), thickness=-1) # The coordinates are y then x, so the sequence needs to be reversed here.
        return 'black'
    elif average_color[0] > constraints[1]: # White stones.
        cv2.circle(img, (y, x), radius=5, color=(102, 255, 255), thickness=-1)
        return 'white'
    else: # Empty intersections.
        cv2.circle(img, (y, x), radius=5, color=(0, 0, 255), thickness=-1)
        return 'empty'

# def getStoneColorCNN(src, x, y, extract_size=15):
#     analyse_area = src[x - extract_size : x + extract_size, 
#                         y - extract_size : y + extract_size]

#     reconstructed_model = tf.keras.models.load_model("1.h5")

#     img = tf.image.resize(analyse_area, (30, 30))
#     inputs = image.img_to_array(img)
#     inputs = np.expand_dims(inputs, axis=0)

#     images = np.vstack([inputs])
#     classes = reconstructed_model.predict(images)
#     if classes[0][0] > 0.8:
#         cv2.circle(src, (y, x), radius=5, color=(153, 255, 51), thickness=-1)
#         return 'black'  
#     elif classes[0][2] > 0.8:
#         cv2.circle(src, (y, x), radius=5, color=(102, 255, 255), thickness=-1)
#         return 'white'  
#     else:
#         cv2.circle(src, (y, x), radius=5, color=(0, 0, 255), thickness=-1)
#         return 'empty'
    
