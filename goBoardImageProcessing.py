import cv2
import numpy as np
import scipy.spatial as spatial
import scipy.cluster as cluster
from collections import defaultdict
from statistics import mean

# def perspectiveTransform(image, corners):
#     def order_corner_points(corners):
#         # Separate corners into individual points
#         # Index 0 - top-right
#         #       1 - top-left
#         #       2 - bottom-left
#         #       3 - bottom-right
#         corners = [(corner[0][0], corner[0][1]) for corner in corners]
#         top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
#         return (top_l, top_r, bottom_r, bottom_l)

#     # Order points in clockwise order
#     ordered_corners = order_corner_points(corners)
#     top_l, top_r, bottom_r, bottom_l = ordered_corners

#     # Determine width of new image which is the max distance between 
#     # (bottom right and bottom left) or (top right and top left) x-coordinates
#     width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
#     width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
#     width = max(int(width_A), int(width_B))

#     # Determine height of new image which is the max distance between 
#     # (top right and bottom right) or (top left and bottom left) y-coordinates
#     height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
#     height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
#     height = max(int(height_A), int(height_B))

#     # Construct new points to obtain top-down view of image in 
#     # top_r, top_l, bottom_l, bottom_r order
#     dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], 
#                     [0, height - 1]], dtype = "float32")

#     # Convert to Numpy format
#     ordered_corners = np.array(ordered_corners, dtype="float32")

#     # Find perspective transform matrix
#     matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

#     # Return the transformed image
#     return cv2.warpPerspective(image, matrix, (width, height))

# def imagePerspectiveTransform(frame, thresh):
#     cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     cnts = cnts[0] if len(cnts) == 2 else cnts[1]

#     for c in cnts:
#             area = cv2.contourArea(c)
#             peri = cv2.arcLength(c, True)
#             approx = cv2.approxPolyDP(c, 0.015 * peri, True)

#             # Use findInputConstraintsForCam.py to find a proper area constraint first
#             # This constraint needs to be reset after changing input image's resolution or the board size.
#             if 60000 < area < 100000 and len(approx) == 4: 
#                 transformed = perspectiveTransform(frame, approx)
#                 return cv2.resize(transformed, (300, 300))
            
#             return None

def findContours(thresh):
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return sorted(contours, key=cv2.contourArea, reverse=True)[0]

def findApproxCorners(cnt):
    epsilon = 0.02 * cv2.arcLength(cnt, True)
    return cv2.approxPolyDP(cnt, epsilon, True)

def getDestinationCorners(corners):
    """
        corners - A -> C -> B -> D
    """
    w1 = np.sqrt((corners[0][0] - corners[3][0]) ** 2 + (corners[0][1] - corners[3][1]) ** 2)
    w2 = np.sqrt((corners[1][0] - corners[2][0]) ** 2 + (corners[1][1] - corners[2][1]) ** 2)
    w = max(int(w1), int(w2))

    h1 = np.sqrt((corners[0][0] - corners[1][0]) ** 2 + (corners[0][1] - corners[1][1]) ** 2)
    h2 = np.sqrt((corners[2][0] - corners[3][0]) ** 2 + (corners[2][1] - corners[3][1]) ** 2)
    h = max(int(h1), int(h2))

    destination_corners = np.float32([(0, h - 1), (w - 1, 0), (0, 0), (w - 1, h - 1)])
    
    print('\nThe destination points are: \n')
    for index, c in enumerate(destination_corners):
        character = chr(65 + index) + "'"
        print(character, ':', c)
        
    print('\nThe approximated height and width of the original image is: \n', (h, w))
    return destination_corners, h, w

def unwarp(img, src, dst, w, h):
    print(img.shape)
    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    print('\nThe homography matrix is: \n', H)
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
    for row in range(int(points_shape[0] / 8)):
        start = row * 8
        end = (row * 8) + 7
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
def getStoneColor(img, x, y):
    EXTRACT_AREA_SIDE_LENGTH = 8
    analyse_area = img[x - EXTRACT_AREA_SIDE_LENGTH : x + EXTRACT_AREA_SIDE_LENGTH, 
                        y - EXTRACT_AREA_SIDE_LENGTH : y + EXTRACT_AREA_SIDE_LENGTH]
    
    average_color_per_row = np.average(analyse_area, axis=0)
    average_color = np.average(average_color_per_row, axis=0)

    if average_color[0] < 30 and average_color[1] < 30 and average_color[2] < 30:
        cv2.circle(img, (y, x), radius=5, color=(153, 255, 51), thickness=-1) # The coordinates are y then x, so the sequence needs to be reversed here.
    elif average_color[0] > 125 and average_color[1] > 125 and average_color[2] > 125:
        cv2.circle(img, (y, x), radius=5, color=(102, 255, 255), thickness=-1)
    else:
        cv2.circle(img, (y, x), radius=5, color=(0, 0, 255), thickness=-1)