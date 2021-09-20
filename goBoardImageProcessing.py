import math
import cv2
import numpy as np
import scipy.spatial as spatial
import scipy.cluster as cluster
from collections import defaultdict
from statistics import mean

def getTopDownView(thresh, src):
    canvas = src.copy()

    contour = findContours(thresh)
    approx_corners = findApproxCorners(contour)

    if len(approx_corners) != 4:
        return None, None
 
    cv2.drawContours(canvas, contour, -1, (0, 255, 0), 3)
    cv2.drawContours(canvas, approx_corners, -1, (255, 255, 0), 10)
    
    approx_corners = np.concatenate(approx_corners).tolist()

    H, W = thresh.shape
    ref_corners = [[0, H], [0, 0], [W, 0], [W, H]]
    sorted_corners = []

    for ref_corner in ref_corners:
        min_distances = [math.dist(ref_corner, corner) for corner in approx_corners]
        min_position = min_distances.index(min(min_distances))
        sorted_corners.append(approx_corners[min_position])

    print('\nThe corner points are ...\n')
    for index, c in enumerate(sorted_corners):
        character = chr(65 + index)
        print(character, ':', c)
        cv2.putText(canvas, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    destination_corners, h, w = getDestinationCorners(sorted_corners)
    un_warped = unwarp(src, np.float32(sorted_corners), destination_corners, w, h)
    cropped = un_warped[0:h, 0:w]
    cropped = cv2.resize(cropped, (300, 300))
    cropped = cropped[10:290, 10:290]
    return cropped, canvas

def getBoardFrame(top_down_edges, top_down):
    ver_hor_frame = top_down.copy()
    board_frame = top_down.copy()

    lines = houghLine(top_down_edges)
    if (lines is not None):    
        h_lines, v_lines = horizontalVerticalLines(lines)

        if h_lines is not None and v_lines is not None:
            intersection_points = lineIntersections(h_lines, v_lines)
            points = clusterPoints(intersection_points)
            augmented_points = augmentPoints(points)

            for index, point in enumerate(augmented_points):
                x = int(point[1]) # The crop step requires integer, this could cause issues.
                y = int(point[0])
                color = getStoneColor(board_frame, x, y)

            for h_line in h_lines:
                drawLine(ver_hor_frame, h_line, (255, 0, 0))
                
            for v_line in v_lines:
                drawLine(ver_hor_frame, v_line, (0, 255, 0))
    return ver_hor_frame, board_frame

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
    EXTRACT_AREA_SIDE_LENGTH = 5
    analyse_area = img[x - EXTRACT_AREA_SIDE_LENGTH : x + EXTRACT_AREA_SIDE_LENGTH, 
                        y - EXTRACT_AREA_SIDE_LENGTH : y + EXTRACT_AREA_SIDE_LENGTH]
    
    average_color_per_row = np.average(analyse_area, axis=0)
    average_color = np.average(average_color_per_row, axis=0)

    # print(average_color)

    # analyse_gray = cv2.cvtColor(analyse_area, cv2.COLOR_BGR2GRAY)
    # analyse_edges = cannyEdge(analyse_gray)
    # lines = houghLine(analyse_edges, 20)
    # if (lines is None):
    #     print('No lines found')
    # else:
    #     print('Lines found')

    # cv2.imshow('analyse_area', analyse_area)
    # cv2.imshow("test", analyse_gray)
    

    if average_color[0] < 50: # Black stones.
        cv2.circle(img, (y, x), radius=EXTRACT_AREA_SIDE_LENGTH, color=(153, 255, 51), thickness=-1) # The coordinates are y then x, so the sequence needs to be reversed here.
        return 'black'
    elif average_color[0] > 150: # White stones.
        cv2.circle(img, (y, x), radius=EXTRACT_AREA_SIDE_LENGTH, color=(102, 255, 255), thickness=-1)
        return 'white'
    else: # Empty intersections.
        cv2.circle(img, (y, x), radius=EXTRACT_AREA_SIDE_LENGTH, color=(0, 0, 255), thickness=-1)
        return 'empty'

def drawLine(frame, line, color):
    rho, theta = line
    if not np.isnan(rho) and not np.isnan(theta):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(frame, (x1,y1), (x2,y2), color, 2)
