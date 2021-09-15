import cv2
import numpy
from numpy.lib.function_base import append
import goBoardImageProcessing as gbip

EXTRACT_AREA_SIDE_LENGTH = 8

'''
Using EXTRACT_AREA_SIDE_LENGTH = 8:

    image 2:
            black: 14 
            white: 49
    black:
            [16.26171875 11.921875    7.1875    ]
    white:
            [163.171875  158.8125    130.2734375]
    board:  
            corner:     [108.23828125 120.52734375 116.21875   ]
            side:       [110.5546875  116.95703125 109.2734375 ]
            crossing:   [111.67578125 115.5078125  108.8203125 ]
'''

WINDOW_IMAGE = 'Image'
cv2.namedWindow(WINDOW_IMAGE)
cv2.namedWindow(WINDOW_IMAGE)
cv2.moveWindow(WINDOW_IMAGE, 0, 0)

blackPoints = []
whitePoints = []
availablePoints = []

image = cv2.imread('image/sample/from-code/4.jpg')
transformedGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

transformedEdges = gbip.canny_edge(transformedGray)
transformedLines = gbip.hough_line(transformedEdges)
h_lines, v_lines = gbip.h_v_lines(transformedLines)

points = gbip.line_intersections(h_lines, v_lines)
points = gbip.cluster_points(points)
points = gbip.augment_points(points)

for point in points:
    x = int(point[1]) # The crop step requires integer, this could cause issues.
    y = int(point[0])

    analyseArea = image[x - EXTRACT_AREA_SIDE_LENGTH : x + EXTRACT_AREA_SIDE_LENGTH, 
                        y - EXTRACT_AREA_SIDE_LENGTH : y + EXTRACT_AREA_SIDE_LENGTH]
    
    averageColorPerRow = numpy.average(analyseArea, axis=0)
    averageColor = numpy.average(averageColorPerRow, axis=0)

    if averageColor[0] < 30 and averageColor[1] < 30 and averageColor[2] < 30:
        blackPoints.append(point)
        cv2.circle(image, (y, x), radius=EXTRACT_AREA_SIDE_LENGTH, color=(153, 255, 51), thickness=-1) # The coordinates are y then x, so the sequence needs to be reversed here.
    elif averageColor[0] > 125 and averageColor[1] > 125 and averageColor[2] > 125:
        whitePoints.append(point)
        cv2.circle(image, (y, x), radius=EXTRACT_AREA_SIDE_LENGTH, color=(102, 255, 255), thickness=-1)
    else:
        availablePoints.append(point)
        cv2.circle(image, (y, x), radius=EXTRACT_AREA_SIDE_LENGTH, color=(0, 0, 255), thickness=-1)

print('Black points:', blackPoints)
print('White points:', whitePoints)
    
while True:
    cv2.imshow(WINDOW_IMAGE, image)
    if cv2.waitKey(1) == 27: break
cv2.destroyAllWindows()

