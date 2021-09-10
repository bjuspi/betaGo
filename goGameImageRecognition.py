import cv2
import numpy as np
import goboardImageProcessing

CAM_INDEX = 1

WINDOW_ORIGINAL = "Original"
WINDOW_TRESH = "Tresh"
WINDOW_TRANSFORMED = "Transformed"
WINDOW_CANNY_EDGES = "Canny Edges"
WINDOW_HOUGH_LINES = "Hough Lines"

cv2.namedWindow(WINDOW_ORIGINAL)
cv2.namedWindow(WINDOW_TRESH)
cv2.namedWindow(WINDOW_TRANSFORMED)
cv2.namedWindow(WINDOW_CANNY_EDGES)
cv2.namedWindow(WINDOW_HOUGH_LINES)

cv2.moveWindow(WINDOW_ORIGINAL, 0, 0)
cv2.moveWindow(WINDOW_TRESH, 400, 0)
cv2.moveWindow(WINDOW_TRANSFORMED, 800, 0)
cv2.moveWindow(WINDOW_CANNY_EDGES, 0, 330)
cv2.moveWindow(WINDOW_HOUGH_LINES, 300, 330)

capture = cv2.VideoCapture(CAM_INDEX)

if capture.isOpened():
    captureVal, frame = capture.read()
else:
    captureVal = False
    print("Cannot open the camera of index " + str(CAM_INDEX) + ".")

while captureVal:
    bilateral = cv2.bilateralFilter(frame,9,75,75)
    gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,40,255, cv2.THRESH_BINARY_INV)[1]

    frame = cv2.resize(frame, (400, 300))
    bilateral = cv2.resize(bilateral, (400, 300))
    gray = cv2.resize(gray, (400, 300))
    thresh = cv2.resize(thresh, (400, 300))

    transformed = goboardImageProcessing.imagePerspectiveTransform(frame, thresh)
    
    cv2.imshow(WINDOW_ORIGINAL, frame)
    cv2.imshow(WINDOW_TRESH, thresh)
    
    if transformed is not None:
        # Test how much to be cropped using findInputConstraints.py
        cropped = transformed[10:290, 10:290]
        cropped = cv2.resize(cropped, (300, 300))
        transformedGray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        transformedGrayBlur = cv2.blur(transformedGray, (5, 5))
        transformedEdges = goboardImageProcessing.canny_edge(transformedGrayBlur)
        transformedLines = goboardImageProcessing.hough_line(transformedEdges)
        houghLine = cropped.copy()

        if transformedLines is not None:        
            for line in transformedLines:
                rho,theta = line[0]
                if not np.isnan(rho) and not np.isnan(theta):
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = a*rho
                    y0 = b*rho
                    x1 = int(x0 + 1000*(-b))
                    y1 = int(y0 + 1000*(a))
                    x2 = int(x0 - 1000*(-b))
                    y2 = int(y0 - 1000*(a))
                    cv2.line(houghLine, (x1,y1), (x2,y2), (0,0,255), 2)
        
        cv2.imshow(WINDOW_TRANSFORMED, transformed)
        cv2.imshow(WINDOW_CANNY_EDGES, transformedEdges)
        cv2.imshow(WINDOW_HOUGH_LINES, houghLine)

    captureVal, frame = capture.read()

    if cv2.waitKey(1) == 27: break # Exit on ESC

cv2.destroyAllWindows()