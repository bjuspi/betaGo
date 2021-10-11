import cv2
import numpy as np

CAM_INDEX = 1

WINDOW_ORIGINAL = "Original"
WINDOW_GRAY = "Gray"
WINDOW_THRESH = "Thresh"
WINDOW_TRANSFORMED = "Transformed"
WINDOW_CROPPED = "Cropped"

cv2.namedWindow(WINDOW_ORIGINAL)
cv2.namedWindow(WINDOW_GRAY)
cv2.namedWindow(WINDOW_THRESH)
cv2.namedWindow(WINDOW_TRANSFORMED)
cv2.namedWindow(WINDOW_CROPPED)

cv2.moveWindow(WINDOW_ORIGINAL, 0, 0)
cv2.moveWindow(WINDOW_GRAY, 400, 0)
cv2.moveWindow(WINDOW_THRESH, 800, 0)
cv2.moveWindow(WINDOW_TRANSFORMED, 0, 330)
cv2.moveWindow(WINDOW_CROPPED, 399, 330)

def perspectiveTransform(image, corners):
    def order_corner_points(corners):
        # Separate corners into individual points
        # Index 0 - top-right
        #       1 - top-left
        #       2 - bottom-left
        #       3 - bottom-right
        corners = [(corner[0][0], corner[0][1]) for corner in corners]
        top_r, top_l, bottom_l, bottom_r = corners[0], corners[1], corners[2], corners[3]
        return (top_l, top_r, bottom_r, bottom_l)

    # Order points in clockwise order
    ordered_corners = order_corner_points(corners)
    top_l, top_r, bottom_r, bottom_l = ordered_corners

    # Determine width of new image which is the max distance between 
    # (bottom right and bottom left) or (top right and top left) x-coordinates
    width_A = np.sqrt(((bottom_r[0] - bottom_l[0]) ** 2) + ((bottom_r[1] - bottom_l[1]) ** 2))
    width_B = np.sqrt(((top_r[0] - top_l[0]) ** 2) + ((top_r[1] - top_l[1]) ** 2))
    width = max(int(width_A), int(width_B))

    # Determine height of new image which is the max distance between 
    # (top right and bottom right) or (top left and bottom left) y-coordinates
    height_A = np.sqrt(((top_r[0] - bottom_r[0]) ** 2) + ((top_r[1] - bottom_r[1]) ** 2))
    height_B = np.sqrt(((top_l[0] - bottom_l[0]) ** 2) + ((top_l[1] - bottom_l[1]) ** 2))
    height = max(int(height_A), int(height_B))

    # Construct new points to obtain top-down view of image in 
    # top_r, top_l, bottom_l, bottom_r order
    dimensions = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], 
                    [0, height - 1]], dtype = "float32")

    # Convert to Numpy format
    ordered_corners = np.array(ordered_corners, dtype="float32")

    # Find perspective transform matrix
    matrix = cv2.getPerspectiveTransform(ordered_corners, dimensions)

    # Return the transformed image
    return cv2.warpPerspective(image, matrix, (width, height))

def imagePerspectiveTransform(frame, thresh):
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
            area = cv2.contourArea(c)
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.015 * peri, True)

            # For finding a proper area constraint after changing input image's resolution or the board size.
            if len(approx) == 4:
                print("Current area is " + str(area) + ".")

            # Comment out this if the constraint is not determined yet
            if 60000 < area < 100000 and len(approx) == 4: 
                transformed = perspectiveTransform(frame, approx)
                return cv2.resize(transformed, (300, 300))
            
            return None

capture = cv2.VideoCapture(CAM_INDEX)

if capture.isOpened():
    captureVal, frame = capture.read()
else:
    captureVal = False
    print("Cannot open the camera of index " + str(CAM_INDEX) + ".")

while captureVal:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,40,255, cv2.THRESH_BINARY_INV)[1]

    frame = cv2.resize(frame, (400, 300))
    gray = cv2.resize(gray, (400, 300))
    thresh = cv2.resize(thresh, (400, 300))

    transformed = imagePerspectiveTransform(frame, thresh)
    
    cv2.imshow(WINDOW_ORIGINAL, frame)
    cv2.imshow(WINDOW_GRAY, gray)
    cv2.imshow(WINDOW_THRESH, thresh)

    # Comment out this if the constraint is not determined yet
    if transformed is not None:
        # Determine how much should be cropped
        cropped = transformed[5:295, 5:295]
        cropped = cv2.resize(cropped, (300, 300))
        cv2.imshow(WINDOW_TRANSFORMED, transformed)
        cv2.imshow(WINDOW_CROPPED, cropped)

    captureVal, frame = capture.read()

    if cv2.waitKey(1) == 27: break # Exit on ESC

cv2.destroyAllWindows()