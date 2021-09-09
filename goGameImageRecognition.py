import cv2
import goboardImageProcessing

CAM_INDEX = 1

WINDOW_ORIGINAL = "Original"
WINDOW_BILATERAL = "Bilateral"
WINDOW_GRAY = "Gray"
WINDOW_TRESH = "Tresh"
WINDOW_TRANSFORMED = "Transformed"

capture = cv2.VideoCapture(CAM_INDEX)

if capture.isOpened():
    captureVal, frame = capture.read()
else:
    captureVal = False
    print("Cannot open the camera of index " + str(CAM_INDEX) + ".")

while captureVal:
    blur = cv2.bilateralFilter(frame,9,75,75)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,40,255, cv2.THRESH_BINARY_INV)[1]

    frame = cv2.resize(frame, (400, 300))
    blur = cv2.resize(blur, (400, 300))
    gray = cv2.resize(gray, (400, 300))
    thresh = cv2.resize(thresh, (400, 300))

    transformed = goboardImageProcessing.imagePerspectiveTransform(frame, thresh)
    
    if transformed is not None:
        cv2.namedWindow(WINDOW_TRANSFORMED)
        cv2.moveWindow(WINDOW_TRANSFORMED, 400, 330)
        cv2.imshow(WINDOW_TRANSFORMED, transformed)

    cv2.namedWindow(WINDOW_ORIGINAL)
    cv2.namedWindow(WINDOW_BILATERAL)
    cv2.namedWindow(WINDOW_GRAY)
    cv2.namedWindow(WINDOW_TRESH)

    cv2.moveWindow(WINDOW_ORIGINAL, 0, 0)
    cv2.moveWindow(WINDOW_BILATERAL, 400, 0)
    cv2.moveWindow(WINDOW_GRAY, 800, 0)
    cv2.moveWindow(WINDOW_TRESH, 0, 330)

    cv2.imshow(WINDOW_ORIGINAL, frame)
    cv2.imshow(WINDOW_BILATERAL, blur)
    cv2.imshow(WINDOW_GRAY, gray)
    cv2.imshow(WINDOW_TRESH, thresh)

    captureVal, frame = capture.read()

    if cv2.waitKey(1) == 27: break # Exit on ECS

cv2.destroyAllWindows()