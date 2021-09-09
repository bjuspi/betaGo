import cv2
import goboardImageProcessing

CAM_INDEX = 1
WINDOW_TRANSFORMED = "Transformed"

capture = cv2.VideoCapture(CAM_INDEX)

if capture.isOpened():
    captureVal, frame = capture.read()
else:
    captureVal = False
    print("Cannot open the camera of index " + str(CAM_INDEX) + ".")

while captureVal:
    processedImg = goboardImageProcessing.processCamImg(frame)
    
    if processedImg[0] is True:
        transformed = processedImg[1]
        cv2.namedWindow(WINDOW_TRANSFORMED)
        cv2.moveWindow(WINDOW_TRANSFORMED, 400, 330)
        cv2.imshow(WINDOW_TRANSFORMED, transformed)

    captureVal, frame = capture.read()

    if cv2.waitKey(1) == 27: break # Exit on ECS

cv2.destroyAllWindows()