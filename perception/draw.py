import numpy as np
import cv2

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
