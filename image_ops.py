import cv2
import numpy as np

camera = cv2.VideoCapture(0)

sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

min_YCrCb = np.array([0,133,77],np.uint8)
max_YCrCb = np.array([255,173,127],np.uint8)

def main():

    while True:

        _, frame = camera.read()

        blurred_frame = cv2.GaussianBlur(frame, (11, 11), cv2.BORDER_DEFAULT)
        sharpened_frame = cv2.filter2D(frame, -1, sharpen_kernel)

        b, g, r = cv2.split(frame)

        """ hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB) """
        """ ycrcb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

        ycrcb_mask = cv2.inRange(ycrcb_frame, min_YCrCb, max_YCrCb)
        skin_region = cv2.bitwise_and(frame, frame, mask = ycrcb_mask) """

        cv2.imshow("Frame", frame)
        cv2.imshow("Blurred Frame", blurred_frame)
        cv2.imshow("Blue", b)
        cv2.imshow("Green", g)
        cv2.imshow("Red", r)

        """ cv2.imshow("Sharpened Frame", sharpened_frame)
        cv2.imshow("Ycrcb Mask", ycrcb_mask)
        cv2.imshow("Skin Region", skin_region) """

        """ cv2.imshow("HSV Frame", hsv_frame)
        cv2.imshow("LAB Frame", lab_frame)
        cv2.imshow("YCrCb Frame", ycrcb_frame) """

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
    camera.release()
    cv2.destroyAllWindows()