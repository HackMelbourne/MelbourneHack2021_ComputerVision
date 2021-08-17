import cv2
import numpy as np
import mediapipe as mp
import os

camera = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Load in Mouth Detector
CASCADE_PATH = r"./haar_cascades"
mouth_haarcascade = os.path.join(CASCADE_PATH, "haarcascade_mouth.xml")
mouth_detector = cv2.CascadeClassifier(mouth_haarcascade)

def main():

    face_mesh = mp_face_mesh.FaceMesh(static_image_mode = False, max_num_faces = 1, min_detection_confidence = 0.5)

    while True:

        _, frame = camera.read()

        # Need to convert BGR image to RGB since opencv uses BGR (Blue, Green, Red)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        height = rgb_frame.shape[0]
        width = rgb_frame.shape[1]

        X = []
        Y = []
        min_X = 9999
        max_X = 0
        max_Y = 0

        if results.multi_face_landmarks:
            face_landmark_list = list(results.multi_face_landmarks)
            face_landmark_list = face_landmark_list[0].landmark

            for i in range(len(face_landmark_list)):
                landmark = face_landmark_list[i]

                real_x = round(landmark.x * width)
                real_y = round(landmark.y * height)
                X.append(real_x)
                Y.append(real_y)

                min_X = min(min_X, real_x)
                max_X = max(max_X, real_x)
                max_Y = max(max_Y, real_y)

                #cv2.circle(frame, (real_x, real_y), 2, (0, 0, 255), 2)

        try:
            centroid = (sum(X) // len(X), sum(Y) // len(Y))
            centroid_X = centroid[0]
            centroid_Y = centroid[1]

            cv2.circle(frame, (centroid_X, centroid_Y), 4, (0, 255, 0), 2)

            # Draw Line From Centroid To Edges of Faces
            cv2.line(frame, centroid, (max_X, centroid_Y), (255, 0, 0), 2, cv2.LINE_AA)
            cv2.line(frame, centroid, (min_X, centroid_Y), (255, 0, 0), 2, cv2.LINE_AA)

            cv2.line(frame, (min_X, centroid_Y), (centroid_X, max_Y), (255, 0, 0), 2, cv2.LINE_AA)
            cv2.line(frame, (centroid_X, max_Y), (max_X, centroid_Y), (255, 0, 0), 2, cv2.LINE_AA)

            # Get corners/vertices of Triangle
            roi_corners = np.array([[(min_X, centroid_Y), (centroid_X, max_Y), (max_X, centroid_Y)]], dtype = np.int32)

            mask = np.zeros(frame.shape, dtype = np.uint8)
            channel_count = frame.shape[2]
            ignore_mask_color = (255, ) * channel_count
            cv2.fillPoly(mask, roi_corners, ignore_mask_color)
            mask_region = cv2.bitwise_and(frame, mask)

            mask_region_gray = cv2.cvtColor(mask_region, cv2.COLOR_BGR2GRAY)
            mouth = mouth_detector.detectMultiScale(mask_region_gray, 1.03, 1, minSize = (10, 10))
            """ for x, y, w, h in mouth:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1) """

            if len(mouth):
                text = "No Mask"
            else:
                text = "Mask"

            cv2.putText(frame, text, (max_X, centroid_Y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
            cv2.imshow("Mask Region", mask_region)

        except Exception as e:
            print(e)

        cv2.imshow("Frame", frame)
  

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
    camera.release()
    cv2.destroyAllWindows()