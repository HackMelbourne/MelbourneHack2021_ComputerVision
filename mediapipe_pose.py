import cv2
import mediapipe as mp
import numpy as np

camera = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def main():

    holistic = mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence = 0.5)

    while True:

        _, frame = camera.read()

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        # Draw Face Mesh
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACE_CONNECTIONS)

        # Draw Right Hand
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Draw Left Hand
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Draw Body Pose
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()
    camera.release()
    cv2.destroyAllWindows()