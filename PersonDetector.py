import cv2
import mediapipe as mp

class PersonDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose                # access MediaPipe's Pose Module
        self.pose = self.mp_pose.Pose()                 # Initialize the Pose object for person detection

    def detect_person(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert image form BGR to RGB 
        results = self.pose.process(image_rgb) # Process image to detect pose landmarks
        return results.pose_landmarks is not None # Return true if pose landmarks are detected, otherwise false