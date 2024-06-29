import mediapipe as mp
import cv2
import time

class HandLandmarks:
    def __init__(self):
        self.landmarks = {} # Initialize an empty dictionary to store landmarks

    @staticmethod
    def update_landmarks(landmarks):
        HandLandmarks.landmarks = landmarks  # Static method to update the landmarks in the class


class HandTracking:
    def __init__(self):
        self.mpHands = mp.solutions.hands # mediapipe Hands module for hand tracking
        self.hands = self.mpHands.Hands()  # Creating a hands object for detecting hands
        self.mpDraw = mp.solutions.drawing_utils # mediapipe drawing utilities for rendering landmarks
        self.previous_hand_position = None  # Variable for saving previous position of the hand
        self.swipe_in_progress = False  # State variable for identifying if a swipe gesture is in progress
        self.min_swipe_distance = 0.1  # Minimum distance (as a fraction of the image width) for swipe detection
        self.last_gesture_time = 0  # Track the time of the last gesture

    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert the BGR image to RGB
        self.results = self.hands.process(imgRGB) # Process the image to detect hands using mediapipe
        # Draw hand landmarks
        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img  # Return the image with drawn landmarks

    # Checking for swipe gesture
    def check_handgesture(self, handLms, img_width):
        if len(handLms.landmark) > mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP:
            current_position = handLms.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]

            if self.previous_hand_position is not None:
                horizontal_movement = current_position.x - self.previous_hand_position.x
                vertical_movement = current_position.y - self.previous_hand_position.y

                min_movement_in_pixels = self.min_swipe_distance * img_width

                # Check if horizontal movement meets swipe criteria
                if abs(horizontal_movement * img_width) > min_movement_in_pixels and abs(horizontal_movement) > abs(vertical_movement):
                    if horizontal_movement > 0 and not self.swipe_in_progress:
                        # hand swiping from left to right
                        self.previous_hand_position = current_position
                        self.swipe_in_progress = True  # Start of swipe movement detected
                        self.last_gesture_time = time.time()  # Record the time of the gesture
                        return True  # Gesture detected
                    elif horizontal_movement <= 0:
                        # hand stops moving to the right
                        self.swipe_in_progress = False  # End of swipe movement
                        self.previous_hand_position = current_position
                        return False
                else:
                    self.previous_hand_position = current_position
                    return False
            else:
                self.previous_hand_position = current_position
                return False

    def close_window(self, handLms):
        mp_hands = mp.solutions.hands # mediapipe Hands module
        # Pinch detection
        if len(handLms.landmark) > mp_hands.HandLandmark.MIDDLE_FINGER_TIP:
            middle_finger_tip = handLms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            thumb_tip = handLms.landmark[mp_hands.HandLandmark.THUMB_TIP]
            distance = ((middle_finger_tip.x - thumb_tip.x) ** 2 + (middle_finger_tip.y - thumb_tip.y) ** 2) ** 0.5
            if distance < 0.03:
                return True #Return True if pinch distance is less than threshold




    



