import mediapipe as mp
import cv2


class HandLandmarks:
    def __init__(self):
        self.landmark= {}
    
    @staticmethod
    def update_landmarks(landmarks):
       HandLandmarks.landmarks = landmarks


class HandTracking:

    def __init__(self):
        self.mpHands = mp.solutions.hands                       # Initialize MediaPipe Hands for hand detection and landmark extraction
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils    
        # Initialize variables for gesture tracking
        self.previous_hand_position = None                      # Variable for saving previous position
        self.swipe_in_progress = False                          # state for identifying swipe gesture
    
    def find_hands(self, img, draw=True):
        
        """
        Detects hands in the image and optionally draws landmarks and connections.
        
        Parameters:
        - img: Input image in BGR format.
        - draw: Boolean flag indicating whether to draw landmarks and connections on the image.
        
        Returns:
        - img: Image with drawn landmarks and connections (if draw=True).
        """
        
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        #draw land handmarks
        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    #checking for swipe gesture
    def check_handgesture(self, handLms):
        if len(handLms.landmark) > mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP:
            current_position = handLms.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]

            if self.previous_hand_position is not None:
                horizontal_movement = current_position.x - self.previous_hand_position.x
                vertical_movement = current_position.y - self.previous_hand_position.y

                if abs(horizontal_movement) > abs(vertical_movement):
                    if horizontal_movement < 0 and not self.swipe_in_progress:
                        # hand swiping from left to right
                        self.previous_hand_position = current_position
                        self.swipe_in_progress = True  # Beginn der Wischbewegung erkannt
                        return True  # Geste erkannt
                    elif horizontal_movement >= 0:
                        # hand stops moving to the left
                        self.swipe_in_progress = False  # end of swipe movement
                        self.previous_hand_position = current_position
                        return False
                else:
                    self.previous_hand_position = current_position
                    return False
            else:
                self.previous_hand_position = current_position
                return False

    # Function to close the artwork frame and get back to the curator 
    def close_window(self, handLms):
        mp_hands = mp.solutions.hands           # why here again?
        
        # pinch detection
        if len(handLms.landmark) > mp_hands.HandLandmark.MIDDLE_FINGER_TIP:
            middle_finger_tip = handLms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            thumb_tip = handLms.landmark[mp_hands.HandLandmark.THUMB_TIP]
            distance = ((middle_finger_tip.x - thumb_tip.x) ** 2 + (middle_finger_tip.y - thumb_tip.y) ** 2) ** 0.5
            if distance < 0.03:
                return True



    



