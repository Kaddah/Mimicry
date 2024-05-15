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
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils
    
    def find_hands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks and draw:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def check_handgesture(self, handLms):
        #for handLms in HandLandmarks.landmarks:
            if len(handLms.landmark) > mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP:
                middle_finger_tip_x = handLms.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].x
                wrist_x = handLms.landmark[mp.solutions.hands.HandLandmark.WRIST].x

                if middle_finger_tip_x > wrist_x:
                    return True  # Indicate that the gesture was detected
                else:
                    return False
                
    def close_window(self, handLms):
        mp_hands = mp.solutions.hands
        if len(handLms.landmark) > mp_hands.HandLandmark.MIDDLE_FINGER_TIP:
                middle_finger_tip = handLms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                thumb_tip = handLms.landmark[mp_hands.HandLandmark.THUMB_TIP]
                distance = ((middle_finger_tip.x - thumb_tip.x) ** 2 + (middle_finger_tip.y - thumb_tip.y) ** 2) ** 0.5
                if distance < 0.03:
                    return True
    



