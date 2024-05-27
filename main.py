import cv2
import os
import numpy as np
import time
from cvzone.SelfiSegmentationModule import SelfiSegmentation
from HandTracking import HandTracking
from HandTracking import HandLandmarks
from PersonDetector import PersonDetector
from Timer import Timer

# Global variables
vc = cv2.VideoCapture(0)
segmentor = SelfiSegmentation()
width, height = 1280, 720
windowName = "Mimicry"
mirror_coords = (320, 125, 195, 385)  # x, y, width, height
images = []

curator_img = cv2.imread("./assets/curator.png", cv2.IMREAD_UNCHANGED)  # Loading picture with alpha channel
curator_img = cv2.resize(curator_img, (width, height))  # First picture in same size as camera

vc.set(3, width)  # 3 stands for width
vc.set(4, height)  # 4 stands for height

# Center crop the image
def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]
    crop_width = min(dim[0], img.shape[1])
    crop_height = min(dim[1], img.shape[0])
    mid_x, mid_y = width // 2, height // 2
    return img[mid_y - crop_height // 2: mid_y + crop_height // 2, mid_x - crop_width // 2: mid_x + crop_width // 2]

# Read images from folder
def read_images(folder="images"):
    path = folder
    for filename in os.listdir(path):
        img = cv2.imread(f"{path}/{filename}")
        if img is not None:
            crop_bg = center_crop(img, (width, height))
            resized_img = cv2.resize(crop_bg, (width, height))
            images.append(resized_img)
    if not images:
        print("No images found in image folder")

# Apply perspective transformation
def apply_perspective_transform(img):
    pts1 = np.float32([[0, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1], 0]])
    pts2 = np.float32([[0, 0], [0, 384], [194, 368], [186, 39]])  # Transform points
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (195, 385))


def main():
    read_images()
    hand_tracking = HandTracking()
    gesture_detected = False
    exit_gesture_detected = False
    change_detected = False
    #timer = Timer()
    #wait_counter = 5
    #current_time = timer.current_timer()
    person_detector = PersonDetector()

    show_curator = True
    img_idx = 0

    while True:
        #t = timer.start()

        success, camera_img = vc.read()
        if not success:
            print("Failed to read camera")
            break

        # Detect hand landmarks without drawing them
        camera_img = hand_tracking.find_hands(camera_img, draw=False)
        current_gesture = False

        # Check for hand gestures only when not showing the curator
        if not show_curator and hand_tracking.results.multi_hand_landmarks:
            current_gesture = any(hand_tracking.check_handgesture(landmark) for landmark in hand_tracking.results.multi_hand_landmarks)

        if current_gesture and not gesture_detected:
            # Change background only if a new gesture is detected and wasn't detected previously
            img_idx = (img_idx + 1) % len(images)
            show_curator = False
            gesture_detected = True  # Set the flag as gesture handled

        if not current_gesture:
            gesture_detected = False

        # Resize and transform image
        camera_img_resized = cv2.resize(camera_img, (mirror_coords[2], mirror_coords[3]))
        camera_img_resized = apply_perspective_transform(camera_img_resized)
        x_start, y_start = mirror_coords[0], mirror_coords[1]   # Top-left corner (x, y)
        x_end, y_end = x_start + mirror_coords[2], y_start + mirror_coords[3]      # Bottom-right corner (x, y)
        sub_image = curator_img[y_start:y_end, x_start:x_end]
        sub_image = sub_image[:, :, :3]
        
        # render out black pixels
        camera_img_resized = segmentor.removeBG(camera_img_resized, sub_image, cutThreshold=0.8)
        black_pixels_mask = np.all(camera_img_resized == [0, 0, 0], axis=-1)
        camera_img_resized[black_pixels_mask] = sub_image[black_pixels_mask]
        

        if show_curator:
            alpha_channel = np.ones(camera_img_resized.shape[:2], dtype=np.uint8) * 255
            camera_img_resized = cv2.merge([camera_img_resized[:, :, 0], camera_img_resized[:, :, 1], camera_img_resized[:, :, 2], alpha_channel])

            curator_copy = curator_img.copy()
            roi = curator_copy[mirror_coords[1]: mirror_coords[1] + mirror_coords[3], mirror_coords[0]: mirror_coords[0] + mirror_coords[2]]

            # Ensure the mask dimensions match the ROI dimensions
            mask = cv2.resize(cv2.bitwise_not(camera_img_resized[:, :, 3]), (roi.shape[1], roi.shape[0]))
            roi_bg = cv2.bitwise_and(roi, roi, mask=mask)
            roi_fg_resized = cv2.resize(camera_img_resized, (roi.shape[1], roi.shape[0]))
            roi_fg = cv2.bitwise_and(roi_fg_resized, roi_fg_resized, mask=roi_fg_resized[:, :, 3])

            # Add the masked foreground and background images
            dst = cv2.add(roi_bg, roi_fg)
            curator_copy[mirror_coords[1]: mirror_coords[1] + mirror_coords[3], mirror_coords[0]: mirror_coords[0] + mirror_coords[2]] = dst
            display_image = curator_copy

        else:
            imgBg = images[img_idx]
            display_image = segmentor.removeBG(camera_img, imgBg, cutThreshold=0.45)

        # Fullscreen
        cv2.namedWindow(windowName, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(windowName, display_image)

        # Switching to Background Changer by being in frame for 5 seconds
       # if show_curator:
            #if not change_detected and person_detector.detect_person(camera_img_resized):
                #timer.reset()
                #if t >= current_time + wait_counter:
                    #show_curator = not show_curator
                    #change_detected = True
                    #wait_counter += 10
            #else:
                #change_detected = False
            

        # Closing with gesture
        exit_gesture = False
        if not show_curator and hand_tracking.results.multi_hand_landmarks:
            exit_gesture = any(hand_tracking.close_window(landmark) for landmark in hand_tracking.results.multi_hand_landmarks)
        #if exit_gesture and not exit_gesture_detected:
            # Exit the Background Changer and show curator
            #show_curator = True
            #exit_gesture_detected = True

        if not exit_gesture:
            exit_gesture_detected = False

        key = cv2.waitKey(1)
        # Close window with esc key
        if key == 27:
            break
        # Switching between images with key press 's'
        if key == ord('s'):
            show_curator = not show_curator
        # Switching between images with key press 'a' for previous image and 'd' for next image
        if key == ord('a') and not show_curator:
            img_idx = (img_idx - 1) % len(images)
        elif key == ord('d') and not show_curator:
            img_idx = (img_idx + 1) % len(images)
        # Close window with clicking on 'x'
        if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1:
            break
    
    #timer.stop()
    #timer.reset()

    vc.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

