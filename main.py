import cv2
import os
import numpy as np
import time
from SelfiSegmentationModule import SelfiSegmentation
from HandTracking import HandTracking
from HandTracking import HandLandmarks
from PersonDetector import PersonDetector
from Timer import Timer
from Filter import Filter

#############################################################################################################
# Global variables                                                                                          #
#############################################################################################################

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


#############################################################################################################
# FUNCTIONS                                                                                                 #
#############################################################################################################
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


# Function to create morph effect between two images
def morph_images(img1, img2, steps=20):
    # Resize images to have the same dimensions
    img1_resized = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

    # Convert img1_resized to RGBA if it is not already
    if img1_resized.shape[2] == 3:
        img1_resized = cv2.cvtColor(img1_resized, cv2.COLOR_BGR2BGRA)

    # Convert img2 to RGBA if it is not already
    if img2.shape[2] == 3:
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2BGRA)

    for i in range(steps + 1):
        alpha = i / steps
        # Linear interpolation between the two images
        morphed_img = cv2.addWeighted(img1_resized, 1 - alpha, img2, alpha, 0)
        
        # Apply dream sequence effect
        # Apply Gaussian blur to the image
        blurred_img = cv2.GaussianBlur(morphed_img, (15, 15), sigmaX=5, sigmaY=5)
        
        # Add a wavy distortion effect
        rows, cols = blurred_img.shape[:2]
        scale = 10  # Adjust scale for intensity of distortion
        wave_shift = scale * np.sin(2 * np.pi * i / steps)
        M = np.float32([[1, 0, wave_shift], [0, 1, 0]])
        distorted_img = cv2.warpAffine(blurred_img, M, (cols, rows))
        
        # Show the morphed image with dream sequence effect
        cv2.imshow(windowName, distorted_img)
        cv2.waitKey(50)  # Adjust delay as needed





# Helper Function to make black lines in the Filter transparent
def make_black_lines_transparent(img):
    # Check if the image has an alpha channel, if not add one
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    # Create a mask where the black pixels (0, 0, 0) will be transparent
    black_pixels = np.all(img[:, :, :3] == [0, 0, 0], axis=-1)
    
    # Set the alpha channel to 0 (transparent) for black pixels
    img[black_pixels, 3] = 0

    return img

# Filter Function for Starry Night
def apply_starry_night_filter(img):
        img = Filter.apply_duotone_and_cartoon(img, Filter.farbe_gelb, Filter.farbe_dunkelblau, 50, 20)
        #img = make_black_lines_transparent(img)
        return img

# Filter Function for the Scream
def apply_the_scream_filter(img):
        img = Filter.apply_cartoon(img)
        #img = make_black_lines_transparent(img)
        return img

# Filter Function for un Dimanche
def apply_un_dimanche_filter(img):
    img = Filter.apply_pencilsketch_and_duotone(img, Filter.farbe_gelb, Filter.farbe_dunkelgruen)
    return img



#############################################################################################################
# MAIN                                                                                                      #
#############################################################################################################

def main():
    
    # 0. read background images (art) from folder
    read_images()
    
    # 1. Set variables and objects
    hand_tracking = HandTracking()                      # Handtracking object
    gesture_detected = False                            # flag for gesture
    exit_gesture_detected = False                       # flag for endgesture 
    change_detected = False                             # ?
    person_detector = PersonDetector()                  # Person detecting object
    filter = Filter()                                   # Filter object 
    person_timer = Timer()                              # Timer object
    person_detected_duration = 0                        # ?
    show_curator = True                                 # Flag for curator scene
    img_idx = 0                                         # Index of the background img (art)



    # 2. MAIN Loop 
    while True:
        
        ############################################################################################
        # Get the current frame from the webcam, else error                                            
        success, camera_img = vc.read()
        if not success:
            print("Failed to read camera")
            break
        ############################################################################################
        # Hand detecting and gestures
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
        ############################################################################################
        # Mirror scene 
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
            # Convert the green area to transparent
            alpha_mask = np.where((camera_img_resized[:,:,0] == 0) & (camera_img_resized[:,:,1] == 0) & (camera_img_resized[:,:,2] == 0), 0, 255).astype(np.uint8)

            # Merge the alpha mask with the person image
            camera_img_resized = cv2.merge((camera_img_resized[:,:,0], camera_img_resized[:,:,1], camera_img_resized[:,:,2], alpha_mask))

            # Erzeuge die Distanztransformation
            dist_transform = cv2.distanceTransform(alpha_mask, cv2.DIST_L2, 5)
            gradient = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

            # Invertiere den Farbverlauf (umkehren)
            gradient = 1 - gradient

            # Erstelle den Farbverlauf (schwarz nach transparent)
            gradient = (gradient * 25).astype(np.uint8)
            gradient = cv2.merge((gradient, gradient, gradient, gradient))

            # Wende den Farbverlauf auf die Person an
            alpha_gradient = gradient[:, :, 3]
            alpha_gradient = cv2.cvtColor(alpha_gradient, cv2.COLOR_GRAY2BGRA)
            alpha_gradient[:, :, 3] = gradient[:, :, 3]

            camera_img_resized = cv2.addWeighted(camera_img_resized, 1, alpha_gradient, -1, 0)
            
            #alpha_channel = np.ones(camera_img_resized.shape[:2], dtype=np.uint8) * 255
            #camera_img_resized = cv2.merge([camera_img_resized[:, :, 0], camera_img_resized[:, :, 1], camera_img_resized[:, :, 2], alpha_channel])

            curator_copy = curator_img.copy()
            roi = curator_copy[mirror_coords[1]: mirror_coords[1] + mirror_coords[3], mirror_coords[0]: mirror_coords[0] + mirror_coords[2]]

            # Ensure the mask dimensions match the ROI dimensions
            mask = cv2.resize(cv2.bitwise_not(camera_img_resized[:, :, 3]), (roi.shape[1], roi.shape[0]))
            roi_bg = cv2.bitwise_and(roi, roi, mask=mask)
            roi_fg_resized = cv2.resize(camera_img_resized, (roi.shape[1], roi.shape[0]))
            roi_fg = cv2.bitwise_and(roi_fg_resized, roi_fg_resized, mask=roi_fg_resized[:, :, 3])

            # Add the masked foreground and background images
            dst = cv2.add(roi_bg, roi_fg_resized)
            curator_copy[mirror_coords[1]: mirror_coords[1] + mirror_coords[3], mirror_coords[0]: mirror_coords[0] + mirror_coords[2]] = dst
            display_image = curator_copy

        ###############################################################################################
        # if the curator scene is not shown, means we see the segmented human in front of the art
        # depending on the art a certain filter is applied to the segmented human 
        # 
        # Kommentar: Noch sind schwarze Linien nicht transparent 
        else:
            
            # img_indx is the index number for each art ( 0 = starry night, 1 = the scream, 2 = un dimanche)
            # imgBG is the img of the art
            imgBg = images[img_idx]       
            
            '''
            # Check if the background and cameraimg have an alpha channel, if not add one
            if imgBg.shape[2] == 3:
                imgBg = cv2.cvtColor(imgBg, cv2.COLOR_BGR2BGRA)          
                
            if camera_img.shape[2] == 3:
                camera_img = cv2.cvtColor(camera_img, cv2.COLOR_BGR2BGRA)            
            '''
            
            # Segmenting the human from the bg ( = art)
            segmented_img, condition = segmentor.get_segments(camera_img, imgBg, cutThreshold=0.45)

            '''
            # Make sure the segmented img has alpha channel
            if segmented_img.shape[2] == 3:
                segmented_img = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2BGRA)
            '''
            
            # Apply filters
            if img_idx == 0:
                segmented_img = apply_starry_night_filter(segmented_img)
            elif img_idx == 1:
                segmented_img = apply_the_scream_filter(segmented_img)
            elif img_idx == 2:
                segmented_img = apply_un_dimanche_filter(segmented_img)

            # Combine the segmented img ( = human plus filter) with the background img ( = art) 
            display_image = segmentor.combine(condition, segmented_img,  imgBg, cutThreshold=0.45)
            
            '''
            # Make sure the displayed img has alpha channel 
            if display_image.shape[2] == 3:
                display_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2BGRA)
            '''
        ###############################################################################################

        # Fullscreen
        cv2.namedWindow(windowName, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(windowName, display_image)
        
        ############################################################################################
        # Switching to Background Changer by being in frame for 5 seconds
        if show_curator:
            if person_detector.detect_person(camera_img):
                if person_detected_duration == 0:
                    person_timer.start()
                person_detected_duration = person_timer.elapsed_time()
                if person_detected_duration >= 5:
                    # Start morph effect
                    morph_images(curator_img, images[0])
                    show_curator = False
                    person_detected_duration = 0
                    person_timer.reset()
            else:
                person_detected_duration = 0
                person_timer.reset()
        else:
            if person_detector.detect_person(camera_img):
                person_detected_duration = 0
                person_timer.reset()
        ############################################################################################  
        # Closing with gesture
        exit_gesture = False
        if not show_curator and hand_tracking.results.multi_hand_landmarks:
            exit_gesture = any(hand_tracking.close_window(landmark) for landmark in hand_tracking.results.multi_hand_landmarks)
        if exit_gesture and not exit_gesture_detected:
            # Exit the Background Changer and show curator
            show_curator = True
            exit_gesture_detected = True
            person_detected_duration = 0
            person_timer.reset()

        if not exit_gesture:
            exit_gesture_detected = False

        ############################################################################################
        # Keyboard 
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
    
    

    # Aufräumen
    vc.release()
    cv2.destroyAllWindows()


# MAIN AUFRUF
if __name__ == "__main__":
    main()
