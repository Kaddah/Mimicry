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
import ctypes 

user32 = ctypes.WinDLL('user32', use_last_error=True)
user32.ShowCursor(False)

#############################################################################################################
# Global variables                                                                                          #
#############################################################################################################

vc = cv2.VideoCapture(0)
segmentor = SelfiSegmentation()
width, height = 1280, 720
windowName = "Mimicry"
image_coords = (209,200,542,380)
mirror_coords = (320, 125, 195, 385)            # x, y, width, height
images = []                                     # initialize empty list for images 

curator_img = cv2.imread("./assets/curator.png", cv2.IMREAD_UNCHANGED)  # Loading picture with alpha channel
curator_img = cv2.resize(curator_img, (width, height))  # First picture in same size as camera

vc.set(3, width)  # 3 stands for width
vc.set(4, height)  # 4 stands for height

#############################################################################################################
# FUNCTIONS                                                                                                 #
#############################################################################################################
# Center crop the image
def center_crop(img, dim):                          # dim: tuple(width, height)
    width, height = img.shape[1], img.shape[0]      # 1=width, 2 = height
    crop_width = min(dim[0], img.shape[1])          # ensure that crop size does not exceed image size
    crop_height = min(dim[1], img.shape[0])         # ensure that crop size does not exceed image size
    mid_x, mid_y = width // 2, height // 2          # calculate center
    return img[mid_y - crop_height // 2: mid_y + crop_height // 2, mid_x - crop_width // 2: mid_x + crop_width // 2]

# Read images from folder
def read_images(folder="images"):
    path = folder                                       # set path to folder 
    for filename in os.listdir(path):                   # iterate over all files in folder
        img = cv2.imread(f"{path}/{filename}")          # read image with OpenCV
        if img is not None:                             # check if image is read successfully
            crop_bg = center_crop(img, (width, height)) # crop images to right dimensions(same of camera)
            resized_img = cv2.resize(crop_bg, (width, height)) #resize image
            images.append(resized_img)                  # add image to list
    if not images:                                      # check if images were processed, else print error message
        print("No images found in image folder")

# Apply perspective transformation
def apply_perspective_transform(img):
    pts1 = np.float32([[0, 0], [0, img.shape[0]], [img.shape[1], img.shape[0]], [img.shape[1], 0]]) # Define source points (corners of the original image)
    pts2 = np.float32([[0, 0], [0, 384], [194, 368], [186, 39]])  # Define destinations points (corner points in the transformed image)
    M = cv2.getPerspectiveTransform(pts1, pts2)     # compute the perspective transform matrix
    return cv2.warpPerspective(img, M, (195, 385))  # Apply perspective transform to the image

# Function to create morph effect between two images
def morph_images(img1, img2, steps=60):
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

# Function to create morph effect between two images
def morph_images(img1, img2, steps=60):
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

# Funktion zur Vorverarbeitung von Bildern
def preprocess_image(img, target_width, target_height):
    # making sure image has the right measurements
    img_resized = cv2.resize(img, (target_width, target_height))
    # making sure image has 3 channels
    if img_resized.shape[2] != 3:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    return img_resized

#############################################################################################################
# MAIN                                                                                                      #
#############################################################################################################

def main():
    read_images()                                           # read images from folder
    if not images:                                          # else error message    
        print("No images available. Exiting program.")
        return
    
    # Handtracking related 
    hand_tracking = HandTracking()                          # Handtracking Object
    gesture_detected = False                                # Flag for gesture detection
    exit_gesture_detected = False                           # Flag for gesture detection
    change_detected = False                                 # Flag for gesture detection
    person_detector = PersonDetector()                      # Person detecting object
    
    # Person detecting related 
    person_timer = Timer()                                  # Timer Object
    person_detected_duration = 0                            # initial timer for person detection
    no_person_detected_duartion = 0                         # initial timer for no person detection

    show_curator = True                                     # Show curator flag
    
    # Filter related 
    filter = Filter()                                       # Filter object 
    img_idx = 0                                             # Index of the images in the folder, needed to apply filters on it

    while True:                                             # as long as camera is open:
        success, camera_img = vc.read()                     # open default camera, else error message
        if not success:
            print("Failed to read camera")
            break

        # Flip the camera image horizontally
        camera_img = cv2.flip(camera_img, 1)

        # Preprocess camera image
        camera_img = preprocess_image(camera_img, width, height)

        # Detect hand landmarks without drawing them
        camera_img = hand_tracking.find_hands(camera_img, draw=False)
        current_gesture = False

        # Check for hand gestures only when not showing the curator
        if not show_curator and hand_tracking.results.multi_hand_landmarks:
            # check each handlandmark for gesture
            current_gesture = any(hand_tracking.check_handgesture(landmark, camera_img.shape[1]) for landmark in hand_tracking.results.multi_hand_landmarks)

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
        sub_image = curator_img[y_start:y_end, x_start:x_end]   # ???
        sub_image = sub_image[:, :, :3]                         # Ausschneiden des Spiegels
                    
            
        # mask to simulate the mirror shape
        alpha_mask = np.where((camera_img_resized[:,:,0] == 0) & (camera_img_resized[:,:,1] == 0) & (camera_img_resized[:,:,2] == 0), 0, 255).astype(np.uint8)
        

        
        # render out black pixels
        camera_img_resized = segmentor.removeBG(camera_img_resized, sub_image, cutThreshold=0.8)
        black_pixels_mask = np.all(camera_img_resized == [0, 0, 0], axis=-1)
        #set black pixels to the same mirror pixels
        camera_img_resized[black_pixels_mask] = sub_image[black_pixels_mask]

        # Switch the background if a person is detected in the frame for 5 seconds 
        if show_curator:                                    # check if curator is shown currently 
            if person_detector.detect_person(camera_img):   # detect if person is in the camera image 
                if person_detected_duration == 0:           # check if detection timer has not started
                    person_timer.start()                    # start timer
                person_detected_duration = person_timer.elapsed_time() # update the duration for which the person has been detected
                if person_detected_duration >= 5: # check if person has been detected for at least five seconds
                    # Start morph effect to switch scene
                    morph_images(curator_img, images[img_idx]) # morph curator with current background
                    show_curator = False # set curator flag to false, means we dont see the curator anymore
                    person_detected_duration = 0 # reset detection duration
                    person_timer.reset() # reset the timer
            else: # if not person is detected
                person_detected_duration = 0 # reset detection duration
                person_timer.reset() # reset timer
        else: # if not showing the curator
            if person_detector.detect_person(camera_img):# detect if a person is in the camera image
                person_detected_duration = 0 # reset detection duration
                person_timer.reset() # reset timer
                no_person_detected_duartion = 0 # set no detection duration to 0
            else: # if no person is detected in the camera image
                if no_person_detected_duartion == 0: # check if no detection timer has started
                    person_timer.start() # start timer
                no_person_detected_duartion = person_timer.elapsed_time() # set do detection duration to started timer
                if no_person_detected_duartion >= 5: # check if no person was detected during the last 5 seconds
                    morph_images(images[img_idx], curator_img) # morph current background image to the curator scene
                    show_curator = True # set curator flag to true to show the curator
                    no_person_detected_duartion = 0 # reset no detection duration to 0 
                    person_timer.reset() # reset timer

        # Closing with gesture
        exit_gesture = False # initialize exit gesture flag
        if not show_curator and hand_tracking.results. multi_hand_landmarks: # check for exit gesture only when not showing the curator scene and hands are detected
            exit_gesture = any(hand_tracking.close_window(landmark) for landmark in hand_tracking.results.multi_hand_landmarks) # detect if the exit gesture is made
        if exit_gesture and not exit_gesture_detected:
            # Start reverse morph effect to return to the curator scene
            morph_images(images[img_idx], curator_img) # morph current background image to the curator scene
            show_curator = True # set the curator flag TRUE, means we see the curator again
            exit_gesture_detected = True # set the exit gesture flag TRUE, means the exit gesture has been handled
            person_detected_duration = 0 # rest detection duration
            person_timer.reset() # reset timer

        if not exit_gesture: # if not exit gesture has been detected
            exit_gesture_detected = False # reset the flag to FALSE

        if show_curator:
            # Convert to BGRA format
            camera_img_resized = cv2.cvtColor(camera_img_resized, cv2.COLOR_BGR2BGRA)

            # Ensure alpha_mask is the right shape and type
            alpha_mask = np.expand_dims(alpha_mask, axis=-1).astype(np.uint8)  # Add an extra dimension and cast to np.uint8

            # Resize alpha_mask to match the dimensions of camera_img_resized
            alpha_mask_resized = cv2.resize(alpha_mask, (camera_img_resized.shape[1], camera_img_resized.shape[0]))

            # Merge camera_img_resized and alpha_mask_resized into a single BGRA image
            camera_img_resized = cv2.merge((camera_img_resized[:,:,0], camera_img_resized[:,:,1], camera_img_resized[:,:,2], alpha_mask_resized)) # combine rgb channels with alpha mask 

            # Create distance transformation
            dist_transform = cv2.distanceTransform(alpha_mask_resized, cv2.DIST_L2, 5)  # calculate the distance to the nearest zero pixel  
            gradient = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)

            # Invert gradient, invert normalized distance transform for gradient effect
            gradient = 1 - gradient     

            # scale gradient to create black to transparent gradient 
            gradient = (gradient * 25).astype(np.uint8)                     # scale for visibility
            gradient = cv2.merge((gradient, gradient, gradient, gradient))  # merge to create alpha channel gradient 

            # Apply gradient to camera_img_resized
            alpha_gradient = gradient[:, :, 3]
            alpha_gradient = cv2.cvtColor(alpha_gradient, cv2.COLOR_GRAY2BGRA) # convert to 4-channel grayscale 
            alpha_gradient[:, :, 3] = gradient[:, :, 3] # set the alpha channel 

            # Combine camera_img_resized and alpha_gradient using alpha compositing
            camera_img_resized = cv2.addWeighted(camera_img_resized, 1, alpha_gradient, -1, 0)
            
            curator_copy = curator_img.copy()
            roi = curator_copy[mirror_coords[1]: mirror_coords[1] + mirror_coords[3], mirror_coords[0]: mirror_coords[0] + mirror_coords[2]]

            # Ensure the mask dimensions match the ROI dimensions
            mask = cv2.resize(cv2.bitwise_not(camera_img_resized[:, :, 3]), (roi.shape[1], roi.shape[0])) #resize mask to match ROI 
            roi_bg = cv2.bitwise_and(roi, roi, mask) # create background region by applying mask 
            roi_fg_resized = cv2.resize(camera_img_resized, (roi.shape[1], roi.shape[0])) # resize transformed img to match roi 
            roi_fg = cv2.bitwise_and(roi_fg_resized, roi_fg_resized, mask=roi_fg_resized[:, :, 3]) # create foreground region

            # Add the masked foreground and background images
            dst = cv2.add(roi_bg, roi_fg) # blend background and foreground regions 
            curator_copy[mirror_coords[1]: mirror_coords[1] + mirror_coords[3], mirror_coords[0]: mirror_coords[0] + mirror_coords[2]] = dst # place blended imgae into the background
            
            display_image = curator_copy # image to be displayed 
 

        ###############################################################################################
        # if the curator scene is not shown, means we see the segmented human in front of the art
        # depending on the art a certain filter is applied to the segmented human 
        else:
            
            # img_indx is the index number for each art ( 0 = starry night, 1 = the scream, 2 = un dimanche)
            # imgBG is the img of the art
            imgBg = images[img_idx]            
                    
            # Segmenting the human from the bg ( = art)
            # Condition determines what img is shown (background / foreground), is needed for def combine (imgBg und segmented img)
            segmented_img, condition = segmentor.get_segments(camera_img, imgBg, cutThreshold=0.45)

            # APPLY FILTERS ####################################################################################################################
            # Filter for first art (= girl with a pearl) = oilpainting 
            if img_idx == 0:
                # apply oilpainting filter
                segmented_img = Filter.oil_painting(segmented_img)
            
            
            # Filter for second art (= starry night) = cartoon filter (lines are transparent) + duotone filter (colors darkblue and yellow, taken from the image)
            if img_idx == 1:
                # Apply the cartoon Filter first on the segmented image, black lines
                segmented_img = Filter.apply_cartoon(segmented_img)
                # in the condition mask we check where are black pixels (= cartoon lines), result is either False or True (boolean array)
                condition = np.where((segmented_img[:,:,0] == 0) & (segmented_img[:,:,1] == 0) & (segmented_img[:,:,2] == 0), False, True)
                # We need to transform the condition array into the shape (720, 1280, 3) in order to be combined with the other imgs
                condition = np.stack((condition, condition,condition), axis= -1)
                # Apply the duotone colors here, else the background gets lost 
                segmented_img = Filter.apply_duotone_filter(segmented_img, Filter.farbe_gelb, Filter.farbe_dunkelblau)
            
            # Filter for third art (= the scream) = cartoon filter (lines are transparent)         
            elif img_idx == 2:
                segmented_img = Filter.apply_cartoon(segmented_img)
                # in the condition mask we check where are black pixels (= cartoon lines), result is either False or True (boolean array)
                condition = np.where((segmented_img[:,:,0] == 0) & (segmented_img[:,:,1] == 0) & (segmented_img[:,:,2] == 0), False, True)
                # We need to transform the condition array into the shape (720, 1280, 3) in order to be combined with the other imgs
                condition = np.stack((condition, condition,condition), axis= -1)
               
            # Filter for fourth art (= un dimanche) = "Pointilismus" Filter  
            # Pencilsketch to make it look like its drawn, just like the art
            # Duotone to adjust the colors to the art >>>>>>>>> Improvement: more colors, taken from the picture  
            elif img_idx == 3:
                #segmented_img = Filter.apply_pencilsketch_and_duotone(segmented_img, Filter.farbe_gelb, Filter.farbe_dunkelgruen)
                #segmented_img = Filter.apply_multitone_filter(segmented_img, Filter.palette)
                segmented_img = Filter.apply_pencilsketch_and_multitone(segmented_img, Filter.palette)

            # Combine the segmented img ( = human plus filter) with the background img ( = art) 
            display_image = segmentor.combine(condition, segmented_img,  imgBg, cutThreshold=0.45)

        ###############################################################################################
        # Fullscreen
        cv2.namedWindow(windowName, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(windowName, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow(windowName, display_image)

        

        # Keyboard, handles the input
        key = cv2.waitKey(1)
        # Close window with esc key (= 27 in ASCII code)
        if key == 27:
            break # exit loop, close the window
        # Switching between images with key press 's'
        if key == ord('s'):
            show_curator = not show_curator # toggle show curator flag
        # Switching between images with key press 'a' for previous image and 'd' for next image
        if key == ord('a') and not show_curator:
            img_idx = (img_idx - 1) % len(images) # decrement image index to show the previous image, wrapping around if necessary 
        elif key == ord('d') and not show_curator:
            img_idx = (img_idx + 1) % len(images)# decrement image index to show the next image, wrapping around if necessary 
        # Close window with clicking on 'x'
        if cv2.getWindowProperty(windowName, cv2.WND_PROP_VISIBLE) < 1: # check if window is no longer visible (for example the x button is pressed)
            break # exit loop and close the window

    # clean up
    vc.release()            # release camera resource
    cv2.destroyAllWindows() # close all OpenCV windows

# MAIN CALL
if __name__ == "__main__":
    main() # call main function = runs the program