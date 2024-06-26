import math
import random 
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import numpy as np
import scipy

# Class that implements the Filters that are applied to the arts ( = backgrounds)

class Filter: 
    
    ##################################################################################################
    # VARIABLES                                                                                      #
    ##################################################################################################
    
    filter_mode = None                          # needed if the filter is seleceted via keyboard, not used in the final project 
    
    
    # defined colors for certain filters         
    farbe_gelb = (47, 235, 255)     # (R, G, B)    
    farbe_orange = (85, 165, 255)
    farbe_rot = (0, 0, 255)
    farbe_rosa = (202, 148, 255)
    farbe_hellgruen = (155, 255, 158)
    farbe_dunkelgruen = (27, 80, 30)
    farbe_hellblau = (255, 200, 150)
    farbe_dunkelblau = (132, 42, 10)
    farbe_braun = (24, 74, 108)
    
    # certain colors for un dimanche art 
    farbe_ud_gelb = (225, 209, 98)
    farbe_ud_grün = (102, 143, 103)
    #farbe_ud_blau = (161, 203, 243)
    farbe_ud_dunkelblau = (49, 56, 74)
    farbe_ud_rot = (213, 93, 43)
    
    # color palette for un dimanche filter, put the defined colors in a list
    palette = [farbe_ud_gelb, farbe_ud_grün, farbe_ud_dunkelblau, farbe_ud_rot]
            
    ##################################################################################################
    # FILTER                                                                                         #
    ##################################################################################################

    ######## SIMPLE FILTER #########################################################################
    
    # Grayscale 
    def apply_grayscale(frame):
        grauBild = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Bild = grauBild[:,:,np.newaxis]           # newaxis turns the img from 2D into 3D 
        return Bild
   
    # Negative
    def apply_negative(frame):
        return cv2.bitwise_not(frame)   # inverts the colors

    # Clear, shows unchanged camera image 
    def clear(frame):
        return frame  

    # Duotone 
    def apply_duotone_filter(frame, color1, color2):
        # convert into grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # create an empty frame, same size as input img
        duotone_frame = np.zeros_like(frame)
        
        # define the two colors 
        color1_bgr = np.array(color1, dtype=np.uint8)
        color2_bgr = np.array(color2, dtype=np.uint8)
        
        # set value of each pixel in the duotone img according to the grayscale value of the orignal image
        duotone_frame[:, :, 0] = (gray / 255.0) * color1_bgr[0] + ((1 - gray / 255.0) * color2_bgr[0])  # sets blue component
        duotone_frame[:, :, 1] = (gray / 255.0) * color1_bgr[1] + ((1 - gray / 255.0) * color2_bgr[1])  # sets green component 
        duotone_frame[:, :, 2] = (gray / 255.0) * color1_bgr[2] + ((1 - gray / 255.0) * color2_bgr[2])  # sets red component 
        
        return duotone_frame
    
    # Multitone, similar to duotone, but more than two colors
    def apply_multitone_filter(frame, colors):
        # Convert the image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Create an empty image with the same dimensions as the original image
        multitone_frame = np.zeros_like(frame, dtype=np.float32)
        
        # Ensure colors are in BGR format and numpy array
        colors_bgr = [np.array(color, dtype=np.float32) for color in colors]
        
        # Normalize grayscale image to range [0, 1]
        gray_normalized = gray / 255.0
        
        # Get the number of colors
        num_colors = len(colors_bgr)
        
        # Define the segments based on the number of colors
        segments = np.linspace(0, 1, num_colors)
        
        # Interpolate colors based on grayscale values
        for i in range(num_colors - 1):
            # Create a mask for the current segment
            mask = (gray_normalized >= segments[i]) & (gray_normalized < segments[i + 1])
            
            # Interpolation factor
            factor = (gray_normalized[mask] - segments[i]) / (segments[i + 1] - segments[i])
            
            # Apply the interpolation for each color channel
            for j in range(3):
                multitone_frame[mask, j] = (
                    (1 - factor) * colors_bgr[i][j] + factor * colors_bgr[i + 1][j]
                )
        
        # Handle the edge case where the grayscale value is exactly 1.0
        mask = gray_normalized == 1.0
        multitone_frame[mask] = colors_bgr[-1]

        # Convert back to uint8
        multitone_frame = np.clip(multitone_frame, 0, 255).astype(np.uint8)
        
        return multitone_frame

<<<<<<< HEAD
    # Cartoon Style 
=======
    # Funktion, die Bild wie gemalt aussehen lässt, Cartoonstil
>>>>>>> 93c7f82580f9b5c038cb0987947d3139beb691e3
    def apply_cartoon(frame):
        # Apply a bilateral filter to the image to smooth the colors while preserving edges
        # This helps in achieving the cartoon effect by reducing the color palette of the image
        cartoon_img = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        
        # Convert the smoothed image to grayscale
        # Grayscale conversion is necessary for edge detection
        gray = cv2.cvtColor(cartoon_img, cv2.COLOR_BGR2GRAY)
        
        # Apply a median blur to the grayscale image to reduce noise
        # This helps in creating smoother edges when we perform edge detection
        blurred = cv2.medianBlur(gray, 7)
        
        # Use adaptive thresholding to detect edges in the blurred grayscale image
        # Adaptive thresholding helps in finding edges in different lighting conditions
        edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
        
        # Combine the edges with the color image using a bitwise AND operation
        # This overlays the detected edges on the smoothed, color-reduced image to create the cartoon effect
        colored_edges = cv2.bitwise_and(cartoon_img, cartoon_img, mask=edges)
        
        return colored_edges

    # Pencilsketch (B/w)
    def sketch(frame):
        # Convert the image to grayscale
        # Purpose: Simplifies the image by reducing it to a single channel, making it easier to detect edges
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply a Gaussian blur to the grayscale image
        # Purpose: Smooths the image, reducing noise and detail which can help in better edge detection
        img_gray_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
        
        # Perform Canny edge detection
        # Purpose: Detects edges in the blurred grayscale image
        canny_edges = cv2.Canny(img_gray_blur, 10, 70)
        
        # Apply a binary inverse threshold to the edge-detected image
        # Purpose: Converts the edges to white and the background to black, creating a sketch-like effect
        ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
        
        return mask

    def pencilsketch(frame):
        # Invert the colors of the image
        # Purpose: Create a negative image for further processing
        inverted_img = cv2.bitwise_not(frame)
        
        # Apply Gaussian blur to the inverted image
        # Purpose: Smooth the inverted image to create a more natural sketch effect
        img_smoothing = cv2.GaussianBlur(inverted_img, (21, 21), sigmaX=0, sigmaY=0)
        
        # Invert the blurred image
        # Purpose: Re-invert the smoothed image to prepare for blending with the original
        inverted_img_smoothing = cv2.bitwise_not(img_smoothing)
        
        # Blend the original image with the inverted, smoothed image to create the pencil sketch effect
        # Purpose: The division highlights the edges and creates a sketch-like appearance
        pencil_sketch = cv2.divide(frame, inverted_img_smoothing, scale=256.0)
        
        return pencil_sketch

    def oil_painting(frame):
        # Apply the oil painting effect using the xphoto module
        # Parameters:
        # - size: Neighborhood size used for filtering (7)
        # - dynRatio: Dynamic range ratio for quantization (1)
        # Purpose: Converts the image to look like an oil painting by filtering and quantizing colors
        oilPainting = cv2.xphoto.oilPainting(frame, 7, 1)
        
        return oilPainting

    def pointilismus(frame, num_points=500, max_radius=10):
        # Create an empty canvas with the same dimensions as the input frame
        # Purpose: To have a blank image where the pointillism effect will be drawn
        pointillism_frame = np.zeros_like(frame)
        
        # Get the dimensions of the input frame
        # Purpose: To know the size of the image for placing points correctly
        h, w, _ = frame.shape
        
        # Create a grid of points with a spacing of max_radius
        # Purpose: To generate a set of candidate points for drawing circles
        points = np.mgrid[0:h:max_radius, 0:w:max_radius].reshape(2, -1).T
        
        # Shuffle the grid points
        # Purpose: To randomly distribute the points over the image
        np.random.shuffle(points)
        
        # Limit the number of points to num_points
        # Purpose: To control the density of points in the pointillism effect
        points = points[:num_points]
        
        for y, x in points:
            # Randomly pick a radius for the circle
            # Purpose: To vary the size of the circles, enhancing the artistic effect
            radius = random.randint(1, max_radius)
            
            # Get the color of the point from the original frame
            # Purpose: To use the original image's color for the pointillism effect
            color = frame[y, x]
            
            # Draw a filled circle at the chosen point with the chosen color
            # Purpose: To create the pointillism effect by drawing colored circles
            cv2.circle(pointillism_frame, (x, y), radius, tuple(int(c) for c in color), -1)
        
        return pointillism_frame

    ######## COMBINED FILTER ######################################################################

    # Pencilsketch + Duotone
    # Apply pencilsketch to the input, then duotone to create a combined effect 
    def apply_pencilsketch_and_duotone(frame, color1, color2):
        # Pencil Sketch 
        pencil_sketch = Filter.pencilsketch(frame)
        
        # Duotone-Filter 
        duotone_frame = Filter.apply_duotone_filter(pencil_sketch, color1, color2)
        
        return duotone_frame

    # Pencilsketch + Multitone
    # Apply pencilsketch to the input, then multitone to create a combined effect
    def apply_pencilsketch_and_multitone(frame, colors):
        # Pencilsketch 
        pencil_sketch = Filter.pencilsketch(frame)
        # Multitone 
        multitone_frame = Filter.apply_multitone_filter(pencil_sketch, colors)
        return multitone_frame
    
    # Duotone + Cartoon
    # Apply duotone to the input, then cartoon to create a combined effect
    def apply_duotone_and_cartoon(frame, color1, color2, alpha, beta):
        # Duotone-Filter 
        duotone_frame = Filter.apply_duotone_filter(frame, color1, color2)
        
        # Stilisierungsfilter 
        stylized_frame = Filter.apply_cartoon(duotone_frame)
        
        return stylized_frame

<<<<<<< HEAD
=======



    # Filter für Picasso:
    # Lass den Menschen wie Geometrische Formen aussehen
    # Male ein paar gewählte Farben drüber 
    # Vlt Cartoon Lines drumherum (schwarze Linien) 
    
    



>>>>>>> 93c7f82580f9b5c038cb0987947d3139beb691e3
    #####################################################################################################################
    # MAIN LOOP OF ORIGINAL FILTER PROGRAMM                                                                             #
    # is not used in the final project, but can be used for further improvements                                        #
    '''
        Main-Loop
        - show camera img: as long as camera is open, read videostream >> vc.read()
        - else error message
        - Keyboard Input: 
            b = toggle show background 
            n = next background 
            s = screenshot
            g = grayscale
            m = negative
            q = Quit (exit the program)
            o = Oilpainting
            d = Duotone
            r = Duotone and Sketch
            t = sketch
            e = pencilsketch and duotone
            p = pencilsketch
            i = pointilismus 
        - use background flag to be able to switch background on and off with only one key
    '''                                                                                                                 #
    #####################################################################################################################

    def apply_filter():
        key = cv2.waitKey(1)
        
            
        if key == ord('g'):  # g = b/w
            filter_mode = 'grayscale' if filter_mode != 'grayscale' else None

        elif key == ord('m'):  # n = Negativ
            filter_mode = 'negative' if filter_mode != 'negative' else None

        elif key == ord('c'):  # c = clear
            filter_mode = 'clear' if filter_mode != 'clear' else None
        
        elif key == ord('g'): # g  Duoton
            filter_mode = 'duotone' if filter_mode != 'duotone' else None
            
        elif key == ord('t'): # t  carToon
            filter_mode = 'cartoon' if filter_mode != 'cartoon' else None
        
        elif key == ord('p'): # p  PencilSketch
            filter_mode = 'pencilsketch' if filter_mode != 'pencilsketch' else None

        elif key == ord('r'):  # r  Duotone and caRtoon
            filter_mode = 'duotone_and_cartoon' if filter_mode != 'duotone_and_cartoon' else None
        
        elif key == ord('e'):   # e  DuotonE and pEncilskEtch
            filter_mode = 'pencilsketch_and_duotone' if filter_mode != 'pencilsketch_and_duotone' else None
            
        elif key == ord('o'):   # o  Oilpainting
            filter_mode = 'oilPainting' if filter_mode != 'oilPainting' else None
        
        elif key == ord('i'):    # i  poIntIlIsmus 
            filter_mode = 'pointilismus' if filter_mode != 'pointilismus' else None


        # apply filter according to keyboard input
        if filter_mode == 'grayscale':
            frame = Filter.apply_grayscale(frame)
            
        elif filter_mode == 'negative':
            frame = Filter.apply_negative(frame)
            
        elif filter_mode == 'clear':
            frame = Filter.clear(frame)
            
        elif filter_mode == 'cartoon':
            frame = Filter.apply_cartoon(frame)
                
        elif filter_mode == 'pencilsketch':
            frame = Filter.pencilsketch(frame)
            
        elif filter_mode == 'oilPainting':
            frame = Filter.oil_painting(frame)
        
        elif filter_mode == 'pointilismus':
            frame = Filter.pointilismus(frame, 10)
         
          
        # Following code used the background index to apply the combined filters,   
        # this is not used in the final project 
        # if you want to use this in the future you need to add the background index logic in this class  
        ''' 
        elif filter_mode == 'duotone':
            if Filter.show_background_flag and current_background_index == 0:                      # Überprüft welcher HG gewählt ist
                frame = Filter.apply_duotone_filter(frame, Filter.farbe_gelb, Filter.farbe_dunkelblau)           # Farben für HG 0
            elif Filter.show_background_flag and current_background_index == 1:
                    frame = Filter.apply_duotone_filter(frame, Filter.farbe_orange, Filter.farbe_dunkelblau)     # Farben für HG 1
            elif Filter.show_background_flag and current_background_index == 2:
                    frame = Filter.apply_duotone_filter(frame, Filter.farbe_rot, Filter.farbe_dunkelgruen)       # Farben für HG 2
            else:
                frame = Filter.apply_duotone_filter(frame, Filter.farbe_dunkelgruen, Filter.farbe_hellblau)      # Farben unabhängig vom Bild, zufällig von mir ausgewählt
                
        elif filter_mode == 'duotone_and_cartoon':
            if Filter.show_background_flag and current_background_index == 0:
                frame = Filter.apply_duotone_and_cartoon(frame, Filter.farbe_gelb, Filter.farbe_dunkelblau, 50, 20)
            elif Filter.show_background_flag and current_background_index == 1:
                frame = Filter.apply_duotone_and_cartoon(frame, Filter.farbe_orange, Filter.farbe_dunkelblau, 50, 20)
            elif Filter.show_background_flag and current_background_index == 2:
                frame = Filter.apply_duotone_and_cartoon(frame, Filter.farbe_rot, Filter.farbe_dunkelgruen, 50, 20)
            else:
                frame = Filter.apply_duotone_and_cartoon(frame, Filter.farbe_dunkelgruen, Filter.farbe_hellblau, 50, 20)
        
        elif filter_mode == 'pencilsketch_and_duotone':
            if Filter.show_background_flag and current_background_index == 0:
                frame = Filter.apply_pencilsketch_and_duotone(frame, Filter.farbe_gelb, Filter.farbe_dunkelblau)
            elif Filter.show_background_flag and current_background_index == 1:
                frame = Filter.apply_pencilsketch_and_duotone(frame, Filter.farbe_orange, Filter.farbe_dunkelblau)
            elif Filter.show_background_flag and current_background_index == 2:
                frame = Filter.apply_pencilsketch_and_duotone(frame, Filter.farbe_rot, Filter.farbe_dunkelgruen)
            else:
                frame = Filter.apply_pencilsketch_and_duotone(frame, Filter.farbe_dunkelgruen, Filter.farbe_hellblau)
        
        '''
