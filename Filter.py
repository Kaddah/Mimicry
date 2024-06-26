import math
import random 
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import numpy as np
import scipy



class Filter: 
    
    ##################################################################################################
    # VARIABLEN                                                                                      #
    # filter_mode für die spätere Auswahl per Tastatur                                               #
    # Farben für Filter händisch festgelegt                                                          #
    ##################################################################################################
    
    filter_mode = None
            
    farbe_gelb = (47, 235, 255)
    farbe_orange = (85, 165, 255)
    farbe_rot = (0, 0, 255)
    farbe_rosa = (202, 148, 255)
    farbe_hellgruen = (155, 255, 158)
    farbe_dunkelgruen = (27, 80, 30)
    farbe_hellblau = (255, 200, 150)
    farbe_dunkelblau = (132, 42, 10)
    farbe_braun = (24, 74, 108)
    
    # Farben für un Dimanche Bild
    farbe_ud_gelb = (225, 209, 98)
    farbe_ud_grün = (102, 143, 103)
    #farbe_ud_blau = (161, 203, 243)
    farbe_ud_dunkelblau = (49, 56, 74)
    farbe_ud_rot = (213, 93, 43)
    
    # Farbpalette für un Dimanche
    palette = [farbe_ud_gelb, farbe_ud_grün, farbe_ud_dunkelblau, farbe_ud_rot]
            
    ##################################################################################################
    # FILTER                                                                                         #
    ##################################################################################################

    ######## EINFACHE FILTER #########################################################################
    
    # Graustufenfilter
    def apply_grayscale(frame):
        grauBild = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        Bild = grauBild[:,:,np.newaxis]           # Mit newaxis machen wir das Bild wieder 3D statt 2D, dann passt es zsm 
        return Bild
   
    # Negativfilter
    def apply_negative(frame):
        return cv2.bitwise_not(frame)

    # Clear, zeigt einfach unverändertes Kamerabild an
    def clear(frame):
        return frame  # Keine Änderung, das Bild bleibt unverändert

    # Funktion zur Anwendung eines Duoton-Filters
    def apply_duotone_filter(frame, color1, color2):
        # Konvertiere das Bild in Graustufen
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Erstelle ein leeres Bild mit den gleichen Abmessungen wie das ursprüngliche Bild
        duotone_frame = np.zeros_like(frame)
        
        # Bestimme die Farben für den Duoton-Effekt
        color1_bgr = np.array(color1, dtype=np.uint8)
        color2_bgr = np.array(color2, dtype=np.uint8)
        
        # Setze den Wert jedes Pixels im duotonen Bild entsprechend des Grauwerts des Originalbildes und der gewünschten Farben
        duotone_frame[:, :, 0] = (gray / 255.0) * color1_bgr[0] + ((1 - gray / 255.0) * color2_bgr[0])  # setzt den Blaukanal entsprechend der Grauwerte und den Colorwerten im Bild
        duotone_frame[:, :, 1] = (gray / 255.0) * color1_bgr[1] + ((1 - gray / 255.0) * color2_bgr[1])  # Setzt den Grünkanal wie oben
        duotone_frame[:, :, 2] = (gray / 255.0) * color1_bgr[2] + ((1 - gray / 255.0) * color2_bgr[2])  # setzt den Rotkanal wie oben
        
        return duotone_frame
    
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

    # Funktion, die Bild wie gemalt aussehen lässt, Cartoonstil
    def apply_cartoon(frame):
        
        cartoon_img = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
        gray = cv2.cvtColor(cartoon_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2)
        colored_edges = cv2.bitwise_and(cartoon_img, cartoon_img, mask=edges)
        
        
        
        return colored_edges

    # Funktion, die Bild wie Pencilsketch aussehen lässt (B/w)
    def sketch(frame):

        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_gray_blur = cv2.GaussianBlur(img_gray, (5,5), 0)
        canny_edges = cv2.Canny(img_gray_blur, 10, 70)
        ret, mask = cv2.threshold(canny_edges, 70, 255, cv2.THRESH_BINARY_INV)
        return mask

    # Funktion, die Bild wie Pencilsketch aussehen lässt, in Farbe
    def pencilsketch(frame):

        # Umkehren des Graustufenbildes für den Sketch-Effekt
        inverted_img = cv2.bitwise_not(frame)
        
        # Anwendung der Glättung (Gaussian Blur)
        img_smoothing = cv2.GaussianBlur(inverted_img, (21, 21), sigmaX=0, sigmaY=0)
        
        # Invertieren des geglätteten Bildes
        inverted_img_smoothing = cv2.bitwise_not(img_smoothing)
        
        # Erstellung des Pencil-Sketch-Effekts durch Berechnung des Differenzbildes
        pencil_sketch = cv2.divide(frame, inverted_img_smoothing, scale=256.0)
        
        return pencil_sketch

    # Funktion, die Bild wie Ölgemälde aussehen lässt
    def oil_painting(frame):
        oilPainting = cv2.xphoto.oilPainting(frame, 7, 1)
        return oilPainting

    # Pointilismus Filter
    def pointilismus(frame, num_points=500, max_radius=10):
        # Create an empty canvas
        pointillism_frame = np.zeros_like(frame)
        
        # Get dimensions of the frame
        h, w, _ = frame.shape
        
        # Create a grid of points
        points = np.mgrid[0:h:max_radius, 0:w:max_radius].reshape(2, -1).T
        
        # Shuffle the grid points
        np.random.shuffle(points)
        
        # Limit the number of points
        points = points[:num_points]
        
        for y, x in points:
            # Randomly pick a radius
            radius = random.randint(1, max_radius)
            
            # Get the color of the point from the original frame
            color = frame[y, x]
            
            # Draw a filled circle at the chosen point with the chosen color
            cv2.circle(pointillism_frame, (x, y), radius, tuple(int(c) for c in color), -1)
        
        return pointillism_frame

    ######## KOMBINIERTE FILTER ######################################################################

    # Pencilsketch + Duotone
    def apply_pencilsketch_and_duotone(frame, color1, color2):
        # Pencil Sketch anwenden
        pencil_sketch = Filter.pencilsketch(frame)
        
        # Duotone-Filter anwenden auf das Pencil Sketch-Bild
        duotone_frame = Filter.apply_duotone_filter(pencil_sketch, color1, color2)
        
        return duotone_frame

    # Pencilsketch + Multitone
    def apply_pencilsketch_and_multitone(frame, colors):
        # Pencilsketch anwenden
        pencil_sketch = Filter.pencilsketch(frame)
        # Multitone anwenden
        multitone_frame = Filter.apply_multitone_filter(pencil_sketch, colors)
        return multitone_frame
    
    # Duotone + Cartoon
    def apply_duotone_and_cartoon(frame, color1, color2, alpha, beta):
        # Duotone-Filter anwenden
        duotone_frame = Filter.apply_duotone_filter(frame, color1, color2)
        
        # Stilisierungsfilter anwenden
        stylized_frame = Filter.apply_cartoon(duotone_frame)
        
        return stylized_frame




    # Filter für Picasso:
    # Lass den Menschen wie Geometrische Formen aussehen
    # Male ein paar gewählte Farben drüber 
    # Vlt Cartoon Lines drumherum (schwarze Linien) 
    
    



    #####################################################################################################################
    # Hauptschleife des Filter Programms:                                                                               #
    # Kann man beibehalten, falls man das Projekt erweitern will vielleicht                                             #
    '''
        Main-Schleife
        - Zeigt Kamerabild an: Solange Kamera an ist, wird der Videostream eingelesen >> vc.read()
        - Sonst Fehlermeldung
        - Nutzereingaben über Tasten: 
            b = Hintergrund anzeigen / ausblenden
            n = nächster Hintergrund
            s = screenshot
            g = Graustufenfilter
            m = Negativfilter
            q = Quit = Programm wird beendet
            o = Oilpainting
            d = Duotone
            r = Duotone und Sketch
            t = sketch
            e = pencilsketch und duotone
            p = pencilsketch
            i = pointilismus 
        - Um Hintergrund mit gleicher Taste an / aus zuschalten wird ein Backgroundflag benutzt
        Ist das Flag gesetzt, ist der Hintergrund an, d.h. beim nächsten Drücken der Taste b wird der HG ausgeschaltet
        und das Flag auf aus gesetzt (False). Wenn Nutzer taste n drückt, wird HG automatisch angezeigt und Flag wird 
        auf True gesetzt.
    '''                                                                                                                 #
    #####################################################################################################################

    def apply_filter():
        # Tastaturbelegung:
        key = cv2.waitKey(1)
        
            
        if key == ord('g'):  # g = Schwarz-Weiß
            filter_mode = 'grayscale' if filter_mode != 'grayscale' else None

        elif key == ord('m'):  # n = Negativ
            filter_mode = 'negative' if filter_mode != 'negative' else None

        elif key == ord('c'):  # c = clear
            filter_mode = 'clear' if filter_mode != 'clear' else None
        
        elif key == ord('g'): # g für Duoton
            filter_mode = 'duotone' if filter_mode != 'duotone' else None
            
        elif key == ord('t'): # t für carToon
            filter_mode = 'cartoon' if filter_mode != 'cartoon' else None
        
        elif key == ord('p'): # p für PencilSketch
            filter_mode = 'pencilsketch' if filter_mode != 'pencilsketch' else None

        elif key == ord('r'):  # r für Duotone und caRtoon
            filter_mode = 'duotone_and_cartoon' if filter_mode != 'duotone_and_cartoon' else None
        
        elif key == ord('e'):   # e für DuotonE und pEncilskEtch
            filter_mode = 'pencilsketch_and_duotone' if filter_mode != 'pencilsketch_and_duotone' else None
            
        elif key == ord('o'):   # o für Oilpainting
            filter_mode = 'oilPainting' if filter_mode != 'oilPainting' else None
        
        elif key == ord('i'):    # i für poIntIlIsmus 
            filter_mode = 'pointilismus' if filter_mode != 'pointilismus' else None


        # Filter anwenden je nach Auswahl 
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
