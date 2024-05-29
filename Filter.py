import math
import random 
import cv2
import cvzone
from cvzone.SelfiSegmentationModule import SelfiSegmentation
import numpy as np
import scipy
from sklearn.cluster import KMeans



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
            
    ##################################################################################################
    # FILTER                                                                                         #
    ##################################################################################################
    
    ######## HILFSFUNKTIONEN #########################################################################
    
    # Hilfsfunktion für Pointilismus
    def compute_color_probabilities(pixels, palette):
        distances = scipy.spatial.distance.cdist(pixels, palette)
        maxima = np.amax(distances, axis=1)
        distances = maxima[:, None] - distances
        summ = np.sum(distances, 1)
        distances /= summ[:, None]
        return distances
    # Hilfsfunktion für Pointilismus
    def get_color_from_prob(probabilities, palette):
        probs = np.argsort(probabilities)
        i = probs[-1]
        return palette[i]
    # Hilfsfunktion für Pointilismus
    def randomized_grid(h, w, scale):
        assert (scale > 0)
        r = scale//2
        grid = []
        for i in range(0, h, scale):
            for j in range(0, w, scale):
                y = random.randint(-r, r) + i
                x = random.randint(-r, r) + j
                grid.append((y % h, x % w))
        random.shuffle(grid)
        return grid
    # Hilfsfunktion für Pointilismus
    def get_color_palette(img, n=20):
        clt = KMeans(n_clusters=n)
        clt.fit(img.reshape(-1, 3))
        return clt.cluster_centers_
    # Hilfsfunktion für Pointilismus
    def complement(colors):
        return 255 - colors

    # Hilfsfunktion um Filter z.T. transparent zu machen
    # Schwarze Linien transparent machen für den Cartoon Filter 
    def make_black_lines_transparent(img):
        # Überprüfe ob das Bild einen alpha Kanal hat, füge einen alpha kanal hinzu
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        
        # Definiere die schwarzen Linien um sie mit einer Maske entfernen zu können
        lower_black = np.array([0, 0, 0], dtype=np.uint8)
        upper_black = np.array([50, 50, 50], dtype=np.uint8)

        # Maske für die schwarzen Pixel
        black_mask = cv2.inRange(img[:, :, :3], lower_black, upper_black)

        # Setze Alphakanal der Maske auf 0 >> Transparent
        img[:, :, 3][black_mask == 255] = 0

        return img
    
    
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
    def pointilismus(image, primary_colors):
        radius_width = int(math.ceil(max(image.shape) / 1000))
        palette = Filter.get_color_palette(image, primary_colors)
        complements = Filter.complement(palette)
        palette = np.vstack((palette, complements))
        canvas = np.zeros_like(image)  # Erstelle eine leere Leinwand mit den gleichen Abmessungen wie das Bild
        grid = Filter.randomized_grid(image.shape[0], image.shape[1], scale=3)
        
        pixel_colors = np.array([image[x[0], x[1]] for x in grid])
        
        color_probabilities = Filter.compute_color_probabilities(pixel_colors, palette)
        for i, (y, x) in enumerate(grid):
            color = Filter.get_color_from_prob(color_probabilities[i], palette)
            # Konvertiere Farbe in das richtige Format (BGR anstelle von RGB)
            color_bgr = (int(color[2]), int(color[1]), int(color[0]))
            # Zeichne einen gefüllten Kreis (anstelle einer Ellipse) auf die Leinwand
            cv2.circle(canvas, (x, y), radius_width, color_bgr, -1, cv2.LINE_AA)
        return canvas


    def apply_pointilismus(frame, radius):
            image = frame.copy()
            for _ in range(1000):  # Adjust the number of iterations as needed
                x = random.randint(0, image.shape[1] - 1)
                y = random.randint(0, image.shape[0] - 1)
                c = image[y, x]
                zufall = random.randint(2, 15) / 10.0
                radius_scaled = int(radius * zufall)
                cv2.circle(image, (x, y), radius_scaled, (int(c[2]), int(c[1]), int(c[0])), -1)
            return image



    ######## KOMBINIERTE FILTER ######################################################################

    # Pencilsketch + Duotone
    def apply_pencilsketch_and_duotone(frame, color1, color2):
        # Pencil Sketch anwenden
        pencil_sketch = Filter.pencilsketch(frame)
        
        # Duotone-Filter anwenden auf das Pencil Sketch-Bild
        duotone_frame = Filter.apply_duotone_filter(pencil_sketch, color1, color2)
        
        return duotone_frame

    # Duotone + Cartoon
    def apply_duotone_and_cartoon(frame, color1, color2, alpha, beta):
        # Duotone-Filter anwenden
        duotone_frame = Filter.apply_duotone_filter(frame, color1, color2)
        
        # Stilisierungsfilter anwenden
        stylized_frame = Filter.apply_cartoon(duotone_frame)
        
        return stylized_frame

    ######## FILTER FÜR JEWEILIGES KUNSTWERK #########################################################
    # zum Aufruf in der Main Klasse gedacht
    # Funktioniert aus irgendeinem Grund aber nicht
    
    @staticmethod
    def apply_un_dimanche_filter(img):
        img = Filter.apply_pointilismus(img, 10)
        return img

    @staticmethod
    def apply_starry_night_filter(img):
        img = Filter.apply_duotone_and_cartoon(img, Filter.farbe_gelb, Filter.farbe_dunkelblau, 50, 20)
        return img

    @staticmethod
    def apply_the_scream_filter(img):
        img = Filter.apply_cartoon(img)
        return img



    #####################################################################################################################
    # Hauptschleife des Filter Programms:                                                                               #
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
