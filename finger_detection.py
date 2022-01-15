import cv2 as cv
import numpy as np

total_rectangle = 9

hand_rect_one_x = None
hand_rect_one_y = None
hand_rect_two_x = None
hand_rect_two_y = None


def draw_rect(frame):
    # Zeichnen der Rechtecke

    rows, cols, _ = frame.shape  # .shape gibt ein Tupel aus Zeilen Spalten und Farbkomponenten zurück

    global total_rectangle, hand_rect_one_x, hand_rect_one_y, hand_rect_two_x, hand_rect_two_y

    hand_rect_one_x = np.array(
        [0.3 * rows, 0.3 * rows, 0.3 * rows, 0.45 * rows, 0.45 * rows, 0.45 * rows, 0.6 * rows, 0.6 * rows, 0.6 * rows], dtype=np.uint32
    )

    hand_rect_one_y = np.array(
        [0.45 * cols, 0.5 * cols, 0.55 * cols, 0.45 * cols, 0.5 * cols, 0.55 * cols, 0.45 * cols, 0.5 * cols, 0.55 * cols], dtype=np.uint32
    )

    hand_rect_two_x = hand_rect_one_x + 10
    hand_rect_two_y = hand_rect_one_y + 10

    for i in range(total_rectangle):
        cv.rectangle(frame, (hand_rect_one_y[i], hand_rect_one_x[i]), (hand_rect_two_y[i], hand_rect_two_x[i]), (0, 255, 0), 1)


def hand_histogram(frame):
    # Erstellen des Histograms

    global hand_rect_one_x, hand_rect_one_y, total_rectangle

    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)  # roi = region of intrest

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10, hand_rect_one_y[i]:hand_rect_one_y[i] + 10]  # Es wird durch das Array durchiteriert und die 900 Pixel in roi gespeichert.

    hand_hist = cv.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])  # berechnet Histogramm für gegebene Arrays
    return cv.normalize(hand_hist, hand_hist, 0, 255, cv.NORM_MINMAX)  # Histogramm wird normalisiert


def masking(frame_face, hist):
    # Erstellen der Maske für Konturenerkennung

    hsv = cv.cvtColor(frame_face, cv.COLOR_BGR2HSV)
    mask = cv.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)  # noch sehr verrauscht

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (37, 37))
    mask = cv.filter2D(mask, -1, kernel)  # Filtern um rauschen zu entfernen

    kernel_erode = cv.getStructuringElement(cv.MORPH_ELLIPSE, (40, 40))
    kernel_dilate = cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10))
    mask = cv.erode(mask, kernel_erode)
    mask = cv.dilate(mask, kernel_dilate)

    _, mask = cv.threshold(mask, 50, 255, cv.THRESH_BINARY)  # Werte der Maske werden entweder 0 oder 255 -> treshold: Entscheidungsschwelle bis zu welchem Wert werden Values 0 gesetzt

    return mask


def contours(hist_mask_image):
    gray_hist_mask_image = cv.cvtColor(hist_mask_image, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray_hist_mask_image, 0, 255, 0)
    cont, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return cont


def centroid(max_contour):
    moment = cv.moments(max_contour)

    # Berechnung geometrischer Schwerpunkt über Moment
    if moment['m00'] != 0:
        x = int(moment['m10'] / moment['m00'])
        y = int(moment['m01'] / moment['m00'])
        return x, y
    else:
        return None


def face_detection(frame):
    # Gesichtserkennung mit Hilfe eines bereits gelernten ML Algorithmus von opencv

    # einlesen der Datei
    face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Anwendung des Algorithmus
    faces = face_cascade.detectMultiScale(frame_gray)
    for x, y, w, h in faces:
        # Gesichter schwärzen
        cv.ellipse(frame, (int(x+0.5*w), int(y+0.5*h)), (int(w*0.75), h), 0, 0, 360, 0, -1)
    return frame


def remove_background(frame, bg):
    # Ansatz zum Entfernen des Hintergrunds
    # Angefertigter Screenshot am Anfang wird mit aktuellem Frame verglichen. Nur veränderte Pixel werden angezeigt.

    threshold = 10
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    bg = cv.cvtColor(bg, cv.COLOR_BGR2GRAY)
    rows, cols = frame_gray.shape
    for i in range(rows):
        for j in range(cols):
            if frame_gray[i, j] >= bg[i, j] - threshold and frame_gray[i, j] <= bg[i, j] + threshold:
                frame_gray[i, j] = 0
            else:
                frame_gray[i, j] = 255
    mask = cv.merge((frame_gray, frame_gray, frame_gray))
    frame = cv.bitwise_and(frame, mask)
    return frame


def finger_detection(frame_bg, frame, hand_hist):
    mask = masking(frame_bg, hand_hist)
    mask = cv.merge((mask, mask, mask))  # wir haben 3 Farbkomponenten
    hist_mask_image = cv.bitwise_and(face_detection(frame_bg), mask)

    # größte Kontur finden und anzeigen
    contour_list = contours(hist_mask_image)
    max_cont = max(contour_list, key=cv.contourArea)
    cv.drawContours(frame_bg, max_cont, -1, 1)

    # Mittelpunkt der Kontur finden und anzeigen
    cont_centroid = centroid(max_cont)
    cv.circle(frame_bg, cont_centroid, 5, [255, 0, 255], -1)

    if max_cont is not None:
        hull = cv.convexHull(max_cont, returnPoints=False)
        draw_hull = cv.convexHull(max_cont)
        defects = cv.convexityDefects(max_cont, hull)
        cv.drawContours(frame_bg, [draw_hull], 0, [0, 0, 255], 1)

        points = []
        dist = lambda p1, p2: (np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2))

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(max_cont[s][0])
            #end = tuple(max_cont[e][0])
            #far = tuple(max_cont[f][0])
            if d/256 > 9:  # nur Punkte mit ensprechender defect tiefe
                points.append(start)

        finger_tips = []
        _, y_rect, b_rect , h_rect = cv.boundingRect(max_cont)
        for i in range(len(points)-1):  # durch alle Punkte iterieren
            #if dist(points[i], points[i+1]) > 45:  # distanz zwischen zwei Punkten muss über 45 sein
            if dist(points[i], cont_centroid) > 0.40*((h_rect+b_rect)/2):  # distanz zwischen Punkt und Handmitte muss über 40% der Größe der Bounding Box entsprechen
                if points[i][1] < y_rect+0.8*h_rect:  # Punkte die an der Unterseite der Handliegen werden nicht eingeblendet
                    if len(finger_tips) < 5:  # maximal 5 finger
                        finger_tips.append(points[i])

        for finger in finger_tips:
            cv.circle(frame, finger, 5, [0, 0, 255], -1)
            cv.circle(frame_bg, finger, 5, [0, 0, 255], -1)

        cv.putText(frame, str(len(finger_tips)), cont_centroid, cv.FONT_HERSHEY_PLAIN, 10, [255, 0, 0], 2)


def main():
    is_hand_hist_created = False
    capture = cv.VideoCapture(0)

    while capture.isOpened():
        _, frame_bg = capture.read()  # background frame
        _, frame = capture.read() # frame
        pressed_key = cv.waitKey(5)

        if pressed_key & 0xFF == ord('c'):
            hand_hist = hand_histogram(frame)
            is_hand_hist_created = True

        if is_hand_hist_created:
            #frame = remove_background(frame, bg)
            finger_detection(frame_bg, frame, hand_hist)

        else:
            draw_rect(frame)

        cv.imshow('webcam', frame)

        if pressed_key & 0xFF == ord('d'):
            break

    capture.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()




