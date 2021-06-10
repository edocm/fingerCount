import cv2 as cv
import numpy as np

total_rectangle = 9

hand_rect_one_x = None
hand_rect_one_y = None
hand_rect_two_x = None
hand_rect_two_y = None


def draw_rect(frame):
    rows, cols, _ = frame.shape  # .shape gibt ein Tupel aus Zeilen Spalten und Farbkomponenten zurück
    print("rows: ", rows)
    print("columns: ", cols)

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

    return frame


def hand_histogram(frame):
    global hand_rect_one_x, hand_rect_one_y, total_rectangle

    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)  # roi = region of intrest

    for i in range(total_rectangle):
        roi[i * 10: i * 10 + 10, 0: 10] = hsv_frame[hand_rect_one_x[i]:hand_rect_one_x[i] + 10, hand_rect_one_y[i]:hand_rect_one_y[i] + 10]  # Es wird durch das Array durchiteriert und die 900 Pixel in roi gespeichert.

    hand_hist = cv.calcHist([roi], [0, 1], None, [180, 256], [0, 180, 0, 256])  # berechnet Histogramm für gegebene Arrays
    return cv.normalize(hand_hist, hand_hist, 0, 255, cv.NORM_MINMAX)  # Histogramm wird normalisiert


def hist_masking(frame, hist):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    mask = cv.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)  # noch sehr verrauscht

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (31, 31))
    mask = cv.filter2D(mask, -1, kernel)  # Filtern um rauschen zu entfernen

    _, mask = cv.threshold(mask, 100, 255, cv.THRESH_BINARY)  # Werte der Maske werden entweder 0 oder 255 -> treshold: Entscheidungsschwelle bis zu welchem Wert werden Values 0 gesetzt

    mask = cv.merge((mask, mask, mask))  # wir haben 3 Farbkomponenten

    return cv.bitwise_and(frame, mask)  # Bitweise AND-Verknüpfung des Originalbildes mit der Maske


def finger_detection(frame, hand_hist):
    hist_mask_image = hist_masking(frame, hand_hist)

    # größte Kontur finden und anzeigen
    contour_list = contours(hist_mask_image)
    max_cont = max(contour_list, key=cv.contourArea)
    cv.drawContours(frame, max_cont, -1, 1)

    # Mittelpunkt der Kontur finden und anzeigen
    cont_centroid = centroid(max_cont)
    cv.circle(frame, cont_centroid, 5, [255, 0, 255], -1)

    return frame


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


def main():
    is_hand_hist_created = False
    capture = cv.VideoCapture(0)

    while capture.isOpened():
        _, frame = capture.read()
        pressed_key = cv.waitKey(5)

        if pressed_key & 0xFF == ord('c'):
            hand_hist = hand_histogram(frame)
            is_hand_hist_created = True

        if is_hand_hist_created:
            frame = finger_detection(frame, hand_hist)

        else:
            frame = draw_rect(frame)

        cv.imshow('webcam', frame)

        if pressed_key & 0xFF == ord('d'):
            break

    capture.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()




