import numpy as np
import cv2

height_counter = 2
width_counter = 1.5

width_length = 100
height_length = 100


def gstreamer_pipeline(
        capture_width=1280,
        capture_height=720,
        display_width=1280,
        display_height=720,
        framerate=30,
        flip_method=0,
):
    return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), "
            "width=(int)%d, height=(int)%d, "
            "format=(string)NV12, framerate=(fraction)%d/1 ! "
            "nvvidconv flip-method=%d ! "
            "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true"
            % (
                capture_width,
                capture_height,
                framerate,
                flip_method,
                display_width,
                display_height,
            )
    )


def process(frame):
    global height_counter, width_counter, height_length, width_length
    color = (255, 0, 0)
    width, height, channels = frame.shape

    thickness = 2
    # WASD управление,
    # 1 - Увеличение высоты
    # 2 - Уменьшение высоты
    # 3 - Увеличение ширины
    # 4 - Уменьшение ширины

    # Частота кадра
    k = cv2.waitKey(100) & 0xFF

    if k == 97:
        # Left
        height_counter += 0.1
    if k == 119:
        # Up
        width_counter += 0.1
    if k == 100:
        # Right
        height_counter -= 0.1
    if k == 115:
        # Down
        width_counter -= 0.1

    if k == 49:
        # Увеличение высоты
        height_length += 25
    if k == 50:
        # Уменьшение высоты
        height_length -= 25
    if k == 51:
        # Увеличение ширины
        width_length += 25
    if k == 52:
        # Уменьшение ширины
        width_length -= 25

    start_point = (int(height / height_counter - width_length / 2), int(width / width_counter - height_length / 2))
    end_point = (int(height / height_counter + width_length / 2), int(width / width_counter + height_length / 2))

    rect = cv2.rectangle(frame, start_point, end_point, color, thickness)
    frame = initColor(frame, rect, color, start_point, end_point, 60, ' Green ')
    frame = initColor(frame, rect, color, start_point, end_point, 120, ' Blue ')
    frame = initColor(frame, rect, color, start_point, end_point, 10, ' Red ')
    frame = initColor(frame, rect, color, start_point, end_point, 170, ' Red ')

    return frame


# Распознает фигуры
def initFig(img):
    # img = cv2.imread('img_2.png')
    # cv2.imshow('mask', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        cv2.drawContours(frame, contour, -1, (255, 0, 0), 3)  # -1 denotes that we need to draw all the contours
        perimeter = cv2.arcLength(contour, True)  # The true indicates that the contour is closed
        print("Perimeter: ", perimeter)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter,
                                  True)  # This method is used to find the approximate number of contours
        print("Corner Points: ", len(approx))
        objCorner = len(approx)
        x, y, w, h = cv2.boundingRect(
            approx)  # In this we get the values of our bounding box that we will draw around the object

        if objCorner == 3:
            objectType = 'Triangle'
            return 'triangle'
        elif objCorner == 4:
            aspectRatio = float(w) / float(h)
            if aspectRatio > 0.95 and aspectRatio < 1.05:
                objectType = 'Square'
                return 'square'
            else:
                objectType = "Rectangle"
                return 'rectangle'
        # else:
        #     return 'no figure'


# Распознает три цвета
def initColor(frame, rect, color, start_point, end_point, colornum, title):
    global height_counter, width_counter, height_length, width_length
    h_sensivity = 20
    s_h = 255
    v_h = 255
    s_l = 50
    v_l = 50
    thickness = 2
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    upper = np.array([colornum + h_sensivity, s_h, v_h])
    lower = np.array([colornum - h_sensivity, s_l, v_l])
    mask_frame = hsv_frame[start_point[1]:end_point[1] + 1, start_point[0]:end_point[0] + 1]

    try:
        number = initFig(frame[start_point[1]:end_point[1] + 1, start_point[0]:end_point[0] + 1])
        mask_green = cv2.inRange(mask_frame, lower, upper)
        rate = np.count_nonzero(mask_green) / (height_length * width_length)
        org = end_point
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.7
        if rate > 0.25:
            text = cv2.putText(rect, title + number, org, font, fontScale, color, thickness, cv2.LINE_AA)
        # else:
        #     text = cv2.putText(rect, 'undefined color', org, font, fontScale, color, thickness, cv2.LINE_AA)
        av_hue = np.average(mask_frame[:, :, 0])
        av_sat = np.average(mask_frame[:, :, 1])
        av_val = np.average(mask_frame[:, :, 2])
        average = [int(av_hue), int(av_sat), int(av_val)]
        text = cv2.putText(rect, str(average) + " " + str(rate), (10, 50), font, fontScale, color, thickness,
                           cv2.LINE_AA)
        frame = text
    except Exception:
        # Rectangle out of range
        pass

    return frame


print('Press Esc to Quit the Application\n')

# Open Default Camera
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=4), cv2.CAP_GSTREAMER)

while cap.isOpened():
    # Take each Frame
    ret, frame = cap.read()

    # Flip Video vertically (180 Degrees)
    frame = cv2.flip(frame, 180)
    # frame = cv2.imread('shape.jpg')
    invert = process(frame)

    # Show video
    cv2.imshow('Cam', invert)

    # Exit if "Esc" is pressed
    k = cv2.waitKey(1) & 0xFF

    if k == 27:  # ord 4
        # Quit
        print('Good Bye!')
        break

# Release the Cap and Video
cap.release()
cv2.destroyAllWindows()