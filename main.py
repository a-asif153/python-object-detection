import cv2
import time
from emailing import send_email

video = cv2.VideoCapture(0)
time.sleep(1)

first_frame = None
status_list = []

while True:
    status = 0

    # check if video.read is true and get its frame
    check, frame = video.read()

    # convert to grayscale (why? :)), apply gaussian blur
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame_gau = cv2.GaussianBlur(gray_frame, (21,21), 0)

    # show the video
    cv2.imshow("My video", gray_frame_gau)

    # to ensure only first frame is stored in variable only once
    if first_frame is None:
        first_frame = gray_frame_gau

    # difference in matrix values
    delta_frame = cv2.absdiff(first_frame, gray_frame_gau)

    # adjusting the threshold, so its absolute white or black (minimizing data complexity)
    thresh_frame = cv2.threshold(delta_frame, 60, 255, cv2.THRESH_BINARY)[1]

    # reduce noise
    dil_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # IDENTIFY CONTOURS - Returns a list of all contours (white areas)
    contours, check = cv2.findContours(dil_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 60000:
            continue

        # build rectangle for contour > 60 k
        x, y, w, h = cv2.boundingRect(contour)
        rectangle = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

        if rectangle.any():
            status = 1

    status_list.append(status)
    status_list = status_list[-2:]

    if status_list[0] == 1 and status_list[1] == 0:
        send_email()

    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

video.release()
