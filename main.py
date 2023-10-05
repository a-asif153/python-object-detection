import cv2
import time
from emailing import send_email
import glob
import os
from threading import Thread

video = cv2.VideoCapture(0)
time.sleep(1)

first_frame = None
status_list = []

count = 1

def clean_folder():
    print("clean folder function STARTED")
    images = glob.glob("images/*.png")
    for image in images:
        os.remove(image)
    print("clean folder function ENDED")

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

            # capture images
            cv2.imwrite(f"images/{count}.png", frame)
            count = count + 1

            # fetching the middle image
            all_images = glob.glob("images/*.png")
            index = int(len(all_images)/2)
            image_with_object = all_images[index]

    status_list.append(status)
    status_list = status_list[-2:]

    if status_list[0] == 1 and status_list[1] == 0:
        # thread 1 to handle email sending in the background
        email_thread = Thread(target=send_email, args=(image_with_object, ))
        email_thread.daemon = True

        # thread 2 to clean folder in the background
        clean_thread = Thread(target=clean_folder)
        clean_thread.daemon = True

        # calling the email thread execution
        email_thread.start()


    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)

    if key == ord("q"):
        break

video.release()

# calling the clean folder thread
clean_thread.start()