import cv2
import pandas
from datetime import datetime

video = cv2.VideoCapture(0)

status_list = [None, None]
times = []
df = pandas.DataFrame(columns=["Start", "End"])

initial_frame = None

while True:
    frame_status = 0
    check, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if initial_frame is None:
        initial_frame = gray
        continue

    # get diference in pixels between initial frame and current frame
    frame_difference = cv2.absdiff(initial_frame, gray)

    # grayscale frame. threshold checks how much movement it tracks, lower value tracks the slightest movements in background
    thresh_frame = cv2.threshold(
        frame_difference, 30, 255, cv2.THRESH_BINARY)[1]

    # smoothen threshold
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # finding coutours. This is stored as a vector
    cnts, hierarchy = cv2.findContours(thresh_frame.copy(),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in cnts:
        if cv2.contourArea(contour) < 10000:
            continue
        frame_status = 1

        # gets contour boundaries
        (x, y, w, h) = cv2.boundingRect(contour)

        # draws a rectangle around the contours
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # print(cnts)

    status_list.append(frame_status)

    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())
    if status_list[-1] == 1 and status_list[-2] == 0:
        times.append(datetime.now())

    cv2.imshow("Capturing", gray)
    cv2.imshow("Frame difference", frame_difference)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)

    key = cv2.waitKey(1)
    # print(gray)
    # print(frame_difference)

    if key == ord('q'):
        if frame_status == 1:
            times.append(datetime.now())
        break


print(status_list)
print(times)

for i in range(0, len(times), 2):
    print(i)
    print(len(times))
    df = df.append({"Start": times[i], "End": times[i+1]}, ignore_index=True)

df.to_csv("times.csv")

video.release()
cv2.destroyAllWindows
