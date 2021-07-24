# Amzar Shaik
# Maxis Graduate Program Case Study: In-Store Foot Traffic (Footfall) Analytics - High Traffic Periods
# This program will detect human activity and count the number of humans using a recorded CCTV Video (mp4 file)

import cv2
import math
import numpy as np
import imutils
import argparse
from tracker import *

# Create Tracker Object
tracker = EuclideanDistTracker()

# Initializing the HOG person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Please identify your footage path
path = r"C:\100 Job Applications\2. Applied Companies\2021\Maxis Graduate Program\Footfall Analytics - Python\HomecareStore3.mp4"

cap = cv2.VideoCapture(path)
frameTime = 25

# Object detection from Stable Camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=10)

while cap.isOpened():
  # Reading the video stream
  ret, image = cap.read()
  detections = []
  if ret:
    # image = imutils.resize(image, width=min(400, image.shape[1]))

    # Detecting all the regions in the Image that has a pedestrians inside it
    (regions, _) = hog.detectMultiScale(image, winStride=(4, 4),padding=(4, 4),scale=1.05)

    # Drawing the regions in the Image
    for (x, y, w, h) in regions:

      detections.append([x, y, w, h])

    # Object Tracking
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids:
          x, y, w, h, id = box_id
          cv2.putText(image, str(id), (x, y - 15), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
          cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
          detections.append([x, y, w, h])

    # Showing the output Image
    cv2.imshow("Image", image)
    if cv2.waitKey(frameTime) & 0xFF == ord('q'):
      break
  else:
    break

cap.release()
cv2.destroyAllWindows()