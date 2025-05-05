# Author: Adapted for vehicle detection from Evan Juras' candy detection script

import os
import sys
import cv2
from ultralytics import YOLO

# ======= USER CONFIGURATION =======
model_path = 'yolo11s_vehicle_model.pt'  # Your custom-trained YOLOv5 vehicle model
min_thresh = 0.50                        # Confidence threshold
cam_index = 0                            # Camera index (0 is default webcam)
imgW, imgH = 1280, 720                   # Camera resolution
record = False                           # Set to True if you want to save video

vehicle_info = {'car', 'bus', 'truck'}


# ======= CHECK MODEL EXISTS =======
if not os.path.exists(model_path):
    print('WARNING: Model path is invalid or model was not found.')
    sys.exit()

# ======= LOAD MODEL =======
model = YOLO(model_path, task='detect')
labels = model.names

# ======= INITIALIZE CAMERA =======
cap = cv2.VideoCapture(cam_index)
cap.set(3, imgW)
cap.set(4, imgH)

# ======= SET UP VIDEO RECORDING =======
if record:
    record_name = 'vehicle_detection.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (imgW, imgH))

# ======= COLORS FOR BOUNDING BOXES =======
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (88,159,106), (159,124,168)]

# ======= INFERENCE LOOP =======
while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        print('Camera not connected or unable to read frame.')
        break

    results = model.track(frame, verbose=False)
    detections = results[0].boxes

    for i in range(len(detections)):
        xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh:
            color = bbox_colors[classidx % len(bbox_colors)]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

            label = f'{classname}: {int(conf * 100)}%'
            labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_ymin = max(ymin, labelSize[1] + 10)

            cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                          (xmin + labelSize[0], label_ymin + baseLine - 10), color, cv2.FILLED)
            cv2.putText(frame, label, (xmin, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Show the frame
    cv2.imshow('Vehicle Detection', frame)
    if record:
        recorder.write(frame)

    key = cv2.waitKey(5)
    if key in [ord('q'), ord('Q')]:
        break
    elif key in [ord('s'), ord('S')]:
        cv2.waitKey()
    elif key in [ord('p'), ord('P')]:
        cv2.imwrite('vehicle_capture.png', frame)

# ======= CLEANUP =======
cap.release()
if record:
    recorder.release()
cv2.destroyAllWindows()
