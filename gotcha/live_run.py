import argparse

import cv2

from sensors import HornetSensor

parser = argparse.ArgumentParser(description=("Runs a live demo of VespAI "
                                              "using the default camera."))
parser.add_argument("-mf", "--max-fps", type=int, default=30)
args = parser.parse_args()
if args.max_fps < 0:
    raise RuntimeError("Cannot use --max-fps negative.")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open Camera.")

sensor = HornetSensor()
ret, frame = cap.read()
while ret is not None:
    detection = sensor.detect_species(frame)
    if detection is not None:
        frame = detection.result
    cv2.imshow("Detection", frame)
    cv2.waitKey(1_000 // args.max_fps)
    ret, frame = cap.read()

print("No frame returned, exiting...")
cv2.destroyAllWindows()
