import argparse
import datetime as dt
import pathlib
import time

import cv2

import sensors

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--brake", type=int, default=3,
                    help="Brake after motion was detected in seconds.")
parser.add_argument("-sd", "--save-dir", type=str, required=True)
parser.add_argument("-a", "--min-motion-area", type=int, default=1.3e4,
                    help="Pixel area in which motion is discounted.")
parser.add_argument("-p", "--print", action="store_true")
args = parser.parse_args()

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open Camera.")

sensor = sensors.HornetSensor()
ret, frame = cap.read()
while ret is not None:
    motion = sensor.detect_motion(frame)
    if motion:
        if args.print:
            print(f"{dt.datetime.now()}: Motion detected")
        file_name = pathlib.Path(args.save_dir, f"{dt.datetime.now()}.jpg")
        cv2.imwrite(file_name, frame)
        time.sleep(args.brake)
    ret, frame = cap.read()

print("No frame returned, exiting...")



