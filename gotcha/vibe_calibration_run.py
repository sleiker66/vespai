import argparse

import cv2

import sensors

parser = argparse.ArgumentParser()
parser.add_argument("--area-tol", type=int, default=1.3e4)
parser.add_argument("--max-samples", type=int, default=200)
parser.add_argument("--video-file", type=str)
args = parser.parse_args()

cap = cv2.VideoCapture(args.video if args.video else 0)
sensor = sensors.HornetSensor()
sensor.test_vibe_calibration(args.area_tol, cap, args.max_samples)