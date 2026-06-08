import cv2

from sensors import HornetSensor

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Cannot open Camera.")

sensor = HornetSensor()
MAX_FPS = 30
ret, frame = cap.read()
while ret is not None:
    detection = sensor.detect(frame)
    if detection is not None:
        frame = detection.result
    cv2.imshow('Detection', frame)
    cv2.waitKey(1_000 // MAX_FPS)
    ret, frame = cap.read()

print("No frame returned, exiting...")
cv2.destroyAllWindows()