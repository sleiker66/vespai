from pathlib import Path
from typing import NamedTuple, Optional, Any

import numpy as np
import cv2
import torch

from vibe import BackgroundSubtractor

REPO_ROOT = Path(__file__).parent.parent
YOLO_DIR = Path(REPO_ROOT, 'models', 'yolov5')
YOLO_PARAMS_FILE = Path(REPO_ROOT, 'models', 'yolov5-params/yolov5s-all-data.pt')

Detection = NamedTuple('Detection', result=Any, ah_count=int, eh_count=int)


class HornetSensor:
    """Sensor for detecting Asian and European hornets in Images."""

    def __init__(self) -> None:
        self.yolo_model = torch.hub.load(
            str(YOLO_DIR),
            'custom',
            path=str(YOLO_PARAMS_FILE),
            source='local',
            _verbose=False
        )
        self.vibe_model = None

    def detect_motion(self, frame: np.ndarray, area_tol: float = 1.3e4, dilation_strength: int = 1) -> bool:
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected a numpy array, got {type(frame).__name__}")

        frame_gray_scale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.vibe_model is None:
            print('Initializing vibe model on first frame received.')
            self.vibe_model = BackgroundSubtractor()
            self.vibe_model.init_history(frame_gray_scale)
            return False

        segmentation = self.vibe_model.segmentation(frame_gray_scale)
        self.vibe_model.update(frame_gray_scale, segmentation)
        segmentation = cv2.medianBlur(segmentation, 3)

        dilation = cv2.dilate(segmentation, None, iterations=dilation_strength)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) > area_tol:  # Motion found
                return True
        return False

    def detect_species(
            self,
            frame: np.ndarray,
            motion_detection=False,
            area_tol: float = 1.3e4,
            dilation_strength: int = 1
    ) -> Optional[Detection]:
        """Detects Vespa velutina (Asian hornet) and Vespa crabro (European hornet) in the given frame.

        Performs object detection on the given frame. When any Vespa velutina or Vespa crabro are detected, bounding
        boxes are applied and the frame gets returned in a Detection object.
        When motion detection is enabled, the Hornet Detector keeps track of the received frames and only performs
        object detection when significant motion is detected otherwise returning None, making the process more
        efficient. When receiving the first frame and motion detection is enabled, None is returned, because the first
        frame is used to init motion detection.

        Args:
            frame: The frame to run the object detection on, expects a np.ndarray in BGR format.
            motion_detection: Enables motion detection.
            area_tol: Tolerated motion, with reasonable default.
            dilation_strength: Dilation used for motion detection.

        Returns:
            A Detection object if hornets are detected, otherwise None.

        Raises:
            TypeError: If frame is not a np.ndarray.
        """

        if not isinstance(frame, np.ndarray):
            raise TypeError(f"Expected a numpy array, got {type(frame).__name__}")

        if motion_detection and not self.detect_motion(frame, area_tol=area_tol, dilation_strength=dilation_strength):
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.yolo_model(frame_rgb)
        predictions = results.pred[0]  # 0-th element of 1-elt list
        ah_count, eh_count = 0, 0
        for p in predictions:
            if p[-1] == 1:
                ah_count += 1
            if p[-1] == 0:
                eh_count += 1

        if ah_count + eh_count > 0:
            results.render()
            frame_with_bounding_boxes = results.ims[0]
            frame_bgr = cv2.cvtColor(frame_with_bounding_boxes, cv2.COLOR_RGB2BGR)
            return Detection(result=frame_bgr, ah_count=ah_count, eh_count=eh_count)
        return None

    def test_vibe_calibration(self, area_tol: int, cap: cv2.VideoCapture, max_samples: int = 200):
        if not cap.isOpened():
            raise RuntimeError("Cap is closed")

        success, frame = cap.read()
        if success:
            self.detect_motion(frame, area_tol=area_tol) # Init vibe on first frame
        else:
            raise RuntimeError("No frame returned")

        num_motions = 0
        num_total = 0
        while success and num_total < max_samples:
            success, frame = cap.read()
            if not success:
                break
            motion = self.detect_motion(frame, area_tol=area_tol)
            num_total += 1
            if motion:
                num_motions += 1

        print("No more frames returned or max samples reached, exiting...")
        print(f"With area_tol {area_tol}: {num_motions} motions in {num_total} frames.")
        return num_total, num_motions
