import shutil
from enum import Enum
from pathlib import Path
from typing import List

import torch
from ultralytics.nn.autoshape import Detections

from backend.backend.config import TEMP_PATH, MODEL_PATH
import cv2

from fastapi import UploadFile


class TaskStatus(Enum):
    NOT_RUNNING = 'NOT_RUNNING'
    IN_PROGRESS = 'IN_PROGRESS'
    FINISH = 'FINISH'
    ERROR = 'ERROR'


class VideoHandler():
    def __init__(self):
        self.result_file_path = None
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', MODEL_PATH)

    def get_status(self) -> TaskStatus:
        pass

    async def start(self, uploaded_file: UploadFile):
        file_path = self.save_file(uploaded_file)
        all_frames = self.cut_file(file_path)
        self.get_predictions(all_frames)

    def save_file(self, uploaded_file: UploadFile) -> Path:
        # файлы надо будет очищать
        # много лишних действий, надо перевести на tempfile
        path = TEMP_PATH / uploaded_file.filename
        with open(path, 'wb') as buffer:
            shutil.copyfileobj(uploaded_file.file, buffer)
        return path

    def cut_file(self, path: Path) -> List[Path]:
        cam = cv2.VideoCapture(path.absolute().as_posix())
        currentframe = 0
        all_frames = []
        while True:
            ret, frame = cam.read()
            if ret:
                name = TEMP_PATH / f'{currentframe}.jpg'
                cv2.imwrite(name.absolute().as_posix(), frame)
                all_frames.append(name)
                currentframe += 1
            else:
                break
        return all_frames

    def get_predictions(self, frame_paths: List[Path]):
        for idx, frame_path in enumerate(frame_paths[:2]):
            prediction: Detections = self.model(frame_path)
            # prediction.pandas() моет пригодиться

            # файл с уже отмеченным креплением
            prediction.save(save_dir=f'backend/temp/{idx}', exist_ok=True)

    def process_frame(self):
        pass

    def create_tagged_video(self):
        pass

    async def stop(self):
        pass
