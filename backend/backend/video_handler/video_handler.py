from asyncio import create_task, Task
import io
import os
from enum import Enum
from pathlib import Path
from typing import List, Union
from tempfile import NamedTemporaryFile
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from typing.io import IO
from ultralytics.nn.autoshape import Detections

from backend.backend.config import YOLO_DEFAULT_SAVE_PATH, MODEL_PATH
import cv2

from fastapi import UploadFile


class TaskStatus(Enum):
    NOT_RUNNING = 'NOT_RUNNING'
    IN_PROGRESS = 'IN_PROGRESS'
    FINISH = 'FINISH'
    ERROR = 'ERROR'


class VideoHandleTask:
    def __init__(self, task_status: TaskStatus = None, result_file_path: Path = None):
        self.task_status = task_status
        self.result_file_path = result_file_path


class VideoHandler:
    def __init__(self):
        self.store = dict()
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', MODEL_PATH)

    def get_status(self, file_hash: int) -> TaskStatus:
        if file_hash in self.store:
            return self.store[file_hash].task_status
        return TaskStatus.NOT_RUNNING

    def start(self, uploaded_file: UploadFile) -> int:
        file_hash = hash(uploaded_file)
        # это io или все таки вычислительная задача?
        self.handle_file(uploaded_file)
        return file_hash

    def handle_file(self, uploaded_file: UploadFile):
        self.store[hash(uploaded_file)] = VideoHandleTask(TaskStatus.IN_PROGRESS)

        temporary_file = self.save_file(uploaded_file)
        file_path = temporary_file.name
        all_frames = self.cut_file(file_path)
        temporary_file.close()
        os.unlink(file_path)
        all_frames_paths = [frame.name for frame in all_frames]
        processed_frames_paths = self.get_predictions(all_frames_paths)
        self.clear_files(all_frames)
        result_file_path = self.create_tagged_video(processed_frames_paths, str(hash(uploaded_file)))
        self.store[hash(uploaded_file)].task_status = TaskStatus.FINISH
        self.store[hash(uploaded_file)].result_file_path = result_file_path
        for frame in processed_frames_paths:
            os.unlink(frame)

    def save_file(self, uploaded_file: UploadFile) -> Union[IO, IO[bytes]]:
        file_fd = NamedTemporaryFile(dir='.', delete=False)
        while data := uploaded_file.file.read(1024):
            file_fd.write(data)
        print(file_fd)
        return file_fd

    def cut_file(self, path: str) -> List[Union[IO, IO[bytes]]]:
        cam = cv2.VideoCapture(path)
        currentframe = 0
        all_frames = []
        while True:
            ret, frame = cam.read()
            if ret:
                is_success, buffer = cv2.imencode(".jpg", frame)
                io_buf = io.BytesIO(buffer)
                frame_file = NamedTemporaryFile(dir='.', delete=False)
                while data := io_buf.read(1024):
                    frame_file.write(data)
                all_frames.append(frame_file)
                currentframe += 1
                print(currentframe)
            else:
                break
        return all_frames

    def get_predictions(self, frame_paths: List[str]) -> List[str]:
        processed_frames_paths = []
        for frame_path in frame_paths:
            prediction: Detections = self.model(frame_path)
            # prediction.pandas() может пригодиться
            prediction.save(exist_ok=True)
            processed_frame_path = YOLO_DEFAULT_SAVE_PATH / f'{Path(frame_path).name}.jpg'
            print(str(processed_frame_path.resolve()))
            processed_frames_paths.append(str(processed_frame_path.resolve()))
        return processed_frames_paths

    def clear_files(self, files: List[Union[IO, IO[bytes]]]):
        for f in files:
            f.close()
            os.unlink(f.name)

    def create_tagged_video(self, frames: List[str], filename: str) -> Path:
        video_name = f'{filename}.mp4'
        print(frames)
        frame = cv2.imread(frames[0])
        height, width, layers = frame.shape
        # частоту кадров нужно доставать из исходного видео
        video = cv2.VideoWriter(video_name, 0, 30, (width, height))  # 30 fps
        for frame in frames:
            video.write(cv2.imread(frame))
        cv2.destroyAllWindows()
        video.release()
        result_video_path = Path(video_name)
        print(result_video_path.exists())
        return result_video_path

    # надо определиться когда удалять итоговый файл
    def get_result(self, file_hash: int) -> (bool, Path):
        return file_hash in self.store, self.store[file_hash].result_file_path

    def stop(self):
        pass
