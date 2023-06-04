import asyncio
from asyncio import create_task
import io
from enum import Enum
from pathlib import Path
from typing import List

import aiofiles
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
from ultralytics.nn.autoshape import Detections

from backend.backend.config import YOLO_DEFAULT_SAVE_PATH, MODEL_PATH
import cv2


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

    def start(self, file_path: Path, file_hash: int):
        # это io или вычислительная задача?

        create_task(self.handle_file(file_path, file_hash))

    async def handle_file(self, file_path: Path, file_hash: int):
        self.store[file_hash] = VideoHandleTask(TaskStatus.IN_PROGRESS)
        all_frames_paths, width, height, fps = await self.cut_file(str(file_path.resolve()), file_hash)
        loop = asyncio.get_running_loop()
        processed_frames_paths = await loop.run_in_executor(None, self.get_predictions, all_frames_paths)
        str_processed_frames_paths = [str(path.resolve()) for path in processed_frames_paths]
        self.store[file_hash].result_file_path = \
            await loop.run_in_executor(None, self.create_tagged_video, str_processed_frames_paths, width, height, fps, file_hash)
        await loop.run_in_executor(None, self.clear_files, *[file_path, *all_frames_paths, *processed_frames_paths])
        self.store[file_hash].task_status = TaskStatus.FINISH

    async def cut_file(self, path: str, file_hash: int) -> tuple[list[Path], int, int, int]:
        cap = cv2.VideoCapture(path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        current_frame = 0
        all_frames = []
        while True:
            ret, frame = cap.read()
            if ret:
                is_success, buffer = cv2.imencode(".jpg", frame)
                io_buf = io.BytesIO(buffer)
                frame_name = f'{file_hash}_frame{current_frame}.jpg'
                async with aiofiles.open(frame_name, 'wb') as out_file:
                    content = io_buf.read()
                    await out_file.write(content)
                frame_path = Path(frame_name).resolve()
                print(frame_path)
                print(frame_path.exists())
                all_frames.append(frame_path)
                current_frame += 1
            else:
                break
        return all_frames, width, height, fps

    # можно извлечь координаты, вручную выделить объект
    # и добавить информацию о скорости
    # prediction.pandas() может пригодиться
    def get_predictions(self, frame_paths: List[Path]) -> List[Path]:
        processed_frames_paths = []
        for frame_path in frame_paths:
            prediction: Detections = self.model(frame_path)
            # save по умолчанию создает лишнюю папку
            prediction.save(exist_ok=True)
            processed_frame_path = YOLO_DEFAULT_SAVE_PATH / frame_path.name
            print(str(processed_frame_path.resolve()))
            processed_frames_paths.append(processed_frame_path.resolve())
        return processed_frames_paths

    def clear_files(self, *paths: Path):
        for path in paths:
            path = path.resolve()
            if path.exists():
                path.unlink()

    def create_tagged_video(self, frames: List[str], width: int, height: int, fps: int, file_hash: int) -> Path:
        video_name = f'{file_hash}_tagged.mp4'
        print(frames)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
        for frame in frames:
            # imread не работает с Path
            video.write(cv2.imread(frame))
        video.release()
        cv2.destroyAllWindows()
        result_video_path = Path(video_name).resolve()
        print(result_video_path.exists())
        return result_video_path

    # надо определиться, когда удалять итоговый файл
    def get_result(self, file_hash: int) -> (bool, VideoHandleTask):
        return file_hash in self.store, self.store[file_hash]

    # не понятно, как его корректно останавливать
    def stop(self):
        pass
