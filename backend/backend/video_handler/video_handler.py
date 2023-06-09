import asyncio
import concurrent.futures
import io
from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional

import cv2
import torch
from PIL import ImageFile
from ultralytics.nn.autoshape import Detections

from ..config import MODEL_PATH

ImageFile.LOAD_TRUNCATED_IMAGES = True


class TaskStatus(Enum):
    NOT_RUNNING = 'NOT_RUNNING'
    IN_PROGRESS = 'IN_PROGRESS'
    FINISH = 'FINISH'
    ERROR = 'ERROR'


class VideoHandleTask:
    def __init__(self, task_status: TaskStatus = TaskStatus.IN_PROGRESS, result_file_path: Path = None):  # type: ignore
        self.task_status = task_status
        self.result_file_path = result_file_path


class VideoParams:
    def __init__(self, width: int, height: int, fps: int):
        self.width = width
        self.height = height
        self.fps = fps


class VideoHandler:
    def __init__(self):
        self.store = {}
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', MODEL_PATH)

    def get_status(self, file_hash: int) -> TaskStatus:
        if file_hash in self.store:
            return self.store[file_hash].task_status
        return TaskStatus.NOT_RUNNING

    async def start(self, file_path: Path, file_hash: int):
        loop = asyncio.get_running_loop()
        with concurrent.futures.ProcessPoolExecutor() as pool:
            self.store[file_hash] = VideoHandleTask(TaskStatus.IN_PROGRESS)
            try:
                task = await loop.run_in_executor(pool, self.handle_file, file_path, file_hash)
                self.store[file_hash] = task
            except Exception:  # pylint: disable=W0718
                self.store[file_hash].task_status = TaskStatus.ERROR

    def handle_file(self, file_path: Path, file_hash: int) -> VideoHandleTask:
        task = VideoHandleTask(TaskStatus.IN_PROGRESS)
        all_frames, params = self.cut_file(str(file_path.resolve()))
        all_frames_paths = [Path(frame) for frame in all_frames]
        processed_frames_paths = self.get_predictions(all_frames_paths)
        str_processed_frames_paths = [str(path.resolve()) for path in processed_frames_paths]
        task.result_file_path = self.create_tagged_video(str_processed_frames_paths, params, file_hash)
        self.clear_files(*[file_path, *all_frames_paths, *processed_frames_paths])
        task.task_status = TaskStatus.FINISH
        return task

    def cut_file(self, path: str) -> tuple[List[str], VideoParams]:
        cap = cv2.VideoCapture(path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        current_frame = 0
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            is_success, buffer = cv2.imencode('.jpg', frame)
            if is_success:
                io_buf = io.BytesIO(buffer)
                frame_file = NamedTemporaryFile(dir='.', delete=False)  # pylint: disable=R1732
                while data := io_buf.read():
                    frame_file.write(data)
                frame_file.close()
                all_frames.append(frame_file.name)
                current_frame += 1
        return all_frames, VideoParams(width, height, fps)

    # можно извлечь координаты, вручную выделить объект
    # и добавить информацию о скорости
    # prediction.pandas() может пригодиться
    def get_predictions(self, frame_paths: List[Path]) -> List[Path]:
        processed_frames_paths = []
        for frame_path in frame_paths:
            prediction: Detections = self.model(frame_path)
            # save по умолчанию создает лишнюю папку
            prediction.save(exist_ok=True, save_dir='.')
            processed_frame_path = Path(f'{frame_path.name}.jpg')
            processed_frames_paths.append(processed_frame_path)
        return processed_frames_paths

    def clear_files(self, *paths: Path):
        for path in paths:
            if path.exists():
                path.unlink()

    def create_tagged_video(self, frames: List[str], params: VideoParams, file_hash: int) -> Path:
        video_name = f'{file_hash}_tagged.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_name, fourcc, params.fps, (params.width, params.height))
        for frame in frames:
            # imread не работает с Path
            video.write(cv2.imread(frame))
        video.release()
        cv2.destroyAllWindows()
        result_video_path = Path(video_name).resolve()
        return result_video_path

    # надо определиться, когда удалять итоговый файл
    def get_result(self, file_hash: int) -> Optional[VideoHandleTask]:
        return self.store.get(file_hash)

    # не понятно, как его корректно останавливать
    def stop(self):
        pass
