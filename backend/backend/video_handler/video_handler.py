import asyncio
import concurrent.futures
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Dict
import logging

import cv2
import numpy as np
from PIL import ImageFile
from ultralytics import YOLO
from ultralytics.yolo.engine.results import Results

from ..config import MODEL_PATH
from ..utils.sort import Sort

ImageFile.LOAD_TRUNCATED_IMAGES = True
logging.basicConfig(level='DEBUG')
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    NOT_RUNNING = 'NOT_RUNNING'
    IN_PROGRESS = 'IN_PROGRESS'
    FINISH = 'FINISH'
    ERROR = 'ERROR'


class RailMount:
    def __init__(self, box: Tuple, mount_number: int = 0, current_speed: float = 0.0, current_distance: float = 0.0):
        self.box = box
        self.mount_number = mount_number
        self.current_speed = current_speed
        self.current_distance = current_distance


class Statistics:
    def __init__(self, total_distance: float = 0, avg_speed: float = 0, ride_time: float = 0, mounts_count: int = 0,
                 mounts_dict: Dict[Results, List[RailMount]] = None):
        self.total_distance = total_distance
        self.avg_speed = avg_speed
        self.ride_time = ride_time
        self.mounts_count = mounts_count
        # кадр видео => прогноз модели => одно/несколько креплений
        if mounts_dict:
            self.mounts_dict = mounts_dict
        else:
            self.mounts_dict = {}


class VideoHandleTask:
    def __init__(self, task_status: TaskStatus = TaskStatus.IN_PROGRESS,
                 generated_statistics: Statistics = None):
        self.task_status = task_status
        self.generated_statistics = generated_statistics


class VideoHandler:
    def __init__(self):
        self.store: Dict[int, VideoHandleTask] = {}
        if MODEL_PATH.exists():
            self.model = YOLO(MODEL_PATH)
        else:
            logger.error('Weights for model not found')
            raise FileNotFoundError

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
            except Exception as err:
                logger.error(err)
                self.store[file_hash].task_status = TaskStatus.ERROR

    def handle_file(self, file_path: Path, file_hash: int) -> VideoHandleTask:
        task = VideoHandleTask()
        logger.debug('Getting predictions for each frame...')
        predictions = self.get_predictions(file_path)
        logger.debug('Handling predictions, calculating mounts...')
        mounts_count, mounts_dict = self.calc_mounts(predictions)
        logger.debug('Getting predictions for each frame...')
        distance = self.calc_distance(mounts_count)
        logger.debug('Getting video durations for future calculations...')
        ride_time = self.get_video_duration_seconds(str(file_path))
        average_speed = distance / ride_time
        logger.debug('Saving statistics for %s', file_hash)
        task.generated_statistics = Statistics(distance, average_speed, ride_time, mounts_count)
        logger.debug('Cleaning temporary files')
        self.clear_files(file_path)
        task.task_status = TaskStatus.FINISH
        return task

    def get_predictions(self, file_path: Path) -> List[Results]:
        predictions: List[Results] = self.model(file_path, stream=True)
        return predictions

    def calc_mounts(self, predictions: List[Results]) -> Tuple[int, Dict[Results, List[RailMount]]]:
        mounts_count = 0
        tracker = Sort()
        mounts_dict = {}
        for result in predictions:
            if result.boxes:
                count = 0
                detects = np.zeros((len(result.boxes), 5))
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    c = int(box.cls)
                    box = np.array([x1, y1, x2, y2, c])
                    detects[count, :] = box[:]
                    count += 1
                    if result not in mounts_dict:
                        mounts_dict[result] = []
                    mounts_dict[result].append(RailMount((x1, y1, x2, y2, c)))
                if len(detects) != 0:
                    trackers = tracker.update(detects)
                    if trackers.any():
                        logger.debug(trackers)
                        for mount, tr in zip(mounts_dict[result], trackers):
                            mount.mount_number = tr[-1]
            else:
                logger.debug('No mounts detected')
                mounts_dict[result] = []
                tracker.update(np.empty((0, 5)))
        return mounts_count, mounts_dict

    def calc_distance(self, mounts_count) -> float:
        # расстояние между креплениями 0,42м
        return (mounts_count-1) * 0.42 if (mounts_count-1) * 0.42 >= 0 else 0.0

    def get_video_duration_seconds(self, file_path: str):
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return frame_count / fps

    def clear_files(self, *paths: Path):
        for path in paths:
            if path.exists():
                path.unlink()

    def get_result(self, file_hash: int) -> Tuple[TaskStatus, Dict]:
        if file_hash in self.store:
            status = self.store[file_hash].task_status
            if status == TaskStatus.FINISH:
                statistics = self.store[file_hash].generated_statistics
                # объекты предсказаний yolo нужно сериализовать через Results.tojson()
                response = {
                    'total_distance': statistics.total_distance,
                    'avg_speed': statistics.avg_speed,
                    'ride_time': statistics.ride_time,
                    'mounts_count': statistics.mounts_count
                }
                return status, response
            return status, {}
        return TaskStatus.NOT_RUNNING, {}

    # не понятно, как его корректно останавливать
    def stop(self):
        pass
