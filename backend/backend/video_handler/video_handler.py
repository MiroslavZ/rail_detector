import asyncio
import concurrent.futures
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Dict, Optional
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
                 mounts_dict: OrderedDict[Results, List[RailMount]] = None):
        self.total_distance = total_distance
        self.avg_speed = avg_speed
        self.ride_time = ride_time
        self.mounts_count = mounts_count
        # кадр видео => прогноз модели => одно/несколько креплений
        if mounts_dict:
            self.mounts_dict = mounts_dict
        else:
            self.mounts_dict = OrderedDict()


class VideoHandleTask:
    def __init__(self, task_status: TaskStatus = TaskStatus.IN_PROGRESS,
                 generated_statistics: Statistics = None, result_file_path: Path = None):
        self.task_status = task_status
        self.generated_statistics = generated_statistics
        self.result_file_path = result_file_path


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
        task.generated_statistics = Statistics(distance, average_speed, ride_time, mounts_count, mounts_dict)
        logger.debug('Calculating of intermediate distance for each mount...')
        self.calc_intermediate_distances(task.generated_statistics)
        # logger.debug('Calculating of intermediate speed for each mount...')
        # расчет промежуточных скоростей для каждого крепления
        logger.debug('Creating tagged video...')
        result_video_path = self.create_tagged_video(str(file_path), file_hash, task.generated_statistics)
        task.result_file_path = result_video_path
        logger.debug('Cleaning temporary files')
        self.clear_files(file_path)
        task.task_status = TaskStatus.FINISH
        logger.debug('Done!!!')
        return task

    def get_predictions(self, file_path: Path) -> List[Results]:
        predictions: List[Results] = self.model(file_path, stream=True)
        return predictions

    def calc_mounts(self, predictions: List[Results]) -> Tuple[int, OrderedDict[Results, List[RailMount]]]:
        mounts_count = 0
        tracker = Sort()
        mounts_dict = OrderedDict()
        for result in predictions:
            if result.boxes:
                count = 0
                detects = np.zeros((len(result.boxes), 5))
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    c = int(box.cls)
                    coord_array = np.array([x1, y1, x2, y2, c])
                    detects[count, :] = coord_array[:]
                    count += 1
                    if result not in mounts_dict:
                        mounts_dict[result] = []
                    # xyxy[0] это массив из float тензоров размера 0,
                    # но для отрисовки нужны целые координаты
                    coords_for_mount = tuple(int(a) for a in box.xyxy[0].tolist())
                    mounts_dict[result].append(RailMount(coords_for_mount))
                if len(detects) != 0:
                    trackers = tracker.update(detects)
                    if trackers.any():
                        logger.debug(trackers)
                        for mount, tr in zip(mounts_dict[result], trackers):
                            mount.mount_number = tr[-1]
                            if tr[-1] > mounts_count:
                                mounts_count = tr[-1]
            else:
                logger.debug('No mounts detected')
                mounts_dict[result] = []
                tracker.update(np.empty((0, 5)))
        logger.debug('detected {} mounts, generated {} predictions'.format(mounts_count, len(mounts_dict.keys())))
        return mounts_count, mounts_dict

    def calc_distance(self, mounts_count) -> float:
        # расстояние между креплениями 0,42м
        return (mounts_count - 1) * 0.42 if (mounts_count - 1) * 0.42 >= 0 else 0.0

    def get_video_duration_seconds(self, file_path: str):
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return frame_count / fps

    def calc_intermediate_distances(self, statistics: Statistics):
        for mounts_list in statistics.mounts_dict.values():
            for mount in mounts_list:
                mount.current_distance = self.calc_distance(mount.mount_number)

    def calc_intermediate_speed(self, statistics: Statistics):
        # не понятно, какое крепление считается "предыдущим"
        # а если крепления оба обнаружены впервые на одном кадре?
        # как с текущей организацией данных считать время?
        for index, mounts_list in enumerate(statistics.mounts_dict.values()):
            pass

    def create_tagged_video(self, file_path: str, file_hash: int, statistics: Statistics) -> Path:
        cap = cv2.VideoCapture(file_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        video_name = f'{file_hash}_tagged.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
        # количество кадров совпадает с количеством прогнозов
        frame_index = 0
        for prediction in statistics.mounts_dict.keys():
            mounts_list = statistics.mounts_dict[prediction]
            ret, frame = cap.read()
            if not ret:
                logger.debug('Unable to read frame {}, maybe the video is over'.format(frame_index))
                break
            draw = frame.copy()
            for mount in mounts_list:
                x_start, y_start, x_end, y_end = mount.box
                color = (52, 250, 66)
                fontScale = 1
                thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_coordinates = self.get_right_text_coordinates(x_start, y_start, width, height)
                number = mount.mount_number
                distance = mount.current_distance
                text = '{} S={}'.format(number, distance)
                logger.debug(text_coordinates)
                draw = cv2.putText(draw, text, text_coordinates, font, fontScale, color, thickness)
                # image = cv2.putText(image, 'OpenCV', org, font, fontScale, color, thickness, cv2.LINE_AA)
                draw = cv2.rectangle(draw, (x_start, y_start), (x_end, y_end), color, thickness)
                video.write(draw)
            frame_index += 1
        video.release()
        # а нужно ли оно здесь? мы же без интерфейса работаем
        cv2.destroyAllWindows()
        result_video_path = Path(video_name).resolve()
        return result_video_path

    # в идеале при размещении информации нужно учитывать границы изображения
    # и определять оптимальное место куда поместить текст
    def get_right_text_coordinates(self, x: int, y: int, width: int, height: int) -> Tuple[int, int]:
        return x, y - 5

    def clear_files(self, *paths: Path):
        for path in paths:
            if path.exists():
                path.unlink()

    def get_statistics(self, file_hash: int) -> Tuple[TaskStatus, Dict]:
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

    def get_tagged_video(self, file_hash: int) -> Tuple[TaskStatus, Optional[Path]]:
        # похоже на копипаст с get_statistics
        if file_hash in self.store:
            status = self.store[file_hash].task_status
            if status == TaskStatus.FINISH:
                return status, self.store[file_hash].result_file_path
            return status, None
        return TaskStatus.NOT_RUNNING, None

    # не понятно, как его корректно останавливать
    def stop(self):
        pass
