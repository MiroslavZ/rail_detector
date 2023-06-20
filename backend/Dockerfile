FROM ultralytics/yolov5:latest

COPY ./requirements.txt /code/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /code/requirements.txt

#RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libgl1  -y
RUN apt-get update && apt-get install libgl1  -y

COPY backend /code/backend

WORKDIR /code

EXPOSE 8000

CMD ["uvicorn","backend.backend.main:app","--host","0.0.0.0","--port","8000"]