FROM python:3.10

COPY ./requirements.txt /code/requirements.txt

RUN pip install --upgrade pip
RUN pip install -r /code/requirements.txt

# lap не может нормально установиться
# а фикс установки еще не принят
# https://github.com/gatagat/lap/pull/46
RUN pip install -e git+https://github.com/DeGirum/lap.git#egg=lap

# для работы yolo
RUN apt-get update && apt-get install libgl1  -y

COPY backend /code

WORKDIR /code

EXPOSE 8000

CMD ["uvicorn","backend.backend.main:app","--host","0.0.0.0","--port","8000"]