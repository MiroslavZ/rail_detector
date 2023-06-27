FROM python:3.11 as stage0

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install libgl1  -y

FROM stage0 as stage1

COPY requirements.txt /python_requirements/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /python_requirements/requirements.txt
# lap не может нормально установиться ,а фикс установки еще не принят
# https://github.com/gatagat/lap/pull/46
RUN pip install -e git+https://github.com/DeGirum/lap.git#egg=lap

ENV ROOT_DIR /app

WORKDIR $ROOT_DIR

COPY backend $ROOT_DIR/backend

FROM stage1 as final

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]