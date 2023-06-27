FROM python:3.11 as stage0

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

FROM stage0 as stage1

COPY requirements.txt /python_requirements/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /python_requirements/requirements.txt

ENV ROOT_DIR /app

WORKDIR $ROOT_DIR

COPY . $ROOT_DIR

FROM stage1 as final

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]