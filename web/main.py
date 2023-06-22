import logging
from time import sleep

from typing import Optional, Dict
import magic
import requests
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
urllib3_logger = logging.getLogger('urllib3')
urllib3_logger.setLevel(logging.CRITICAL)
HOST = 'http://backend:8000'


# add python-magic-bin to requirements.txt
def upload(file_to_upload: UploadedFile):
    bytes_data = file_to_upload.getvalue()
    mime = magic.Magic(mime=True)
    content_type = mime.from_buffer(bytes_data)
    files = {'uploaded_file': (file_to_upload.name, bytes_data, content_type)}
    url = f'{HOST}/upload'
    response = requests.post(url=url, files=files, timeout=30)
    if response.status_code == 200:
        file_hash = response.json().get('hash')
        st.session_state.last_file_hash = file_hash
        wait_handling(file_hash)


def get_statistics(file_hash: int):
    logger.debug('Getting statistics for %s', file_hash)
    url = f'{HOST}/statistics/{file_hash}'
    response = requests.get(url=url, timeout=30)
    if response.status_code == 200:
        result: Dict = response.json()
        st.write('Пройденная дистанция {} м'.format(result.get("total_distance")))
        st.write('Средняя скорость {:.2} м/с'.format(result.get("avg_speed")))
        st.write('Время {:.2} с'.format(result.get("ride_time")))
        st.write('Количество креплений {} шт.'.format(result.get("mounts_count")))


def wait_handling(file_hash: int):
    status = None
    with st.spinner('Пожалуйста подождите...'):
        logger.debug('Waiting for a file to be ready')
        while True:
            url = f'{HOST}/status/{file_hash}'
            response = requests.get(url=url, timeout=30)
            status = response.json().get('status')
            if status != "IN_PROGRESS":
                break
            sleep(5)
    if status == 'FINISH':
        get_statistics(file_hash)


if 'last_file_name' not in st.session_state:
    st.session_state.last_file_name = None
if 'last_file_size' not in st.session_state:
    st.session_state.last_file_size = None

uploaded_file: Optional[UploadedFile] = st.file_uploader(label='Загрузить видео')
if uploaded_file is not None:
    if (
        st.session_state.last_file_name == uploaded_file.name
        and st.session_state.last_file_size == uploaded_file.size
        and 'last_file_hash' in st.session_state
    ):
        logger.debug('Getting already loaded file')
        wait_handling(st.session_state.last_file_hash)
    else:
        st.session_state.last_file_name = uploaded_file.name
        st.session_state.last_file_size = uploaded_file.size
        upload(uploaded_file)
