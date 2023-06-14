import logging
from time import sleep

import magic
import requests
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# add python-magic-bin to requirements.txt
def upload(uploaded_file: UploadedFile):
    bytes_data = uploaded_file.getvalue()
    mime = magic.Magic(mime=True)
    content_type = mime.from_buffer(bytes_data)
    files = {'uploaded_file': (uploaded_file.name, bytes_data, content_type)}
    url = 'http://127.0.0.1:8000/upload'
    response = requests.post(url=url, files=files)
    if response.status_code == 200:
        file_hash = response.json().get('hash')
        st.session_state.last_file_hash = file_hash
        wait_handling(file_hash)


def download(file_hash: int):
    logger.debug(f'Downloading file {file_hash}')
    url = f'http://127.0.0.1:8000/download/{file_hash}'
    response = requests.get(url=url)
    if response.status_code == 200:
        st.download_button(label='Скачать видео', data=response.content, file_name='video.mp4', mime='video/mp4')


def wait_handling(file_hash: int):
    status = None
    with st.spinner('Пожалуйста подождите...'):
        logger.debug('Waiting for a file to be ready')
        while True:
            url = f'http://127.0.0.1:8000/status/{file_hash}'
            response = requests.get(url=url)
            status = response.json().get('status')
            if status != "IN_PROGRESS":
                break
            sleep(5)
    if status == 'FINISH':
        download(file_hash)


if 'last_file_name' not in st.session_state:
    st.session_state.last_file_name = None
if 'last_file_size' not in st.session_state:
    st.session_state.last_file_size = None

uploaded_file: UploadedFile = st.file_uploader(label='Загрузить видео')
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
