import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import requests


def upload(uploaded_file: UploadedFile):
    bytes_data = uploaded_file.getvalue()
    files = {'uploaded_file': bytes_data}
    response = requests.post('http://127.0.0.1:8000/upload', files=files)
    st.write(response.status_code)


uploaded_file: UploadedFile = st.file_uploader(label='Загрузить видео')
if uploaded_file is not None:
    upload(uploaded_file)


def download():
    response = requests.get('http://127.0.0.1:8000/download')
    return response.content


st.download_button(label='Скачать видео', data=download(), file_name='temp.gif')
