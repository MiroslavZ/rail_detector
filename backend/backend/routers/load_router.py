from asyncio import create_task
from pathlib import Path

import aiofiles
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from ..video_handler.video_handler import TaskStatus, VideoHandler

router = APIRouter()
UPLOADED_FILE_PATH = ''


@router.post('/upload')
async def upload_video(request: Request, uploaded_file: UploadFile = File(...)):
    handler: VideoHandler = request.app.state.video_handler
    print(uploaded_file.content_type)
    if uploaded_file.content_type != 'video/mp4':
        print('Wrong format file sent')
        raise HTTPException(status_code=422, detail='Wrong format file sent')
    file_hash = hash(uploaded_file)
    async with aiofiles.open(f'{file_hash}.mp4', 'wb') as out_file:
        content = await uploaded_file.read()
        await out_file.write(content)
    file_path = Path(f'{file_hash}.mp4')
    print(file_path)
    print(file_path.exists())
    create_task(handler.start(file_path, file_hash))
    return {'hash': file_hash}


@router.get('/status/{file_hash}')
def get_status(file_hash: int, request: Request):
    handler: VideoHandler = request.app.state.video_handler
    status = handler.get_status(file_hash)
    return {'status': status}


@router.get('/download/{file_hash}')
def download_video(file_hash: int, request: Request):
    handler: VideoHandler = request.app.state.video_handler
    task = handler.get_result(file_hash)
    if task:
        if task.task_status == TaskStatus.IN_PROGRESS:
            raise HTTPException(status_code=503, detail='Final file is not ready yet')
        if task.task_status == TaskStatus.ERROR:
            raise HTTPException(status_code=502, detail='An error occurred while processing the file')
        path = task.result_file_path
        return FileResponse(path=path, filename=path.name, media_type='multipart/form-data')
    raise HTTPException(status_code=404, detail='Item not found')
