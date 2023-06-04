from pathlib import Path

import aiofiles
from fastapi import APIRouter, UploadFile, Request, File
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY, HTTP_503_SERVICE_UNAVAILABLE
from fastapi import Response, HTTPException
from fastapi.responses import FileResponse

from backend.backend.video_handler.video_handler import VideoHandler, TaskStatus

router = APIRouter()
UPLOADED_FILE_PATH = ''


@router.post('/upload')
async def upload_video(request: Request, uploaded_file: UploadFile = File(...)):
    handler: VideoHandler = request.app.state.video_handler
    if uploaded_file.content_type != 'video/mp4':
        return Response(status_code=HTTP_422_UNPROCESSABLE_ENTITY)
    file_hash = hash(uploaded_file)
    async with aiofiles.open(f'{file_hash}.mp4', 'wb') as out_file:
        content = await uploaded_file.read()
        await out_file.write(content)
    file_path = Path(f'{file_hash}.mp4')
    print(file_path)
    print(file_path.exists())
    handler.start(file_path, file_hash)
    return {'hash': file_hash}


@router.get('/status/{file_hash}')
def get_status(file_hash: int, request: Request):
    handler: VideoHandler = request.app.state.video_handler
    status = handler.get_status(file_hash)
    return {'status': status}


@router.get('/download/{file_hash}')
def download_video(file_hash: int, request: Request):
    handler: VideoHandler = request.app.state.video_handler
    ret, task = handler.get_result(file_hash)
    if ret:
        if task.task_status == TaskStatus.FINISH and task.result_file_path and task.result_file_path.exists():
            path = task.result_file_path
            return FileResponse(path=path, filename=path.name, media_type='multipart/form-data')
        raise HTTPException(status_code=502, detail="An error occurred while processing the file")
    raise HTTPException(status_code=404, detail="Item not found")
