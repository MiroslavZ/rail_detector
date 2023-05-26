from pathlib import Path

from fastapi import APIRouter, UploadFile, Request
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY, HTTP_503_SERVICE_UNAVAILABLE
from fastapi import Response, HTTPException
from fastapi.responses import FileResponse

from backend.backend.video_handler.video_handler import VideoHandler

router = APIRouter()
UPLOADED_FILE_PATH = ''


@router.post('/upload')
def upload_video(uploaded_file: UploadFile, request: Request):
    handler: VideoHandler = request.app.state.video_handler
    if uploaded_file.content_type != 'video/mp4':
        return Response(status_code=HTTP_422_UNPROCESSABLE_ENTITY)
    file_hash = handler.start(uploaded_file)
    return {'hash': file_hash}


@router.get('/download/{file_hash}')
def download_video(file_hash: int, request: Request):
    handler: VideoHandler = request.app.state.video_handler
    ret, path = handler.get_result(file_hash)
    if ret:
        if Path(path).exists():
            return FileResponse(path=path.resolve(), filename=path.name, media_type='multipart/form-data')
        raise HTTPException(status_code=502, detail="An error occurred while processing the file")
    raise HTTPException(status_code=404, detail="Item not found")
