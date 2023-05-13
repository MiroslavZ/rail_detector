from fastapi import APIRouter, UploadFile, Request
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY, HTTP_503_SERVICE_UNAVAILABLE
from fastapi import Response

from backend.backend.video_handler.video_handler import VideoHandler

router = APIRouter()
UPLOADED_FILE_PATH = ''


@router.post('/upload')
async def upload_video(uploaded_file: UploadFile, request: Request):
    handler: VideoHandler = request.app.state.video_handler
    if uploaded_file.content_type != 'video/mp4':
        return Response(status_code=HTTP_422_UNPROCESSABLE_ENTITY)
    await handler.start(uploaded_file)
    return {'filename': uploaded_file.filename}


@router.get('/download')
async def download_video():
    return Response(status_code=HTTP_503_SERVICE_UNAVAILABLE)
    # if pathlib.Path(UPLOADED_FILE_PATH).exists():
    #     return FileResponse(path=UPLOADED_FILE_PATH, filename=UPLOADED_FILE_PATH, media_type='multipart/form-data')
