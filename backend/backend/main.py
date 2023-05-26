from fastapi import FastAPI
from backend.backend.routers.load_router import router
from backend.backend.video_handler.video_handler import VideoHandler

app = FastAPI()
app.include_router(router)


@app.on_event("startup")
async def task_create() -> None:
    app.state.video_handler = VideoHandler()


@app.on_event("shutdown")
async def task_stop() -> None:
    app.state.video_handler.stop()
