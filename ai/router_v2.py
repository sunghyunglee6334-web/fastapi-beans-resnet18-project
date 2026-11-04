from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.templating import Jinja2Templates

from .config import BASE_DIR, MAX_UPLOAD_BYTES
from .scene_service import classify_scene
from .services import InvalidImage

router = APIRouter(tags=["v2 장면 분석 (근접 · 원거리)"])
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@router.post("/api/v2/classify/")
def classify_v2(image: UploadFile = File(...), draw: bool = False):
    """사진을 근접 / 원거리로 자동 판별하고, 원거리면 잎마다 잘라 분류한다 (v1 API 는 그대로)."""
    image_bytes = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="5MB 이하 파일만 허용합니다.")
    try:
        return classify_scene(image_bytes, draw=draw)
    except InvalidImage as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/vision/scene/")
def scene_page(request: Request):
    return templates.TemplateResponse(request=request, name="scene.html")
