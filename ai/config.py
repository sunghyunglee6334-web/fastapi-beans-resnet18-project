import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
MODEL_PATH = os.getenv("MODEL_PATH", "microsoft/resnet-18")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://vision_user:vision_pass@127.0.0.1:5432/vision_db",
)
REDIS_CACHE_URL = os.getenv("REDIS_CACHE_URL", "redis://127.0.0.1:6379/1")
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/2")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/3")
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
MEDIA_DIR = BASE_DIR / "media" / "jobs"

# v2 장면 분석 (근접 / 원거리 자동 판별 · 잎별 분류) · 값은 Val 로 다시 맞춤 (tools/scene_preview.py)
SCENE_CLOSE_HULL = float(os.getenv("SCENE_CLOSE_HULL", "0.22"))    # 가장 큰 잎이 이 비율 이상 → 근접
SCENE_FAR_SIMILAR = int(os.getenv("SCENE_FAR_SIMILAR", "3"))       # 비슷한 크기 잎이 이 수 이상 → 원거리
SCENE_LEAF_SOLIDITY = float(os.getenv("SCENE_LEAF_SOLIDITY", "0.85"))  # 속이 꽉 찬 잎 모양 기준 (잎맥 조각 거르기)
SCENE_MAX_LEAVES = int(os.getenv("SCENE_MAX_LEAVES", "8"))         # 원거리에서 분류할 최대 잎 수
SCENE_CHECK_SCORE = float(os.getenv("SCENE_CHECK_SCORE", "0.5"))   # 잎 점수가 이보다 낮으면 CHECK
SCENE_MIN_SHARPNESS = float(os.getenv("SCENE_MIN_SHARPNESS", "20"))
SCENE_MIN_BRIGHT = float(os.getenv("SCENE_MIN_BRIGHT", "60"))
SCENE_MAX_BRIGHT = float(os.getenv("SCENE_MAX_BRIGHT", "215"))
SCENE_MIN_CONTRAST = float(os.getenv("SCENE_MIN_CONTRAST", "16"))
