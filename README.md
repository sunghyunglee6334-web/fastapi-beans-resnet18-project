# FastAPI Beans ResNet18 동기 · 비동기 분류 서비스

콩잎 이미지를 angular_leaf_spot · bean_rust · healthy 3클래스로 분류하는 ResNet18 Fine-tuning 모델을 FastAPI 동기 API 와 Celery 비동기 작업 큐로 서빙합니다.

| 항목 | 결과 |
|---|---|
| 모델 | microsoft/resnet-18 → Head-only Fine-tuning → `artifacts/beans_resnet18` |
| Best Validation Accuracy | Head-only 0.7669 (서비스 모델) · 선택 실험 Full Fine-tuning 0.9699 · Train 1,034 · Validation 133 · Test 미측정 |
| 실행 | Docker Compose: web (FastAPI) · worker (Celery) · redis · postgres |
| 성능 중앙값 | 동기 21.1 ms · 비동기 접수 10.0 ms · 캐시 HIT 1.8 ms |

## 1. 환경

- Windows + WSL2 Ubuntu · Docker Desktop (Settings → Resources → WSL integration → Ubuntu 켜기)
- 학습: WSL `.venv` (Python 3.14) · 서빙: Docker 이미지 python:3.11-slim (CPU torch)
- 포트: 8000 (web) · 6379 (redis) · 5432 (postgres) · 127.0.0.1 에서만 접근

## 2. 데이터 준비

1. https://huggingface.co/datasets/AI-Lab-Makerere/beans/tree/main/data 에서 `train.zip` · `validation.zip` · `test.zip` 다운로드
2. 압축 해제 후 원본은 보존하고 아래처럼 복사 (**validation → val**)

```
dataset/
├── train/{angular_leaf_spot,bean_rust,healthy}/   # 1,035장
├── val/{angular_leaf_spot,bean_rust,healthy}/     # 133장 (원본 validation)
└── test/{angular_leaf_spot,bean_rust,healthy}/    # 128장 (학습 미사용)
```

## 3. 모델 준비

```bash
python -m venv .venv && source .venv/bin/activate
python -m pip install -r requirements.txt
python -m training.finetune --data dataset --output artifacts/beans_resnet18 --epochs 3 --batch-size 8 --freeze-backbone
```

- 결과: `artifacts/beans_resnet18/` (config.json · model.safetensors · preprocessor_config.json · training_report.json)
- `.env` : `cp .env.example .env` → `MODEL_PATH=artifacts/beans_resnet18`

## 4. 구동 순서

```bash
docker compose up -d --build
docker compose ps                                   # 4개 Up (healthy)
docker compose exec redis redis-cli ping            # PONG
docker compose exec postgres pg_isready -U vision_user -d vision_db
```

| 주소 | 용도 |
|---|---|
| http://127.0.0.1:8000/health | 서버 확인 |
| http://127.0.0.1:8000/vision/sync/ | 동기 분류 화면 |
| http://127.0.0.1:8000/vision/async/ | 비동기 분류 화면 |
| http://127.0.0.1:8000/docs | Swagger API 문서 |

- 모델 · `MODEL_PATH` 변경 시: `docker compose up -d --force-recreate web worker`
- 중지: `docker compose stop` · `docker compose down -v` 는 DB · Redis 볼륨 삭제라 사용하지 않음

## 5. 확인할 응답

- `/health` → `{"status":"ok"}`
- 동기 POST → 200 · `top1` · `all` · `inference_ms` · `cache_hit`
- 비동기 POST → **202** · `job_id` · `check_url` → GET 상태 조회 `status: done`
- PostgreSQL `image_jobs` 에 비동기 작업 기록 (동기 요청은 기록하지 않음)
- 모의 테스트: `docker compose exec web python -m pytest -q tests/test_api.py` (4 passed)
- 성능 측정: `python perf.py` → `reports/data/timings.csv` · 그래프 `python perf_chart.py` → `reports/figures/perf_compare.png`
- 선택 실험 Full Fine-tuning: `python -m training.finetune --data dataset --output artifacts/beans_resnet18_full --epochs 3 --batch-size 8`

## 6. 결과물 위치

| 파일 | 내용 |
|---|---|
| `reports/01_FineTuning_결과보고서.md` | 데이터 · 학습 설정 · 결과 · 모델 파일 |
| `reports/02_시스템_아키텍처.md` | 구성도 · 역할 · 실행환경 |
| `reports/03_API_명세서.md` | API 표 · 요청 · 응답 · 상태코드 |
| `reports/04_DB_구조_작업이력.md` | image_jobs 구조 · 실제 이력 |
| `reports/05_AI버튼_처리흐름.md` | 동기 · 비동기 시퀀스 · 코드 연결 |
| `reports/06_성능비교_보고서.md` | 측정 조건 · 원본 · 해석 |
| `reports/07_기능테스트_오류분석.md` | 테스트표 · 오류 이력 · 오분류 |
| `reports/08_최종기술보고서.md` | 종합 · 설계 이유 · 개선 제안 · 한계 |
| `reports/data/timings.csv` | 성능 원본 |
| `reports/figures/perf_compare.png` | 성능 비교 그래프 |
| `reports/evidence/` | 실행 캡처 · 목록 `00_INDEX.md` |
| `artifacts/beans_resnet18/training_report.json` | 학습 기록 |

## 7. 제출 · 보안

- `.env` · 비밀번호는 Git 에 올리지 않음 (`.gitignore`)
- `dataset/` · `artifacts/` 가중치 · `media/` 는 용량이 커서 Git 제외 → 위 2 · 3장 절차로 다시 만들거나 허용된 경로로 별도 전달
