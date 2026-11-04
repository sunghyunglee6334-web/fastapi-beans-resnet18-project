"""v2 장면 분석 증빙 다시 만들기 · 실행 중인 서버(최종 모델)에 사진 3장을 보내 결과 저장
실행: ~/fastapi_resnet18 에서  python tools/v2_evidence.py   (docker compose 가 켜져 있어야 함)
출력: reports/evidence/v2_*_boxes.jpg (번호 동그라미 그림) · reports/data/v2_*.json (판정 · 잎별 점수)
"""
import base64, json, urllib.request, uuid
from pathlib import Path

FILES = {"v2_far_original": "robust/variants/00_original.jpg",
         "v2_close_test": "dataset/test/healthy/healthy_test.0.jpg",
         "v2_retake_blur": "robust/variants/05_blur_gaussian.jpg"}
URL = "http://127.0.0.1:8000/api/v2/classify/?draw=true"
Path("reports/data").mkdir(parents=True, exist_ok=True)

for name, path in FILES.items():
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{Path(path).name}\"\r\n"
            f"Content-Type: image/jpeg\r\n\r\n").encode() + Path(path).read_bytes() + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    d = json.load(urllib.request.urlopen(req))
    img = d.pop("image_jpg_base64", None)
    if img:
        Path(f"reports/evidence/{name}_boxes.jpg").write_bytes(base64.b64decode(img))
    Path(f"reports/data/{name}.json").write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    leaves = " · ".join(f'{x["no"]} {x["label"]} {x["score"]:.2f}' + ("?" if x["check"] else "") for x in d["leaves"])
    print(f'{name:16s} 판정 {d["decision"]:8s} 경로 {d["mode"]:6s} 잎 {len(d["leaves"])}장  {leaves}')
print(f'\n모델 {d["model_id"]}')
