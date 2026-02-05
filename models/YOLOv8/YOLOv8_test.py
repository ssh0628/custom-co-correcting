from ultralytics import YOLO
import json
from pathlib import Path

RUN_DIR = Path("runs/classify/YOLOv8_Cls/Yolos1")
BEST = RUN_DIR / "weights/best.pt"

def to_jsonable_results(r):
    """
    Ultralytics 버전차 흡수:
    - r.results_dict 가 메서드일 수도, dict일 수도 있음
    - 없으면 r 자체 문자열로 대체
    """
    if hasattr(r, "results_dict"):
        rd = getattr(r, "results_dict")
        if callable(rd):
            return rd()
        return rd  # dict인 경우
    return {"repr": str(r)}

def eval_test():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(BEST))
    r = model.val(
        data="/root/project/dataset/dataset_cropped",
        split="test",          # 중요
        imgsz=224,
        batch=256,
        device="cuda:0",
        workers=4,             # <= 평가에서는 0~4 권장 (안정성)
        verbose=False
    )

    out = {
        "results": to_jsonable_results(r),
        "save_dir": str(getattr(r, "save_dir", RUN_DIR))
    }

    (RUN_DIR / "test_metrics.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print("[OK] saved:", RUN_DIR / "test_metrics.json")
    print("[OK] save_dir:", out["save_dir"])

if __name__ == "__main__":
    eval_test()