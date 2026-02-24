"""
피드백 & 선택 이력 API
"""
import json
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path

router = APIRouter()

FEEDBACK_FILE = Path(__file__).parent.parent.parent / "feedback_data.jsonl"


class FeedbackRequest(BaseModel):
    """피드백 요청"""
    target_vehicle: dict
    selected_reference_id: str | None = None
    recommended_references: list[str] = []  # 추천된 기준차 ID 목록
    estimated_price: float | None = None
    actual_price: float | None = None
    feedback_type: str          # 적절/높음/낮음/wrong_recommendation
    comment: str | None = None


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    피드백 + 선택 이력 저장.
    향후 퓨샷 예시 풀 확장 및 벡터 서치 학습 데이터로 활용.
    """
    record = {
        "timestamp": datetime.now().isoformat(),
        **request.model_dump(),
    }
    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return {
        "status": "saved",
        "timestamp": record["timestamp"],
    }
