"""
피드백 & 선택 이력 API
"""
from fastapi import APIRouter
from pydantic import BaseModel
from datetime import datetime

router = APIRouter()


class FeedbackRequest(BaseModel):
    """피드백 요청"""
    target_vehicle: dict
    selected_reference_id: str | None = None
    recommended_references: list[str] = []  # 추천된 기준차 ID 목록
    estimated_price: float | None = None
    actual_price: float | None = None
    feedback_type: str          # 적절/높음/낮음
    comment: str | None = None


@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    피드백 + 선택 이력 저장.
    향후 퓨샷 예시 풀 확장 및 벡터 서치 학습 데이터로 활용.
    """
    # TODO: JSONL 파일 또는 DB에 저장
    return {
        "status": "saved",
        "timestamp": datetime.now().isoformat(),
    }
