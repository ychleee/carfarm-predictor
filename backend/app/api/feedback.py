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
    """피드백 요청 — 전체 맥락 저장 (few-shot 학습 데이터용)"""
    # 대상차량
    target_vehicle: dict

    # 추천 맥락
    selected_reference_id: str | None = None
    recommended_references: list[str] = []          # 기준차 ID 목록 (하위호환)
    recommendations_detail: list[dict] = []         # 기준차 전체 상세 (vehicle_name, year, mileage, price, similarity_reason 등)
    llm_reasoning: str | None = None                # LLM 추론 전문
    tokens_used: dict | None = None                 # {input, output}
    tool_calls_count: int | None = None

    # 가격 산출 맥락 (산출했을 경우)
    calculation_result: dict | None = None          # CalculateResponse 전체 (adjustments, estimated_retail 등)

    # 피드백
    estimated_price: float | None = None
    actual_price: float | None = None
    feedback_type: str          # wrong_recommendation / price_high / price_low / price_ok
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
