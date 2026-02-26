"""
기준차량 추천 API — LLM 리즈닝 모델 호출

프라이싱 매니저의 추론 프로세스를 LLM으로 구현:
  1. 모델+트림 동일한 후보 검색
  2. 연식 근접한 차량 우선
  3. 옵션 구성이 조금씩 다른 3건 추천 (삼각측량)
  4. 각 기준차의 선택 이유를 자연어로 설명
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.schemas.vehicle import TargetVehicleSchema
from app.services.llm_recommender import recommend_references
from app.services.auction_db import get_vehicle_detail

router = APIRouter()


class ReferenceVehicle(BaseModel):
    """추천된 기준차량"""
    auction_id: str
    vehicle_name: str | None = None
    year: int | None = None
    mileage: int | None = None
    auction_price: float | None = None
    auction_date: str | None = None
    color: str | None = None
    options: str | None = None
    usage_type: str | None = None
    is_export: bool = False       # 내수(False) / 수출(True)
    trim: str | None = None       # 트림
    similarity_reason: str        # LLM이 설명하는 선택 이유


class RecommendResponse(BaseModel):
    """추천 응답"""
    target: TargetVehicleSchema
    recommendations: list[ReferenceVehicle]
    reasoning: str                # LLM 전체 추론 과정
    tool_calls_count: int = 0     # 도구 호출 횟수
    tokens_used: dict = {}        # 토큰 사용량


@router.post("/recommend", response_model=RecommendResponse)
async def recommend_reference_vehicles(target: TargetVehicleSchema):
    """
    대상차량에 대한 기준차량 3건 추천.
    LLM 리즈닝 모델이 DB 검색 + 도메인 지식으로 추론.
    """
    try:
        # LLM 추천 실행
        exclude_ids = target.exclude_auction_ids
        target_dict = target.model_dump(exclude={"exclude_auction_ids"})
        result = recommend_references(target_dict, exclude_ids=exclude_ids)

        # 추천된 차량의 상세 정보 조회
        recommendations = []
        for rec in result.recommendations:
            auction_id = rec.get("auction_id", "")
            reason = rec.get("reason", "")

            # DB에서 상세 정보 조회
            detail = get_vehicle_detail(auction_id) if auction_id else None

            ref = ReferenceVehicle(
                auction_id=auction_id,
                vehicle_name=detail.get("차명", "") if detail else None,
                year=detail.get("연식") if detail else None,
                mileage=detail.get("주행거리") if detail else None,
                auction_price=detail.get("낙찰가") if detail else None,
                auction_date=detail.get("개최일") if detail else None,
                color=detail.get("색상") if detail else None,
                options=detail.get("옵션") if detail else None,
                usage_type=detail.get("usage_type") if detail else None,
                is_export=bool(detail.get("is_export", 0)) if detail else False,
                trim=detail.get("trim") if detail else None,
                similarity_reason=reason,
            )
            recommendations.append(ref)

        return RecommendResponse(
            target=target,
            recommendations=recommendations,
            reasoning=result.reasoning,
            tool_calls_count=len(result.tool_calls_log),
            tokens_used={
                "input": result.total_input_tokens,
                "output": result.total_output_tokens,
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM 추천 엔진 오류: {type(e).__name__}: {str(e)}"
        )
