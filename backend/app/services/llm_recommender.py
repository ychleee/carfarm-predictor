"""
CarFarm v2 — LLM 기준차량 추천 서비스

프라이싱 매니저의 실제 감정 프로세스를 LLM 리즈닝 모델로 구현.

프로세스:
  1. 대상차량 정보 확인
  2. DB에서 후보 검색 (모델+트림 필수 → 연식 → 옵션 → 주행거리)
  3. 기준차량 3건 선정 + 선택 이유 설명
  4. 옵션 가치를 삼각측량으로 교차검증

사용하는 도구:
  - search_auction_db: 낙찰 DB 검색
  - get_vehicle_detail: 차량 상세 정보
  - get_price_stats: 모델별 시세 통계
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import anthropic

from app.services.firestore_db import (
    search_auction_db,
    get_vehicle_detail,
    get_price_stats,
)


# =========================================================================
# 시스템 프롬프트 — 프라이싱 매니저의 추론 프로세스 인코딩
# =========================================================================

SYSTEM_PROMPT = """당신은 자동차 경매 전문 프라이싱 매니저입니다.
대상차량에 대해 **기준차량 최대 3건**을 추천하고, 각각의 선택 이유를 설명하는 것이 당신의 역할입니다.

## ⚠️ 절대 제약 (반드시 준수)

1. **트림이 동일하거나 거의 동일해야 합니다** (필수). 트림이 다른 차량은 절대 추천하지 마세요.
   - 예: 대상이 "뉴 라이즈 모던"이면 기준도 "뉴 라이즈 모던" 또는 매우 유사한 트림
   - 트림이 맞지 않으면 가격 비교 자체가 무의미합니다
2. **연식 차이 ±3년 이내** 권장. 4년 이상 차이 나는 차량은 가급적 추천하지 마세요.
3. **주행거리는 보정 가능**하므로 다소 차이나도 괜찮습니다 (룰 엔진이 자동 보정).
4. 위 조건을 만족하는 차량이 **3건 미만이면 있는 만큼만 추천**하세요. 무리하게 3건을 채우지 마세요.
5. 조건에 맞는 차량이 **0건이면** recommendations를 빈 배열로 반환하고, reasoning에 왜 없는지 설명하세요.

## 추론 프로세스 (반드시 이 순서로 진행)

### 1단계: 트림+연식 고정으로 후보 확보
- **먼저 trim 파라미터로 검색**하세요 (가장 중요)
- 연식은 ±2년 이내로 year_min/year_max 설정
- 결과가 적으면 trim 조건만 조금 완화 (예: 부분 매칭)
- 결과가 0이면 trim 없이 검색하되, 결과에서 트림이 유사한 것만 선별

### 2단계: 후보 중 키로수/옵션 다양성 확보
- 주행거리가 비슷한 차량 우선
- 출고가가 비슷한 것 = 옵션이 비슷할 가능성 높음
- 완전히 같은 것이 없으면, 키로수가 다른 여러 대를 찾아서 보정으로 비교

### 3단계: 주행거리 보정 — 반복 수렴
공식으로 1차 보정한 뒤, 여러 기준차의 추정이 수렴할 때까지 감가율을 조정합니다.

예시:
- 대상: 15만km. 기준차A(18만km, 낙찰 1500만), 기준차B(14만km, 낙찰 1700만)
- 1차 보정: A→1680만, B→1640만 → 갭 40만
- 감가율 조정하며 갭 축소 → 수렴하면 그게 정답

이것이 핵심입니다: **공식은 출발점이지 정답이 아닙니다.** 여러 기준차를 비교하며 실제 감가율을 추정합니다.

### 4단계: 이상치 조사
보정해도 가격이 안 맞는 차량이 있으면 **버리지 말고 원인을 조사**합니다:
- get_vehicle_detail로 상세 정보 확인
- 왜 안 맞는지 추론: 수출차? 사고이력(교환/판금)? 비싼 옵션(선루프)? 색상?
- 원인이 설명되면 보정 후 활용, 설명 안 되면 제외

### 5단계: 출고가 비율 (있으면 활용)
출고가 정보가 있으면 빠른 추론이 가능합니다:
- 낙찰가/출고가 비율로 잔존가치 파악
- 데이터 적은 희귀 차종에서 특히 유용

## 기준차 선택 우선순위

**중요도 순서 (절대적):**
1. **트림** — 동일 필수. 트림이 다르면 가격 차이가 수백만원일 수 있음
2. **연식** — ±2년 이내 필수. 연식 차이가 크면 보정이 부정확
3. **옵션/출고가** — 같으면 최고, 다르면 교차검증용으로 활용
4. **주행거리** — 보정 가능하므로 우선순위 가장 낮음

### 좋은 구성
- 가능하면 키로수나 옵션이 **조금씩 다른** 차량을 섞기
- 이를 통해 감가율과 옵션 가치를 교차검증 가능
- 3건의 보정 결과가 수렴하면 → 추정 신뢰도 높음

## 보정 룰 요약 (룰 엔진 자동 적용)

- 주행거리: 2~3년 2%/만km, 4~6년 1.5%, 7~9년 1%, 10년+ 1%/10만원 (20만km 천장)
- 연식 차이: 연당 2%
- 교환(X): 1회당 3%
- 판금(W): 연식/가격대별 1~2%
- 색상: 선호도별 20만원
- 선호옵션: 1개당 50만원 × 연식가중치

## 도구 사용 가이드

### search_auction_db
- **반드시 trim 파라미터를 먼저 사용**하세요
- maker + model + trim + year_min/year_max로 검색 시작
- 결과가 0이면 trim을 빼고 재검색, 수동으로 트림 유사성 판단

### get_price_stats
- 시세 맥락 파악용. 평균/중앙/최고/최저가 확인

### get_vehicle_detail
- 이상치 조사 시 사용. 옵션, 사고이력, 수출 여부 등 확인

## 응답 형식

반드시 아래 JSON 형식으로 응답하세요:
```json
{
  "recommendations": [
    {
      "auction_id": "차량 ID",
      "reason": "이 차량을 선택한 이유 (한국어, 2~3문장). 대상차량과 무엇이 같고 무엇이 다른지, 어떤 보정이 필요한지 설명."
    }
  ],
  "reasoning": "전체 추론 과정 (한국어, 3~5문장). 어떻게 검색했고, 왜 이 N건을 골랐고, 감가율이나 가격 추정이 수렴하는지 설명. 3건 미만이면 왜 부족한지 설명."
}
```
"""


# =========================================================================
# Tool 정의 (Claude API tool_use 형식)
# =========================================================================

TOOLS = [
    {
        "name": "search_auction_db",
        "description": (
            "낙찰 DB에서 조건에 맞는 차량을 검색합니다. "
            "결과에 낙찰가, 주행거리, 옵션, 사고이력, 색상 등이 포함됩니다. "
            "maker와 model은 필수입니다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "maker": {
                    "type": "string",
                    "description": "제작사 (예: 현대, 기아, 제네시스)"
                },
                "model": {
                    "type": "string",
                    "description": "모델명 (예: 그랜저, K5, 쏘나타)"
                },
                "generation": {
                    "type": "string",
                    "description": "세대 코드 (예: GN7, DN8). 없으면 전체 세대 검색."
                },
                "year_min": {
                    "type": "integer",
                    "description": "최소 연식 (예: 2022)"
                },
                "year_max": {
                    "type": "integer",
                    "description": "최대 연식 (예: 2024)"
                },
                "fuel": {
                    "type": "string",
                    "description": "연료 (가솔린, 디젤, 하이브리드, LPG 등)"
                },
                "trim": {
                    "type": "string",
                    "description": "트림명 (예: 프리미엄, 익스클루시브)"
                },
                "mileage_max": {
                    "type": "integer",
                    "description": "최대 주행거리 (km)"
                },
                "usage": {
                    "type": "string",
                    "description": "차량경력: personal(자가용) 또는 rental(렌터카)"
                },
                "limit": {
                    "type": "integer",
                    "description": "최대 결과 수 (기본 30)"
                },
                "sort_by": {
                    "type": "string",
                    "description": "정렬 기준: 날짜 또는 가격"
                }
            },
            "required": ["maker", "model"]
        }
    },
    {
        "name": "get_vehicle_detail",
        "description": (
            "특정 낙찰 차량의 전체 상세 정보를 조회합니다. "
            "옵션, 사고이력, 평가점 등 모든 정보가 포함됩니다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "auction_id": {
                    "type": "string",
                    "description": "차량 ID (search_auction_db 결과의 auction_id)"
                }
            },
            "required": ["auction_id"]
        }
    },
    {
        "name": "get_price_stats",
        "description": (
            "특정 모델의 최근 N개월 시세 통계를 조회합니다. "
            "평균, 중앙값, 최고가, 최저가, 표준편차를 반환합니다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "maker": {
                    "type": "string",
                    "description": "제작사"
                },
                "model": {
                    "type": "string",
                    "description": "모델명"
                },
                "generation": {
                    "type": "string",
                    "description": "세대 코드"
                },
                "year": {
                    "type": "integer",
                    "description": "특정 연식으로 한정"
                },
                "months": {
                    "type": "integer",
                    "description": "최근 N개월 (기본 3)"
                }
            },
            "required": ["maker", "model"]
        }
    }
]


# =========================================================================
# Tool 실행 함수
# =========================================================================

def _execute_tool(name: str, input_data: dict) -> str:
    """LLM이 요청한 도구를 실행하고 결과를 JSON 문자열로 반환"""
    if name == "search_auction_db":
        results = search_auction_db(**input_data)
        return json.dumps(results, ensure_ascii=False, default=str)
    elif name == "get_vehicle_detail":
        result = get_vehicle_detail(**input_data)
        if result is None:
            return json.dumps({"error": "해당 ID의 차량을 찾을 수 없습니다."}, ensure_ascii=False)
        return json.dumps(result, ensure_ascii=False, default=str)
    elif name == "get_price_stats":
        result = get_price_stats(**input_data)
        return json.dumps(result, ensure_ascii=False, default=str)
    else:
        return json.dumps({"error": f"알 수 없는 도구: {name}"}, ensure_ascii=False)


# =========================================================================
# 추천 결과 모델
# =========================================================================

@dataclass
class RecommendationResult:
    """LLM 추천 결과"""
    recommendations: list[dict] = field(default_factory=list)
    reasoning: str = ""
    tool_calls_log: list[dict] = field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0


# =========================================================================
# 메인 추천 함수
# =========================================================================

def recommend_references(
    target: dict,
    exclude_ids: list[str] | None = None,
    max_iterations: int = 10,
    model: str = "claude-sonnet-4-20250514",
) -> RecommendationResult:
    """
    LLM 리즈닝 모델을 호출하여 기준차량 3건을 추천.

    Args:
        target: 대상차량 정보 dict
            - maker, model, year, mileage 필수
            - generation, trim, fuel, color, usage, options 등 선택
        max_iterations: 최대 도구 호출 반복 횟수
        model: 사용할 Claude 모델

    Returns:
        RecommendationResult: 추천 결과 + 추론 과정
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY 환경 변수가 설정되지 않았습니다.")

    client = anthropic.Anthropic(api_key=api_key)
    result = RecommendationResult()

    # 사용자 메시지 구성
    user_message = _build_user_message(target, exclude_ids=exclude_ids)

    messages = [{"role": "user", "content": user_message}]

    # Agentic loop — LLM이 도구를 호출하면 실행하고 결과를 돌려줌
    for iteration in range(max_iterations):
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        result.total_input_tokens += response.usage.input_tokens
        result.total_output_tokens += response.usage.output_tokens

        # 응답 처리
        if response.stop_reason == "end_turn":
            # LLM이 최종 응답을 줌
            final_text = _extract_text(response)
            parsed = _parse_llm_response(final_text)
            result.recommendations = parsed.get("recommendations", [])
            result.reasoning = parsed.get("reasoning", final_text)
            break

        elif response.stop_reason == "tool_use":
            # 도구 호출 처리
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_name = block.name
                    tool_input = block.input

                    # 도구 실행 로그
                    result.tool_calls_log.append({
                        "iteration": iteration,
                        "tool": tool_name,
                        "input": tool_input,
                    })

                    # 도구 실행
                    tool_output = _execute_tool(tool_name, tool_input)

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": tool_output,
                    })

            # 대화에 어시스턴트 응답 + 도구 결과 추가
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

        else:
            # 예상치 못한 종료
            result.reasoning = f"예상치 못한 종료: {response.stop_reason}"
            break

    # max_iterations 도달 시 최종 답변 요청
    if not result.recommendations:
        messages.append({
            "role": "user",
            "content": "도구 호출 한도에 도달했습니다. 지금까지의 검색 결과를 바탕으로 최종 추천 JSON을 반환해주세요.",
        })
        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        result.total_input_tokens += response.usage.input_tokens
        result.total_output_tokens += response.usage.output_tokens
        final_text = _extract_text(response)
        parsed = _parse_llm_response(final_text)
        result.recommendations = parsed.get("recommendations", [])
        result.reasoning = parsed.get("reasoning", final_text)

    return result


def _build_user_message(target: dict, exclude_ids: list[str] | None = None) -> str:
    """대상차량 정보를 사용자 메시지로 구성"""
    lines = ["아래 대상차량에 대한 기준차량을 추천해주세요.", ""]
    lines.append("## 대상차량 정보")

    field_names = {
        "maker": "제작사",
        "model": "모델명",
        "generation": "세대",
        "year": "연식",
        "mileage": "주행거리(km)",
        "fuel": "연료",
        "displacement": "배기량",
        "drive": "구동방식",
        "trim": "트림",
        "color": "색상",
        "usage": "차량경력",
        "options": "옵션",
        "exchange_count": "교환 부위 수",
        "bodywork_count": "판금 부위 수",
        "domestic": "내수 여부",
    }

    for key, label in field_names.items():
        value = target.get(key)
        if value is not None and value != "" and value != []:
            if isinstance(value, list):
                value = ", ".join(str(v) for v in value)
            elif isinstance(value, bool):
                value = "내수" if value else "수출"
            lines.append(f"- {label}: {value}")

    # 검색 가이드
    trim = target.get("trim", "")
    year = target.get("year", 0)
    lines.append("")
    lines.append("## 검색 가이드")
    if trim:
        lines.append(f"1. **먼저** maker={target.get('maker')} + model={target.get('model')} + trim=\"{trim}\" + year_min={year-3} + year_max={year+3} 로 검색하세요.")
        lines.append("2. 결과가 0이면 trim을 빼고 재검색한 뒤, 트림이 유사한 것만 선별하세요.")
    else:
        lines.append(f"1. maker={target.get('maker')} + model={target.get('model')} + year_min={year-3} + year_max={year+3} 로 검색하세요.")
    lines.append("3. 트림과 연식이 맞는 차량 중에서 주행거리/옵션이 조금씩 다른 것을 섞어주세요.")

    if exclude_ids:
        lines.append("")
        lines.append("## 제외할 차량")
        lines.append("아래 auction_id 차량은 이미 추천되어 있으므로 **반드시 제외**하세요:")
        for aid in exclude_ids:
            lines.append(f"- {aid}")

    return "\n".join(lines)


def _extract_text(response) -> str:
    """응답에서 텍스트 블록 추출"""
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return ""


def _parse_llm_response(text: str) -> dict:
    """LLM 응답에서 JSON 파싱"""
    # JSON 블록 추출 시도
    import re
    json_match = re.search(r'```json\s*(.*?)\s*```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 직접 JSON 파싱 시도
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # JSON이 아닌 경우 텍스트 그대로 반환
    return {"reasoning": text, "recommendations": []}
