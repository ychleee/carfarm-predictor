"""
LLM 기준차량 추천 서비스 end-to-end 테스트

사용법:
  ANTHROPIC_API_KEY=sk-xxx python test_llm_recommend.py
"""

import sys
import os
import json

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(__file__))

# .env.local에서 API 키 로드
env_path = os.path.join(os.path.dirname(__file__), '..', '.env.local')
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, val = line.split('=', 1)
                os.environ.setdefault(key.strip(), val.strip())

from app.services.llm_recommender import recommend_references

# ======================================================
# 테스트 케이스: 그랜져 IG 프리미엄 2022년식
# ======================================================

target_vehicle = {
    "maker": "현대",
    "model": "그랜져",
    "generation": "IG",
    "year": 2022,
    "mileage": 50000,  # 5만km
    "fuel": "가솔린",
    "trim": "프리미엄",
    "color": "흰색",
    "usage": "personal",
    "domestic": True,
}

print("=" * 60)
print("CarFarm v2 — LLM 기준차량 추천 테스트")
print("=" * 60)
print()
print("## 대상차량")
for k, v in target_vehicle.items():
    print(f"  - {k}: {v}")
print()
print("LLM 호출 중... (최대 10회 도구 호출)")
print()

try:
    result = recommend_references(target_vehicle, max_iterations=10)

    print("=" * 60)
    print("## 결과")
    print("=" * 60)
    print()

    print(f"### 토큰 사용량")
    print(f"  - Input: {result.total_input_tokens:,}")
    print(f"  - Output: {result.total_output_tokens:,}")
    print()

    print(f"### 도구 호출 ({len(result.tool_calls_log)}회)")
    for i, call in enumerate(result.tool_calls_log, 1):
        print(f"  [{i}] {call['tool']}({json.dumps(call['input'], ensure_ascii=False)[:120]})")
    print()

    print(f"### 추론 과정")
    print(f"  {result.reasoning}")
    print()

    print(f"### 추천 기준차량 ({len(result.recommendations)}건)")
    for i, rec in enumerate(result.recommendations, 1):
        print(f"  [{i}] ID: {rec.get('auction_id', 'N/A')}")
        print(f"      이유: {rec.get('reason', 'N/A')}")
        print()

    # 추천된 차량의 상세 정보 출력
    if result.recommendations:
        print("### 추천 차량 상세")
        from app.services.auction_db import get_vehicle_detail
        for i, rec in enumerate(result.recommendations, 1):
            aid = rec.get('auction_id')
            if aid:
                detail = get_vehicle_detail(aid)
                if detail:
                    print(f"  [{i}] {detail.get('차명', 'N/A')}")
                    print(f"      연식: {detail.get('연식')}, 주행거리: {detail.get('주행거리')}km")
                    print(f"      낙찰가: {detail.get('낙찰가')}만원, 색상: {detail.get('색상')}")
                    print(f"      옵션: {detail.get('옵션', 'N/A')[:80]}")
                    print(f"      교환: {detail.get('exchange_count', 0)}, 판금: {detail.get('bodywork_count', 0)}")
                    print()

except Exception as e:
    print(f"오류 발생: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
