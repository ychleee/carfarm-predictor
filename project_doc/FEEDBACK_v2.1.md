# CarFarm v2.1 — 프라이싱 매니저 피드백 반영

날짜: 2026-02-24
피드백 제공자: 프라이싱 매니저 (파일럿 테스트)

---

## 피드백 요약

### 1. 대상차량 입력 — 세대/트림 선택 개선 필요

| 이슈 | 상세 |
| --- | --- |
| 세대 구분 불가 | "17년~현재"만 표시, LF 소나타 뉴라이즈인지 알 수 없음 |
| 트림 세분화 부족 | 엔카처럼 연료/배기량 → 세부 트림 단계 필요 |
| 등록일 vs 연식 | 동일 연식이라도 등록일에 따라 가격 차이 (사소) |

### 2. 기준차량 추천 — 우선순위 잘못됨

| 이슈 | 상세 |
| --- | --- |
| 트림 불일치 | 트림이 다른 차량이 추천됨. 트림이 가장 중요 |
| 연식 차이 과다 | 10년 차이 차량 추천. 연식도 매우 중요 |
| 주행거리 과중 | 주행거리 중심 매칭. 주행거리는 보정 가능 |

### 3. 가격산출 — 보정 로직 문제

| 이슈 | 상세 |
| --- | --- |
| 옵션 보정 과다 | 일괄 50만원/개, 연식 감가 미반영. 11년식에 100만원 가산은 부적절 |
| 색상 매칭 오류 | "흰색" vs "white" 다른 색으로 인식, 40만원 보정 |
| 연식 보정 없음 | 2% 이상 차이에도 보정 없음 |
| 트림 차이 무경고 | 다른 트림 기준으로 산출 시 부적절한 가격 |

---

## 구현된 변경사항

### Phase 3: 룰엔진 수정

#### 3.1 색상 정규화 (`normalize_color`)

**파일**: `backend/app/services/rule_engine.py`

**문제**: 대상차량 색상이 "흰색"(한국어)인데, 기준차량 색상은 "white"(정규화)로 저장. 서로 다른 색으로 인식되어 잘못된 보정 발생.

**해결**: `normalize_color()` 함수 추가. 한국어 색상명을 정규화 그룹으로 매핑.
- 흰색/백색/화이트/아이보리/크림/진주 → white
- 검정/블랙 → black
- 은색/실버 → silver
- 회색/메탈/그레이/건메탈/티탄 → gray
- 기타 → other

`calculate_price()` 진입 시 target/reference 양쪽에 자동 적용.

#### 3.2 옵션 보정 연식 감가

**문제**: 선호옵션 보정이 연식 무관하게 일괄 50만원. 11년식 차량에 선루프 50만원은 과다.

**해결**: 색상 보정과 동일한 age_weight 적용.

| 차령 | 가중치 | 선루프 보정 예시 |
| --- | --- | --- |
| 3년 이하 | 1.0 | 50만원 |
| 7년 이하 | 0.7 | 35만원 |
| 10년 이하 | 0.5 | 25만원 |
| 10년 초과 | 0.3 | 15만원 |

#### 3.3 연식 보정 룰 (신규)

**파일**: `backend/rules/pricing_rules.yaml`, `rule_engine.py`

**문제**: 대상차량과 기준차량의 연식이 다를 때 보정이 없었음.

**해결**: 연당 2% 보정 룰 추가.
- 대상이 기준보다 오래되면 감가, 새로우면 가산
- 예: 기준 2020년, 대상 2018년 → 2년 × 2% × 기준가 = 감가

#### 3.4 트림 차이 경고 (신규)

**문제**: 트림이 다른 기준차량으로 산출 시 경고 없이 결과 제공.

**해결**: 트림이 다르면 경고 행 추가 + 신뢰도 자동 "낮음".
- 가격 보정은 하지 않음 (트림 가격차는 정형화 불가)
- AdjustmentTable에 amber 배경으로 경고 표시

---

### Phase 2: LLM 추천 우선순위 조정

#### 2.1 SYSTEM_PROMPT 하드 제약

**파일**: `backend/app/services/llm_recommender.py`

**변경**: 프롬프트 최상단에 절대 제약 섹션 추가.
1. 트림 동일 필수
2. 연식 ±3년 이내 권장
3. 주행거리는 보정 가능
4. 3건 미만이면 있는 만큼만 (무리하게 채우지 않기)

**우선순위**: 트림 > 연식 >> 옵션/출고가 >> 주행거리

#### 2.2 검색 가이드 강화

**_build_user_message** 에서 구체적 검색 가이드 추가:
- "먼저 trim으로 검색 → 결과 없으면 trim 빼고 재검색"
- year_min/year_max를 ±3년으로 설정

#### 2.3 displacement 필드

**TargetVehicle**에 `displacement` (배기량) 필드 추가. LLM에 variant 정보 전달.

---

### Phase 1: 세대/트림 UX 개선

#### 1.1 세대 표시명 (display_name)

**파일**: `backend/app/services/taxonomy_search.py`

**해결**: 트림명 공통 접두사에서 세대명 추출.

| 기존 | 개선 |
| --- | --- |
| 17년~현재 | 뉴 라이즈 (17년~현재) |
| 14년~현재 | LF (14년~현재) |
| 12년~16년 | 더 브릴리언트 (12년~16년) |
| 09년~12년 | YF (09년~12년) |
| 23년~현재 | 디 엣지 (23년~현재) |

알고리즘: 전체 트림의 70% 이상이 공유하는 첫 1~2 단어를 접두사로 추출.

#### 1.2 변형(Variant) 캐스케이드

**새 API**: `GET /api/variants/{maker}/{model}/{generation}`
- 세대 내 연료/배기량 조합 목록 반환
- `GET /api/trims/...?variant_key=가솔린|N/A|2.0` variant별 트림 필터

**Frontend**: 세대 선택 후 변형(연료/배기량) 버튼 그룹 표시
- variant가 1개면 자동 선택 (단계 건너뛰기)
- variant 선택 시 연료/배기량 자동 세팅

**캐스케이드 순서**:
```
제작사 → 모델 → 세대(display_name) → [변형(연료/배기량)] → 트림
```

---

## 변경 파일 목록

| 파일 | 변경 내용 |
| --- | --- |
| `backend/app/services/rule_engine.py` | normalize_color, 옵션감가, 연식보정, 트림경고 |
| `backend/rules/pricing_rules.yaml` | year_adjustment 설정 추가 |
| `backend/app/api/calculate.py` | reference에 trim/color 필드 추가 |
| `backend/app/services/llm_recommender.py` | SYSTEM_PROMPT 강화, 검색가이드 |
| `backend/app/api/recommend.py` | displacement 필드 추가 |
| `backend/app/services/taxonomy_search.py` | display_name, get_variants, variant_key 필터 |
| `backend/app/api/vehicle_info.py` | variants 엔드포인트 추가 |
| `frontend/src/types/index.ts` | VariantInfo, displacement, display_name |
| `frontend/src/api/client.ts` | getVariants 추가, getTrims variantKey |
| `frontend/src/components/VehicleForm.tsx` | variant 캐스케이드, display_name |
| `frontend/src/components/AdjustmentTable.tsx` | 트림 경고 스타일 |

---

## 인프라 수정

### Docker 데이터 경로 호환성

**파일**: `taxonomy_search.py`, `auction_db.py`

**문제**: `Path(__file__).parent.parent.parent.parent`가 Docker 내 `/`로 해석되어 Cloud Run에서 500 에러 발생.

**해결**: `_resolve_data_root()` 함수 추가.

1. `CARFARM_DATA_ROOT` 환경변수 우선
2. 로컬 개발 경로 (`__file__` 기준)
3. Docker 경로 (`/app/car_price_prediction/output/`)

---

## 데이터 품질 참고

- 택소노미 variant 연료/배기량 완전 데이터: 전체 29%, 인기 모델 ~50%
- N/A variant는 "전체" 옵션으로 처리 (graceful fallback)
- 향후: 외부 API(엔카/카라피스) 연동으로 트림 데이터 보강 검토

## 남은 과제

- 등록일 vs 연식 구분 (현재 미구현, 사소한 피드백)
- 옵션별 차등 가격 (파노라마 선루프 vs 일반 선루프 등)
- 렌터카 감가 차종별 차등 (현재 단일 5%)
- 데이터 최신성 가중치 (최근 데이터에 가산)
- 낙찰 데이터 일자 표시
