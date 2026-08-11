from __future__ import annotations

from pydantic import BaseModel


class ConfigTreemapGroupSchema(BaseModel):
    # 원문과 서버 공통 파서가 분해한 계층을 함께 반환한다.
    config: str
    model: str
    equipment: str
    chamber: str
    n: int
    mean: float
    median: float
    p5: float
    p95: float


class ConfigTreemapResponse(BaseModel):
    dataset_id: str
    step: int
    target: str
    target_label: str
    deprecated_target: bool = False
    # 이 스텝에서 집계 가능한 선택 target의 전체 평균 -- 트리맵 색 스케일의
    # 중앙값으로 쓰인다(스텝을 바꿔도 중앙이 흔들리면 패널 간 비교가
    # 무의미해진다).
    overall_mean: float
    # 참고 통계: 이 스텝의 Config가 선택 target 차이를 설명하는지
    # (더미회귀 Adjusted R²의 ANOVA F검정 + BH-FDR -- 가족은 이
    # 데이터셋의 모든 스텝 Config 검정 전체). 색상 자체는 선택 target의
    # 평균 불량률을 나타내며 이 통계로 가리지 않는다.
    significant: bool
    groups: list[ConfigTreemapGroupSchema]
    empty_reason: str | None = None
    target_provenance: dict | None = None
    # "모델 분석" 파이프라인이 마지막으로 저장한 스냅샷의
    # analysis_id -- 모니터링/원인분석/수율예측과 같은 값이면 네 화면이
    # 같은 분석 회차를 보고 있다는 뜻이다. 스냅샷이 없으면 null.
    analysis_id: str | None = None
    # 20,000행 초과 데이터셋은 로트 단위 표본으로 집계했다는
    # 고지 -- None이면 전량 기준.
    sample_info: dict | None = None
