from __future__ import annotations

from pydantic import BaseModel


class ConfigTreemapGroupSchema(BaseModel):
    # 원문 그대로("Step7_Model2_EQC_CH3") -- Model/EQ/Chamber 분해는
    # src/config_parser.py와 같은 원칙으로 서버에서 하지 않는다. 프론트가
    # 표시용으로 쪼갠다.
    config: str
    n: int
    mean: float
    median: float
    p5: float
    p95: float


class ConfigTreemapResponse(BaseModel):
    dataset_id: str
    step: int
    # 이 스텝이 아니라 데이터셋 전체의 Y 평균 -- 트리맵 색 스케일의 고정
    # 중앙값으로 쓰인다(스텝을 바꿔도 중앙이 흔들리면 패널 간 비교가
    # 무의미해진다).
    overall_mean: float
    # C-3: 이 스텝의 Config(장비 구성)가 최종 수율 차이를 설명하는지
    # (ANOVA eps2 + BH-FDR, 범주형 히트맵과 같은 규칙 -- 가족은 이
    # 데이터셋의 모든 스텝 Config 검정 전체) -- false면 프론트는 타일을
    # 전량 중립색으로 렌더한다. 검출한계 이하 차이를 색으로 강조하지
    # 않기 위함이다.
    significant: bool
    groups: list[ConfigTreemapGroupSchema]
