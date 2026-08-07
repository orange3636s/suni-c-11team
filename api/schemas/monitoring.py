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
    groups: list[ConfigTreemapGroupSchema]
