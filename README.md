# SUNI 11팀 — 반도체 공정 원인 분석 & 사전 알람

반도체 제조 공정 데이터에서 **Y1~Y5 불량률과 통계적으로 연관된 공정 인자를 특정하고, 인자별 SPC 관리한계로 개별 이탈을 표시하며, 권장 구간으로 개선 방향을 제시하는** FastAPI + Next.js 애플리케이션입니다. 이와 별개로, wafer 최종 수율(Y = 100 − Y1~Y5의 합) 오름차순 순위로 검토 우선순위를 제시하는 수율 예측 파이프라인을 포함합니다(신뢰도 n/5로 근거 강도 표기 — 목표 수율 절대 임계·민감도 조절·5단계 등급 판정은 R²가 낮아 절대값을 신뢰할 수 없다는 검증 결과에 따라 폐기됨). Upstage Solar LLM 연동으로 이 분석 결과를 자연어 보고서·대화형 답변으로 변환하는 SUNI 챗봇을 포함합니다.

**이 프로젝트의 산출물은 수율 예측 정확도가 아니라 "어떤 step의 어떤 인자가 어떤 불량과 관계있는가"의 통계적 특정입니다.** Y1~Y5를 학습하는 회귀 모델이 포함되어 있지만, 이는 스크리닝 결과의 구현 정확성을 확인하는 회귀 테스트 기준값이지 제품 기능(수율 예측 서비스)이 아닙니다.

## 데이터

내장 데이터셋 2종(`data/bundled/`, `src/runtime/datasets.py`의 `BUNDLED_DATASET_FILES`).

| 파일 | 행 × 열 | LOT | 비고 |
|---|---|---|---|
| train.CSV | 10,000 × 102 | L001~L400 | 기본 학습 데이터. `Lot_ID` 컬럼 있음 |
| test_remove_y.CSV | 1,000 × 102 | L401~L440 | 평가용(데이터셋 선택 UI에는 "test"로 표시). `Y`·`Y1`~`Y5` 컬럼이 있지만 전량 결측입니다 — 화면에 보이는 값은 그 시점의 승인된 챔피언 모델 예측값으로 채운(하이드레이션) 것입니다(`src/analysis/target_hydration.py`). `Lot_ID` 컬럼 있음 |

**본문 곳곳의 "train.CSV"·"test.CSV" 수치(§인자 스크리닝, §SPC 관리한계, §권장 구간, §웨이퍼 수율 예측 알람)는 위 내장본이 아니라 `data/raw/train.CSV`·`data/raw/test.CSV`(실측 Y가 전량 채워진 비공개 원본, 저장소에는 포함되지 않고 골든 회귀 테스트 실행 시에만 로컬에 둡니다 — `tests/test_*_golden.py`의 `skipif`)를 가리킵니다.** 모니터링 홈 블록③의 계측 편향·분산 분해처럼 "지금 서버가 실제로 서빙 중인 eval 데이터셋" 기준 수치는 내장 `test_remove_y.CSV`를 씁니다 — Y가 전량 결측이라는 사실이 그 블록의 해석에 직접 영향을 줍니다(아래 「화면 구성」 참고).

- 컬럼 규칙: `Step{n}_R{m}`(센서), `Step{n}_D{m}`(결함수), `Step{n}_Config` 또는 `Step{n}_EQ`(장비 구성), 타깃 `Y1`~`Y5`(+ `Y6`~`Y10`, 최종 `Y`)
- `_Config`와 `_EQ`는 같은 성격의 컬럼이며 컬럼명 접미사만 다릅니다. `_Config`/`_EQ` 뒤에 숫자가 더 붙는 형태(`_EQ1`, `_EQ2`…)는 인식하지 않습니다
- `Lot_ID`가 없는 데이터셋은 `Lot_Wafer_ID`(예: `L001W01`)에서 LOT과 wafer 순번을 파싱합니다(`src/dataset_normalization.py`)
- 결측률(train.CSV 기준): R 인자 하나당 평균 약 85% 결측, D 인자 약 95% 결측, Config 0%(전수 계측)
- **결측은 랜덤이 아니라 계측 샘플링 구조입니다.** R 인자의 결측률 85%는 LOT 25장 중 그 인자가 계측된 wafer가 평균 4장 미만이라는 뜻입니다
- 업로드로 데이터셋 추가 가능(`POST /api/datasets`), 상한 20MB(`MAX_UPLOAD_SIZE_MB`)

## 설계 원칙

각 항목에 근거 수치를 병기합니다.

- **결측을 채우지 않는다.** NaN을 유지하고 `{feature}_miss` 이진 태그를 별도 피처로 둡니다. 중앙값으로 대체하면 상관관계가 무너집니다 — `Step1_D1`–`Y3` Spearman ρ가 pairwise 기준 **0.843**에서, 전체 행에 중앙값 대체 후에는 **−0.041**로 바뀝니다.
- **양측 clipping을 하지 않는다.** R–Y 관계가 U자형인 경우가 많아 꼬리 값 자체가 신호입니다.
- **인자 선정은 편향보정 ε²(epsilon-squared) 기준.** 선형 Pearson r²은 U자 관계에서 좌우가 상쇄되어 과소평가합니다 — `Step24_R1`–`Y4`: 선형 r² **0.012** vs ε² **0.073**.
- **등급 판정에 효과 크기 하한을 함께 둔다.** p값만으로는 표본이 크면 미미한 효과도 최고 등급이 될 수 있습니다 — `Step6_R1`→`Y1`: n=1,462, p=0.006, ε²=0.009. eps2 하한(0.02) 없이 p값만 봤다면 "강함"으로 잘못 표시됐을 사례입니다(`src/analysis/screening/selector.py`의 `confidence_tier`).
- **평가 분할은 LOT 단위 GroupShuffleSplit.** 같은 LOT의 wafer가 학습/평가 양쪽에 걸치면 정보가 샙니다(`sklearn.model_selection.GroupShuffleSplit(groups=Lot_ID)`, 15% 홀드아웃, `random_state=42`). K-fold 교차검증은 오프라인 벤치마크 스크립트(`scripts/benchmark.py`, `GroupKFold(5)`)에서만 쓰이며, 실제 학습 API는 이 단일 분할만 사용합니다.
- **관리한계는 X축 SPC 방식.** Y의 분위수 구간에서 X 범위를 역산하지 않고, 인자 자신의 분포(mean/std/Q1/Q3)에서 산출합니다.

## 분석 파이프라인

### 1. 인자 스크리닝

88개 인자(R 48 + D 10 + Config 30) × 타깃 5개(Y1~Y5)에 대해 편향보정 ε²을 산출합니다. 연속형(R/D)은 분위수 8구간 ANOVA, 범주형(Config)은 카테고리 ANOVA를 사용합니다. 타깃별로 BH-FDR(Benjamini-Hochberg)을 적용해 q값을 계산합니다. 기여율(`contribution_pct`) 분모는 **해당 타깃의 전체 88개 인자 풀**입니다.

train.CSV 기준 타깃별 1위 인자:

| 타깃 | 1위 인자 | ε² | 기여율 | 등급 |
|---|---|---|---|---|
| Y1 | Step28_R1 | 0.192 | 63.9% | 강함 |
| Y2 | Step16_R1 | 0.159 | 63.1% | 강함 |
| Y3 | Step1_D1 | 0.660 | 92.6% | 강함 |
| Y4 | Step24_R1 | 0.073 | 41.2% | 보통 |
| Y5 | Step18_R1 | 0.287 | 76.7% | 강함 |

**Config 30개는 어떤 타깃에서도 BH-FDR을 통과하지 못했습니다.** Config만 따로 묶어 검정하면(30개 Config × 5개 타깃 = 150건, `src/analysis/llm_stats.py`의 `config_main_effect_screening`) 이 표본에서 검출 가능한 최소 효과 크기(MDE)는 ε² **0.00261**이고, 관측된 최대 ε²는 **0.00271**(`Step10_Config`→Y2)입니다. 둘이 거의 같으므로 "장비 영향이 없다"가 아니라 "이 표본으로는 검출 한계 부근의 효과까지만 잡을 수 있다"는 뜻입니다.

**다만 인자별 챔버 교호작용은 2건 유의합니다.** `Step16_R1`(q=0.0007)과 `Step18_R1`(q=0.0016)은 인자-타깃 관계 자체가 챔버(CH1~CH4)에 따라 다르게 나타납니다 — 장비 주효과는 없지만 인자와 챔버가 상호작용한다는 뜻이며, 이 두 인자는 챔버별로 별도 관리 구간을 둡니다.

화면(원인 분석 탭, 학습 탭)에서는 BH-FDR 게이트로 인자를 걸러내지 않고 **4단계 신뢰도 배지**(강함 p<0.01 & ε²≥0.10 / 보통 p<0.05 & ε²≥0.05 / 약함 p<0.20 / 참고, ε²<0.02는 항상 참고)로 모든 인자를 표시합니다. q값은 계산되어 툴팁에 노출되지만 화면 표시 여부를 걸러내는 필터로는 쓰이지 않습니다. **단, 사전 알람 생성과 JSON 보고서 포함 여부는 p<0.05(강함·보통 등급)로 제한됩니다.**

### 2. 학습 및 검증

타깃별로 독립된 `lightgbm.LGBMRegressor(n_estimators=300, random_state=0, verbose=-1)`(`LGBM_PARAMS`, `src/ml/pipeline.py:52`)를 학습합니다. 이전에 쓰던 scikit-learn 히스토그램 기반 그래디언트 부스팅 회귀 모델은 LightGBM으로 대체됐습니다 — 같은 조건에서 약 32% 빠릅니다(`src/ml/pipeline.py:17` 모듈 설명, 커밋 `374ae0d`·`cd1572f`). 피처는 그 타깃에서 BH-FDR을 통과한 인자(들)의 원본값 + `_miss`(결측 여부) + U자 인자의 `_dev`(최적 중심으로부터의 절대편차)입니다. FDR 통과 인자가 없는 타깃은 모델을 만들지 않고 train 평균을 상수로 예측합니다.

`POST /api/train`은 업로드된 CSV 한 장을 받아 **그 파일 내부**에서 LOT 기준 85/15로 단일 분할(GroupShuffleSplit)하고, 남은 15%(내부 홀드아웃)로 평가합니다 — 별도로 준비된 평가셋을 자동으로 붙여 평가하지 않습니다(응답 스키마의 `final_y_metrics.test`는 "내부 홀드아웃 분할"을 가리키는 키 이름이지, 번들 `test_remove_y.CSV`를 의미하지 않습니다 — 애초에 그 파일은 Y가 전량 결측이라 평가 자체가 불가능합니다). 내장 train.CSV를 업로드해 학습했을 때의 내부 홀드아웃 성능(8,500행 학습 / 1,500행 평가):

| | Y1 | Y2 | Y3 | Y4 | Y5 | Y(최종) |
|---|---|---|---|---|---|---|
| R² | 0.085 | 0.031 | 0.533 | 0.020 | 0.182 | 0.175 |
| MAE | 1.152 | 1.906 | 0.910 | 1.100 | 0.303 | 2.739 |

**R² 값 자체는 성공 기준이 아닙니다.** 결측률이 구조적으로 높아 미계측 wafer는 예측할 정보가 없습니다. 이 수치는 구현이 깨지지 않았는지 확인하는 회귀 테스트 기준값입니다.

전처리 방식별 비교(`scripts/benchmark.py`의 `GroupKFold(5)` out-of-fold 평가 기준, Y 최종 R² -- 이 비교 자체는 `GET /api/training/preprocessing-comparison`으로 조회할 수 있지만 현재 UI에는 표시하는 화면이 없습니다):

| 방식 | R² |
|---|---|
| A. 중앙값 대체 + 클리핑 | 0.114 |
| B. 전체 인자 + NaN 보존 | 0.146 |
| C. 선정 인자 + `_dev` + `_miss` (채택) | 0.177 |

전처리 선택이 알고리즘 선택보다 성능에 크게 기여합니다. `src/ml/training.py`는 저장소에 남아 있지만 **`/api/train`에서는 호출되지 않고 테스트에서만 실행되는 미사용 모듈**입니다. GroupKFold 기반 후보 앙상블을 고르던 옛 모듈과 HGBR-vs-RandomForest 비교 함수는 LightGBM 단일화 정리 커밋(`cd1572f`, −5,858줄)에서 저장소에서 완전히 삭제됐습니다 — `src/ml/hybrid.py`에는 이제 전처리·아티팩트 유틸(`AutoFeaturePreprocessor`, `ModelStagingDirectory`, `save_hybrid_bundle` 등)만 남아 있고, 이들은 `api/routes/data.py`·`src/ml/pipeline.py`가 실제로 사용합니다. LightGBM은 이제 `requirements.txt`에 있고(`lightgbm==4.7.0`) 주 모델입니다. XGBoost/CatBoost 등 다른 부스팅 라이브러리와의 비교는 여전히 의존성에 없어 이 저장소 안에서는 재현되지 않습니다.

### 3. SPC 관리한계 (인자별 이탈 판정)

인자별로 **X 자신의 산포**에서 관리한계를 산출합니다(train.CSV 기준). Y를 참조하지 않습니다. 이 관리한계는 산점도 참조선, JSON 보고서/SUNI 챗봇의 `control_limits`·`alarms`(개별 인자가 정상 범위를 벗어난 wafer 목록)에 쓰입니다 — 아래 "5. 웨이퍼 수율 예측 알람"이 쓰는 수율 예측 탭의 판정 기준과는 다른, 별개의 파이프라인입니다.

```
UCL(LCL) = Q3(Q1) ± 1.5 × IQR   ← 알람 기준(IQR×1.5)
mean ± 3σ, mean ± 6σ            ← 참조선(알람에는 쓰이지 않음)
```

단조 증가/감소 인자(예: `Step1_D1`)는 "나쁜 쪽" 한 방향에서만 알람이 발생합니다(단측). 관리한계선의 값이 해당 인자의 관측 범위 [min, max]를 벗어나면 그 선은 그리지 않습니다.

test.CSV(1,000장)에 적용한 결과: **알람 wafer 19장(1.9%), 알람군 최종 수율(83.15)이 무알람군(89.29)보다 6.14%p 낮습니다.** (알람은 wafer×인자 조합 단위로도 집계되어 개별 레코드는 20건입니다.)

기준 선택 근거(동일 5개 인자, test.CSV 적용):

| 기준 | 알람 wafer | 수율차 |
|---|---|---|
| Q1 ~ Q3 | 279장 (27.9%) | −2.39%p |
| IQR 1.0배 | 42장 (4.2%) | −4.11%p |
| **IQR 1.5배 (채택)** | **19장 (1.9%)** | **−6.14%p** |
| ±3σ | 15장 (1.5%) | −6.05%p |
| ±6σ | 0장 | 전부 데이터 범위 밖 |

±3σ·±6σ·Q1/Q3는 산점도 참조선으로만 표시하고 알람 판정에는 사용하지 않습니다.

### 4. 권장 구간과 개선 제안

관리한계와 별개로, 구간 평균 불량률이 train 전체 평균 이하인 **연속 quantile-bin 구간**을 인자별 권장 구간으로 제시합니다(`src/analysis/recommendations.py`). 이 구간은 관리한계(LCL~UCL) 안쪽으로 clamp되며, clamp 결과 구간이 사라지면 그 인자는 목록에서 제외됩니다.

권장 구간은 이 SPC 방식 하나로 고정되지 않습니다. **SPC·ML 두 방식을 각각 산출해 F2×안정성 점수가 더 나은 쪽을 채택**합니다(`compare_methods`, `src/analysis/window_methods.py`). SPC는 위 12-quantile-bin 규칙 그대로이고, ML은 얕은 `DecisionTreeRegressor`(`max_depth=3`)의 리프 경계입니다. 두 방식 모두 같은 train x/y 쌍에서 출발해 같은 방식으로 관리한계 안쪽으로 clamp되고, 부트스트랩 재표집(60회)으로 추정한 폭 안정성까지 곱해 채점하므로 ML이 좁은 구간에 과적합해 점수만 높이는 경우를 걸러냅니다. 채택된("adopted") 쪽만 알람 로그·개선 권장 목록·`optimal_center`처럼 실제 판정에 쓰이고, 진 쪽은 화면 비교용으로만 남습니다 — 원인 분석 인자 카드의 SPC/ML 토글(`MethodToggle`, `root-cause/page.tsx`)로 두 방식의 구간·점수를 나란히 비교하고, 채택된 쪽에는 체크 배지가 붙습니다.

| 인자 | 타깃 | 권장 구간(train 기준) | 구간 내/전체 평균 대비 기대 개선 |
|---|---|---|---|
| Step28_R1 | Y1 | 54.7~61.5 | −18% |
| Step16_R1 | Y2 | 55.7~61.7 | −17% |
| Step1_D1 | Y3 | 3.0~9.0 | −20% |
| Step24_R1 | Y4 | 53.2~61.5 | −9% |
| Step18_R1 | Y5 | 52.4~59.7 | −27% |

test.CSV 기준 개선 권장 레코드는 254건이며(관리한계 이탈로 이미 알람에 잡힌 건은 중복 집계하지 않음), 등급이 강함·보통인 인자만 담습니다.

### 5. 웨이퍼 수율 예측 순위 (기대 회수 기반)

위 1~4는 인자 단위 통계 분석입니다. 이와 별개로 `GET /api/alerts/ranking`(수율 예측 탭)이 쓰는 독립된 파이프라인이 있습니다(`src/analysis/yield_prediction.py`) — "이 인자 값이 평소와 다른가"(SPC)가 아니라 **예측 최종 수율(Y = 100 − Y1~Y5의 합) 오름차순 순위**로 검토 우선순위를 제시합니다. R² 0.12로 예측값 절대 크기는 신뢰할 수 없다는(순위는 맞지만 값은 못 맞추는) 검증 결과에 따라 목표 수율 절대 임계·5단계 등급 판정 방식은 폐기됐습니다(`docs/decisions.md`).

- **정렬**: y(=100 − Σ Y1~Y5, 실측 우선으로 채운 뒤 재계산) 오름차순. 절대 컷·등급 구간은 쓰지 않습니다.
- **신뢰도(n/5)**: 5개 불량모드(Y1~Y5) 중 기여율 10% 이상인 핵심 인자가 실측된 타깃 수입니다. 신뢰도가 0인(다섯 타깃 모두 핵심 인자 미계측) wafer는 순위 목록에서 빼고 "미계측" 묶음으로 따로 보여줍니다 — 민감도 슬라이더로 재현율을 조절하던 방식은 올릴수록 재현율이 오히려 떨어져(0.317→0.281) 폐기됐고, 대신 상위 N(표시 개수) 슬라이더로 대체됐습니다.
- **권장사항**: 핵심 인자가 계측된 타깃은 권장 구간 조정(SPC/ML 권장 구간 재사용 + 2차 곡선 감소량), 미계측 타깃은 계측 확대를 제안합니다.
- **예측 구간(conformal)**: `src/analysis/alarm_gbdt.py`의 `compute_holdout_predictions`가 train을 랏 단위 `GroupKFold(5)`로 나눈 out-of-fold 잔차의 90분위를 여유(`q`)로 내지만, 이는 웨이퍼 단위 판정 기준이 아니라 보고서/챗봇의 `limitations` 한 줄로만 쓰이는 참고 캐비어트입니다(웨이퍼 단위 여유가 데이터셋에 따라 ±5%p대로 넓어 그 폭으로 판정 가능한 wafer가 6% 수준에 그친 것이 폐기 근거) — 데이터셋마다 다시 계산되므로 고정 수치를 답으로 쓰지 않습니다.

## 화면 구성

6개 탭 — 모니터링 홈 / Config별 트리맵 / 원인 분석 / 수율 예측 / 알림 기록 / 즐겨찾기(`frontend/components/Sidebar.tsx`의 `navigationItems`). 좌측 접이식 사이드바 + 우측 SUNI AI 어시스턴트 패널. 첫 접속 시 두 패널 모두 펼쳐진 상태로 시작하며, 접힘/펼침 상태는 쿠키에 저장되어 다음 방문에도 유지됩니다. 모델 학습은 별도 탭이 아니라 사이드바 하단 4버튼(순서대로 **모델 학습 / 모델 분석 / 알림·자동화 설정 / 화면 모드**) 중 하나로 여는 팝업입니다. 세 상태 버튼(화면 모드 제외)은 같은 모양의 점을 공유하지만 의미는 서로 다릅니다 — 모델 학습은 이 세션에서 수동 업로드 학습을 실행했는지(연결)와 내장 데이터로 학습된 상태(오프라인) 둘만 구분하고, 모델 분석은 초록(분석 완료)·회색(미실행)·주황(실패)이며, SQL 연결 여부는 모델 분석 점이 아니라 알림·자동화 설정 점(초록=채널 연결+자동화 켜짐, 회색=미설정, 주황=자동화 실행 오류)이 보여줍니다(`Sidebar.tsx:91-137`). 모델 학습 팝업은 최근 학습 정보 3줄, SQL 호스트·포트, Refresh 주기, 파일 첨부·수동 학습 실행을 담습니다. **재학습(수동·자동 공통)은 더 이상 승격 게이트를 거치지 않습니다 — 학습이 성공하면 무조건 기존 모델을 교체합니다**(`src/runtime/store.py:20`, `:367`의 `promote_if_better`). 예전에 있던, 홀드아웃 R²가 기존 모델보다 0.005 넘게 나빠지면 교체를 막던 승격 게이트는 폐지됐고, 대신 R²가 기존 모델의 절반 이하로 나빠지면(`REGRESSION_WARNING_RATIO`) 교체는 그대로 진행하되 팝업에 경고만 표시합니다(`docs/decisions.md`: "승격 게이트: 성능 저하 시에도 무조건 교체").

**자동화.** 감시 디렉터리를 폴링해 Y 컬럼 유무로 학습/평가 데이터셋을 가르던 옛 모듈(`AUTO_INGEST_DIR`)은 삭제됐습니다 — "자동화는 수율 예측만 계산해야 한다"는 원칙에 따라 자동 학습·자동 원인분석 트리거를 걷어냈습니다(`api/main.py` 주석, 커밋 `be0e1f7` "사이드바 4버튼 재편과 자동화 분리"). 지금 주기적으로 자동 실행되는 것은 **`src/automation/yield_dispatch.py`** 하나뿐입니다 — "알림·자동화 설정" 팝업에 저장한 Refresh 주기로 APScheduler `IntervalTrigger`가 돌며(`api/main.py`, 설정 저장 시 곧바로 재스케줄), SQL로 신규 배치를 가져와(`sql_source.py`) 그 시점의 챔피언 모델로 **수율만** 예측해 알림을 발송합니다 — 학습도, 원인분석도, 모니터링·트리맵 스냅샷 갱신도 건드리지 않습니다. 접속 정보(host/port/db/user, 알림·자동화 설정 팝업에 저장)와 서버 환경변수(`AUTO_INGEST_DB_DRIVER`, `DB_PASSWORD`, `AUTO_INGEST_QUERY`, 선택적으로 `AUTO_INGEST_CURSOR_COLUMN`)가 모두 갖춰지고 실제 접속·조회에 성공해야(10초 타임아웃) 동작하며, SQL이 설정돼 있지 않거나 실패하면 이 잡은 그냥 건너뜁니다 — 내장 데이터로 자동 폴백하지 않습니다. DB 엔진은 코드에 고정하지 않고 SQLAlchemy dialect+driver 문자열을 그대로 씁니다.

모니터링·Config별 트리맵·원인 분석·수율 예측 네 화면을 한 번에 갱신하는 리프레시 파이프라인(`src/automation/refresh.py`)은 더 이상 주기 잡이 아닙니다 — "모델 분석" 팝업의 [분석 시작] 버튼, 서버 부트스트랩, 재학습 직후(활성 모델 스냅샷 무효화) 세 시점에만 실행됩니다. 이제는 SQL을 직접 조회하지 않고, 수동 업로드로 등록한 평가 데이터셋이 있으면 그것을, 없으면 내장 `test_remove_y.CSV`로 폴백합니다(`_resolve_source`) — SQL 조회는 "데이터베이스에서 불러오기" 버튼과 위 `yield_dispatch.py`만 담당합니다.

알림 발송은 매일 09:00·13:00(KST, APScheduler `CronTrigger`), `yield_dispatch.py`의 주기 실행, "분석 실행 직후", 수동 "알림 전송" 버튼 등 여러 트리거가 모두 같은 `dispatch_yield_update`를 거칩니다. 억제 규칙은 시간당 발송 예산(6건, 모든 트리거 공통)과 수동 트리거 전용 10분 최소 간격 두 가지만 남았습니다 — 예전에 있던 "알람 신뢰도 낮음 스킵"과 "이전 스냅샷 대비 신규 알람만 발송" 필터, 그리고 그 필터를 쓰던 옛 등급별 알람 자동 발송은 폐기됐습니다(`src/notifications/yield_update_dispatch.py`, `src/automation/refresh_dispatch.py`).

- **모니터링 홈**(`/monitoring`) — 가장 최근 원인 분석 결과를 세 블록으로 요약합니다. 기준은 "엔지니어가 오늘 결정할 수 있는가"입니다.
  - **① 조치 우선순위** — 타깃별 파레토 기여율 10% 이상인 인자를 전부(개수 고정 아님, train.CSV 기준으로는 5개), 기대 회수(=회수 폭 × 손실 비중) 내림차순으로 보여줍니다. 회수 폭은 "구간 밖 평균 손실 − 구간 안 평균 손실"(`src/analysis/action_priority.py`), 비중은 그 타깃이 5개 타깃 전체 손실에서 차지하는 몫입니다 — 둘을 곱한 결과만이 아니라 두 요소를 함께 보여줍니다. 기대 회수 0.1%p 미만인 행은 흐리게 표시하고 사유(회수 폭이 작음/비중이 낮음)를 붙입니다.
  - **② 조치 가능 범위** — 같은 인자들의 "구간 밖 / 구간 안 / 미계측" 스택 막대. 분모는 항상 전체 wafer 수입니다(계측된 것만 100%로 그리지 않습니다) — 계측률이 인자당 15% 안팎이라 트랙 대부분이 미계측(빈 구간)으로 보이는 것이 정상입니다.
  - **③ 이 화면을 얼마나 믿을 수 있나** — 계측 편향(MNAR, 불량 상위 10% wafer의 계측률이 전체 계측률의 몇 배인지)과 랏 간/랏 내 분산 분해(무효과 기대선 포함)를 보여줍니다(`DataLimitationDiagnostics.tsx`). 분산 분해 패널 하단에는 그 변동이 **어느 불량모드(Y1~Y5)에서 오는지**를 100% 누적 막대로 한 번 더 쪼갠 「불량모드별 변동 기여」가 있습니다 — 총 손실 `L = Y1+…+Y5`일 때 `cov(Yᵢ, L) / var(L) × 100`으로 정의해, 공분산의 선형성 덕분에 5개 모드의 합이 정확히 100%가 됩니다(`compute_mode_variance_share`, `src/analysis/data_limitations.py`). **이 막대만 train.CSV 실측 기준입니다**(블록③의 나머지·블록①·②는 각각 아래 설명대로 다른 기준입니다) — 내장 eval(`test_remove_y.CSV`)은 Y·Y1~Y5가 전량 결측이라 그 기준으로 계산하면 값이 100% 모델 예측값이 되어(예측 σ 1.98 vs 실측 σ 3.99), 불량모드 비중이 공정 사실이 아니라 모델이 학습한 것을 되비추기 때문입니다.

  블록①·②는 항상 train.CSV(학습 데이터셋) 기준입니다 — 지금 보고 있는 eval 데이터셋이 바뀌어도 흔들리지 않는, 학습 데이터가 뒷받침하는 판단이기 때문입니다. 블록③은 현재 분석 중인 eval 데이터셋 기준이며, 그 하단의 불량모드별 변동 기여만 예외적으로 train.CSV 기준입니다(위 설명 참고). 세 블록 모두 서버가 미리 계산해 스냅샷에 담아 보내므로(`_action_priority_payload`/`_fmea_payload`), 화면은 계산 없이 표시만 합니다. 화면 자체는 읽기 전용이며, 원인 분석 재실행·재학습·명시적 새로고침 전까지는 탭을 오가도 재조회하지 않습니다.
- **Config별 트리맵**(`/config-treemap`) — 모니터링 홈에서 분리된, 설비 구성(Model/EQ/Chamber) 전용 트리맵 탭입니다. 타일 면적은 표본 수(n), 색은 선택한 불량률 평균입니다. Config 인자는 어떤 타깃에서도 BH-FDR 유의를 통과하지 못하므로(약 600건 검정에 걸쳐 통과 0건, `docs/decisions.md`) 유의하지 않은 스텝은 타일을 무채색으로 두고 채색을 끕니다(`significant` 플래그, `ConfigTreemap.tsx`) — 안 보이는 게 아니라 "이 스텝은 근거가 없다"는 뜻입니다. Config 데이터 자체가 없는 스텝은 `empty_reason` 문구로 안내합니다.
- **원인 분석**(`/root-cause`) — "원인 분석 실행" → 상관관계 히트맵(R,D vs Y1~Y5 수치형 전용 — 셀 농도 ε²(설명력)·색상 방향 ρ(부호)를 항상 함께 표시. Config vs Y1~Y5 범주형 히트맵과 그 전환용 `보기` 토글은 삭제됐습니다 — Config는 위 트리맵과 마찬가지로 BH-FDR 통과 0건이라 실효가 없었고, 그 빈자리는 Config별 트리맵 탭이 대신합니다) + 타깃(Y1~Y5)별 Pareto 차트(타깃당 1개, 화면 상단에 고정) + 인자 카드 그리드(산점도/Box Plot). 인자 카드는 파레토 기여율 10% 이상인 인자만 고정 표시합니다 — "유의 인자만 보기"·"상위 N개" 같은 별도 토글은 없습니다(히트맵 안의 표시 체크박스와는 별개입니다). 각 카드의 `보기` 토글은 이제 Scatter Plot·Box Plot 두 가지뿐입니다(기본값 Scatter) — Pareto는 카드 토글에서 빠지고 화면 상단 고정 섹션으로 옮겨졌습니다. `비교` 토글로 "Y1~Y5 비교"·"장비별 Trellis"(Model/EQ/Chamber 분할) 모달을 엽니다. 산점도는 드래그로 사각 영역을 선택하면 평균·중앙값·최솟값·최댓값 통계 박스가 뜨고, 카드 우상단의 "⬇ 이미지 저장" 버튼은 카드 DOM 전체를 `{인자또는Pareto}_{타깃}_{뷰}_{YYYYMMDD-HHmm}.png` 규칙의 PNG로 캡처해 내려받으며(`frontend/lib/chartExport.ts`), 헤더의 별(☆) 버튼으로 즐겨찾기에 저장할 수 있습니다. 실행 직후 조치 우선순위(블록①)도 함께 재계산되어 저장됩니다
- **수율 예측**(`/alerts`) — "이 모델은 순위는 맞지만 값은 못 맞춘다"(상위 20장 적중 95%, R² 0.12)는 전제 위에 설계된 **순위 도구**입니다. Y(=100 − Σ Y1~Y5, 실측 우선 하이드레이션) 오름차순으로 전체 wafer를 나열하고, 웨이퍼·타깃별 핵심 인자(파레토 기여율 10% 이상, 계측된 것 중 최고 순위로 폴백)와 신뢰도(그 임계 이상 인자가 계측된 타깃 수 / 5)를 함께 보여줍니다. 권장사항은 인자→목표 구간 화살표 한 줄로 압축해 표시합니다. 신뢰도 0인 wafer(핵심 인자가 아예 없음)는 별도 "미계측" 블록으로 뺍니다. 화면 우상단에서 CSV 내려받기와 수동 "알림 전송"(연결된 Slack/Telegram/Gmail로 즉시 발송, 억제 규칙은 자동 발송과 동일)을 할 수 있습니다
- **알림 기록**(`/notify-history`) — 발송된(또는 조건 미달로 건너뛴) 알림 이력을 최신순으로 다시 볼 수 있는 탭입니다. 행을 펼치면 발송 당시의 메시지 전문을 재계산 없이 그대로 보여줍니다(`NotifyHistoryItem.message_text`).
- **즐겨찾기**(`/favorites`) — 원인 분석에서 저장한 그래프를 최신순 카드 그리드로 모아 보여줍니다. 점 데이터는 저장하지 않고 저장된 조건(데이터셋·타깃·인자·뷰 종류)으로 다시 조회해 썸네일을 그립니다. 카드 클릭 시 해당 인자의 원인 분석 화면으로 이동합니다

원인 분석·모니터링 홈·수율 예측 세 화면 모두 제목 아래에 같은 `LastRunNote` 컴포넌트로 마지막 실행 시각을 표시합니다(24시간이 지나면 "하루가 지났습니다"가 붙습니다).

데이터셋은 각 탭의 선택 UI에서 CSV를 업로드해 추가할 수 있고(`POST /api/datasets`), 내장 2종(train/test)과 함께 목록에 표시됩니다.

산점도는 Spotfire 방식(원형 마커, Color By)의 커스텀 SVG 컴포넌트(`ScatterChart.tsx`)이며, SPC 관리한계선과 12구간 구간 평균 불량률 추세선·권장 구간 밴드를 함께 표시합니다. Pareto 차트(`ParetoChart.tsx`)와 상관관계 히트맵(`CorrelationHeatmap.tsx`)도 정밀한 클릭/호버 제어를 위해 커스텀 SVG로 구현했습니다. Config(범주형) 인자의 박스플롯만 Plotly를 사용합니다.

## LLM 연동 (SUNI 챗봇)

- Upstage Solar API(OpenAI 호환 스펙)를 스트리밍(SSE)으로 호출합니다(`api/routes/chat.py`)
- `report` 모드(예시: "분석 보고서 생성" 버튼, 메시지에 "보고서" 등 키워드 포함): 6개 섹션(요약/불량 유형별 소견/장비 구성 소견/관리 대역 제안/한계/확인 필요 사항)으로 구성된 공정 보고서를 생성합니다
- `chat` 모드(그 외 자유 입력): 3~5문장으로 답합니다. 알람 개별 건(예: 특정 wafer의 알람)에 대한 질문에도 해당 레코드를 근거로 답하고, 수율 예측 탭의 순위 기반 판정(목표 수율·민감도 절대 컷·5분류·미분류 2사유는 폐기됨)·신뢰도(n/5)·conformal 여유(참고용 캐비어트, 판정 기준 아님)·대시보드 기능처럼 이번 분석 JSON에는 없는 시스템 동작 지식도 프롬프트에 실린 배경 지식으로 답합니다(`recommendations` 필드는 존재하지 않습니다 -- 권장 구간은 각 인자의 `window`에만 있습니다)
- 두 모드 모두 `/api/analysis/context`가 만드는 동일한 분석 결과 JSON(타깃별 1위 인자, 관리한계, 권장 구간, 챔버 교호작용, 알람 레코드, config 스크리닝, 한계)을 근거로 답하며, 시스템 프롬프트(`prompts/report_system.md`, `prompts/chat_system.md`)가 **숫자 생성 금지, 인과 표현 금지, "값을 조정하라" 같은 설정값 표현 금지**를 명시적으로 규정합니다
- `confidence`(신뢰도)는 LLM이 아니라 코드가 판정한 값을 그대로 따르게 합니다 — 판정 근거가 부족한 인자에 LLM이 임의로 관리 대역을 만들어내지 못하게 하는 장치입니다
- 원인 분석이 아직 실행되지 않은 상태의 질문은 LLM을 호출하지 않고 백엔드가 즉시 안내 메시지로 응답합니다
- 화면과 챗봇의 역할 분담 원칙(I-1): **서사적 해석·배경 설명은 챗봇, 판정 기준·표본 한계는 화면.** 챗봇을 열어야만 알 수 있는 판정 기준이 있으면, 챗봇을 클릭하지 않은 사용자가 잘못된 판단을 내린다. 이 원칙에 따라 다음은 화면에 유지한다(삭제 대상 아님): 수율 예측 순위·신뢰도(n/5) 표기, 계측률·표본 수 표기. (구 목표 수율·민감도 절대 컷, 알람 등급 마커, 신뢰도 게이트 배너는 그 판정 체계 자체가 폐기되며 함께 제거됐다.) 그 외 배경 설명·정성적 소견(계측률 한계, 인과 아님, 근거 부족 등급의 의미, 정밀도·재현율이 추정치라는 점 등)은 `LIMITATIONS`(`src/analysis/report.py`)로 옮겨 챗봇 컨텍스트에 실리며, "이 분석의 한계는?" 프리셋 질문으로 확인할 수 있습니다
- 환경변수: `UPSTAGE_API_KEY`, `UPSTAGE_BASE_URL`(기본 `https://api.upstage.ai/v1`), `UPSTAGE_MODEL`(기본 `solar-pro3`). 키는 백엔드 프로세스에만 두며 프런트 번들에는 절대 노출하지 않습니다(`NEXT_PUBLIC_` 접두사 미사용)
- API 키가 설정되지 않은 환경에서는 `/api/chat`이 503과 안내 메시지를 반환하며 서버 자체는 정상 동작합니다

## 검토했으나 채택하지 않은 것

**"안 만든 것"이 아니라 "검증하고 배제한 것"입니다.** (모니터링 홈 재설계에서 나온 화면/판정 로직 결정은 `docs/decisions.md`에 별도로 정리했습니다 — 핵심 인자 임계값 20%→10%, 예측 수율 카드·히스토그램·FMEA RPN·계측 확대 시뮬레이션 삭제와 그 대체 지표.)

- **LOT 단위 알람** — 랏 평균 분산비 `var(랏평균)/var(Y)` = 0.045입니다. 다만 랏당 25장이면 랏 효과가 전혀 없는 순수 노이즈에서도 이 값의 기대값이 1/25 = 0.04이므로, 이 수치 자체는 랏 효과의 증거가 아니라 노이즈 기대치입니다 — 진짜 ICC(1,1)은 0.005로 훨씬 작습니다(I-2: 이전 버전에서 0.045를 ICC 근사로 잘못 표기했던 것을 바로잡음). test.CSV에서 알람 2건 이상 LOT의 평균 수율(89.41)이 알람 0건 LOT(89.06)보다 오히려 높아 LOT 단위 신호가 뚜렷하지 않다는 결론은 이 정정으로 바뀌지 않으며 오히려 강화됩니다. wafer 단위로 설계했습니다.
- **관리한계 롤링 윈도우** — train 400 LOT을 100개씩 4구간으로 나눠 각각 관리한계를 산출해보면, 인자별로 구간 간 변동 폭이 대체로 1~2 수준(Step1_D1은 다소 크게 변동)으로 뚜렷한 시간 드리프트는 보이지 않았습니다.
- **Y의 Q1~Q3 구간에서 X 범위 역산** — 초기 구현이었으나 조건부 확률의 방향이 반대입니다(`P(X|Y)` 대신 `P(Y|X)`가 필요). X 자신의 분포 기반 SPC 관리한계로 교체 후 알람이 58장→19장으로 줄고 수율차는 −5.25%p→−6.14%p로 개선됐습니다.
- **알고리즘 교체(앙상블/RandomForest 비교)** — `src/ml/` 안의 옛 GroupKFold 후보 앙상블 모듈과 `src/ml/hybrid.py`의 비교 함수에 구현이 있었으나 실제 학습 API 경로에서는 호출되지 않았습니다. 피처가 타깃당 1~3개뿐이라 트리 앙상블이 찾을 상호작용이 거의 없고, 이득 대비 유지 비용이 크다고 판단했습니다. 그 구현은 이후 LightGBM 단일화 정리(커밋 `cd1572f`)에서 저장소에서 제거했습니다 — 지금은 `src/ml/hybrid.py`에 전처리·아티팩트 유틸만 남아 있습니다. 부스팅 라이브러리 자체는 이제 LightGBM으로 단일화됐고(`lightgbm==4.7.0`), XGBoost/CatBoost 등과의 비교는 여전히 의존성에 없어 이 저장소에서 재현되지 않습니다.
- **수율 예측 서비스(`/api/predict`) 및 SHAP 기여도 원인 분석** — 프로젝트 범위에서 제외했습니다. 통계적 원인 특정(ε² + BH-FDR + SPC)에 집중합니다. (해당 패키지 의존성도 `requirements.txt`에서 제거했습니다.)

## 한계

1. 이 분석은 **해당 인자가 계측된 wafer만** 대상으로 합니다. R은 전체의 15%, D는 5%만 계측됩니다. test.CSV 1,000장 중 상당수(선정 인자 기준 489장)는 선정 인자가 하나도 계측되지 않아 **판정 자체가 불가능**합니다.
2. ε²는 통계적 연관성이지 인과가 아닙니다. 공정 순서상 선행·후행 관계나 교락 인자는 반영되지 않았습니다.
3. Config에서 유의 인자가 검출되지 않은 것은 검출력 부족(MDE ε² 0.00261, 관측 최대 0.00271)이며, 영향이 없다는 뜻이 아닙니다.
4. 관리한계는 "평소와 다른가"를 판정하며 "수율이 좋은가"를 보장하지 않습니다.
5. `POST /api/train`의 성능 지표는 업로드 파일 내부 15% 홀드아웃 기준이며, 별도의 test.CSV로 일반화 성능을 확인한 것이 아닙니다.
6. 계측된 wafer와 미계측 wafer의 최종 수율 차이는 이 데이터셋에서 통계적으로 유의하지 않았습니다(R/D 계측이 하나도 없는 wafer 기준 t-검정 p=0.74). 계측 편향이 관측되지 않았다는 뜻이지, 다른 데이터셋에도 성립한다는 보장은 아닙니다.
7. 웨이퍼 수율 예측 알람의 conformal 구간이 넓은 것(웨이퍼 단위 약 ±5.6%p)은 모델 결함이 아니라 데이터의 설명력 한계입니다 — 전체 R+D 인자를 다 써도 test.CSV 기준 R²는 약 0.26입니다. 이 여유를 SUMMARY 등 여러 웨이퍼의 평균에 그대로 적용하면 평균의 불확실성을 개별값 수준으로 과대평가하므로, 집계에는 랏 블록 부트스트랩으로 별도 산출한 여유(약 ±0.2%p)를 씁니다.
8. 관리한계(mean±3σ/6σ, Q1/Q3 기반 IQR×1.5)는 인자별 SPC 참조·보고서 표기에는 그대로 쓰이지만, 웨이퍼 수율 예측 순위에서는 더 이상 쓰이지 않습니다 — 판정은 위 "5. 웨이퍼 수율 예측 순위"의 y 오름차순 정렬과 신뢰도(n/5)를 씁니다. 두 "관리한계"는 대상이 다르므로(하나는 인자 값 하나, 하나는 웨이퍼 최종 수율) 혼동하지 않아야 합니다.

## 구조

```text
api/            FastAPI 라우트(data/datasets/analysis/chat/state/notify/favorites/monitoring)·스키마·설정
src/analysis/   인자 스크리닝(ε², BH-FDR), 히트맵, SPC 관리한계, 권장 구간(SPC/ML 두 방식, window_methods.py),
                 웨이퍼 수율 예측 알람(GBDT+conformal, alarm_gbdt.py), 모니터링 홈 블록①·②(action_priority.py),
                 MNAR 계측 편향·랏 간/랏 내 분산 분해·불량모드별 변동 기여(블록③ 원천, data_limitations.py),
                 FMEA 원천(screening/fmea.py), 챗봇 컨텍스트용 통계, JSON 보고서
src/ml/         학습 파이프라인(LightGBM `LGBMRegressor`), 추론 메타데이터, 모델 저장
src/runtime/    데이터셋 레지스트리, 학습 Job, SQLite 이력 저장(즐겨찾기·모델 승격 이력 포함)
src/notifications/ Slack/Telegram/Gmail 발송(senders.py), 수율 예측 갱신 발송 오케스트레이션·dedupe·재시도
                 (yield_update_dispatch.py/yield_update_senders.py -- 자동 갱신·수동 버튼 공용), Telegram 인증 코드 흐름
                 (telegram_bot.py), 채널 설정 영속화(대기 상태 TTL 포함, settings_store.py)
src/automation/ 네 화면 리프레시 파이프라인(refresh.py -- 이제 이벤트 트리거 전용, 더 이상 주기 잡 아님)과 주기 자동화
                 (yield_dispatch.py -- SQL 배치 → 챔피언 모델 수율 예측 → 알림만 수행), SQL 데이터 소스 판단·증분
                 수집(sql_source.py), 발송 본문의 출처 한 줄만 남은 refresh_dispatch.py(옛 등급별 알람 자동 발송
                 로직은 폐기되고 src/notifications/yield_update_dispatch.py로 대체됨)
frontend/       Next.js UI (app/monitoring, app/root-cause, app/alerts, app/favorites, app/config-treemap, components/ai-panel)
prompts/        SUNI 챗봇 시스템 프롬프트(report_system.md, chat_system.md)
tests/          Python 테스트(pytest)
scripts/        오프라인 벤치마크(전처리 방식 비교)
config/         결측 스키마·알람 severity·전처리 정책 YAML
docs/decisions.md  폐기된 설계와 근거(임계값 변경, 삭제된 화면 블록 등) -- "검토했으나 채택하지 않은 것"과는 별개 목록
```

`src/ml/training.py`는 저장소에 남아 있지만 API 경로에서는 사용되지 않고 테스트에서만 실행됩니다. GroupKFold 기반 후보 앙상블을 고르던 옛 모듈과 HGBR-vs-RandomForest 비교 함수는 LightGBM 단일화 정리(커밋 `cd1572f`)에서 저장소에서 아예 삭제됐습니다 — `src/ml/hybrid.py`에 남은 `AutoFeaturePreprocessor`·`ModelStagingDirectory`·`save_hybrid_bundle` 등은 죽은 코드가 아니라 `api/routes/data.py`·`src/ml/pipeline.py`가 실제로 사용하는 전처리·아티팩트 유틸입니다.

## 로컬 실행

Python 의존성을 설치하고 저장소 루트에서 백엔드를 실행합니다.

```powershell
pip install -r requirements.txt
uvicorn api.main:app --reload
```

프런트엔드는 Node.js 20.9 이상을 사용합니다.

```powershell
Set-Location frontend
npm.cmd install
npm.cmd run dev
```

`frontend/.env.local.example`을 `frontend/.env.local`로 복사합니다. 프런트엔드가 읽는 값은 `NEXT_PUBLIC_API_BASE_URL` 하나뿐입니다.

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

SUNI 챗봇을 쓰려면 백엔드 프로세스 환경변수에 Upstage 키를 추가합니다(`.env.example` 참고).

```env
UPSTAGE_API_KEY=up_...
UPSTAGE_BASE_URL=https://api.upstage.ai/v1
UPSTAGE_MODEL=solar-pro3
```

기본 주소는 Next.js `http://localhost:3000`, FastAPI `http://127.0.0.1:8000`, Swagger `http://127.0.0.1:8000/docs`입니다.

### 콜드 스타트 부트스트랩과 모델 자동 복구

처음 띄우는 서버에는 학습된 모델도, 스냅샷도 없습니다. 첫 요청이 들어오면(`api/main.py`의 `ensure_usable_champion`) 쓸 수 있는 챔피언 모델이 있는지 확인하고, 없으면 내장 `train.CSV`로 업로드 학습과 같은 경로(`api/routes/data.py`의 `train_model`)를 그대로 태워 챔피언을 학습한 뒤 `run_refresh_pipeline(dispatch=False)`로 첫 예측·원인분석·모니터링 스냅샷까지 한 번에 만듭니다(`_train_bootstrap_champion`/`_run_bootstrap`, `api/main.py:187`). 진행 단계는 "데이터 확인 중 → 학습 중 → 평가·원인분석 중" 세 단계만 화면에 알리고, 그 밖의 가짜 진행률(%)은 표시하지 않습니다(`frontend/components/BootstrapStatusBanner.tsx`). 레지스트리에 모델 행은 있는데 아티팩트 파일이 없는 경우(배포 볼륨 초기화 등)도 같은 경로로 자동 복구합니다 — 활성 모델 로드에 실패하면 죽은 포인터를 레지스트리에서 지우고 재학습을 다시 트리거합니다. 부트스트랩이 완전히 실패했을 때는 사유를 구분합니다: 내장 `train.CSV` 자체가 없는 `bundled_train_data_missing`은 재시도해도 소용없는 케이스라 배너에 재시도 버튼을 띄우지 않고, 그 외 사유는 다음 요청이 자동으로 다시 시도합니다(`reason` 필드, 커밋 `d5f7d71`·`809c8be`·`53c6b0a`).

알림 연동(Slack/Telegram/Gmail)을 쓰려면 아래 환경변수도 추가합니다(`.env.example` 참고, 전부 선택값입니다). 설정하지 않으면 해당 채널은 UI에서 "연결하기"를 눌러도 실제 발송은 되지 않습니다. Slack은 env var가 필요 없습니다 -- Webhook URL을 설정 패널에서 직접 입력해 `RuntimeStore`에 저장합니다.

```env
TELEGRAM_BOT_TOKEN=
TELEGRAM_BOT_USERNAME=
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
SMTP_FROM_EMAIL=
NOTIFY_VERIFY_BASE_URL=
```

**Telegram 연결 절차**: ① BotFather에서 봇을 만들어 토큰과 username을 발급받습니다. ② `TELEGRAM_BOT_TOKEN`(그대로)과 `TELEGRAM_BOT_USERNAME`(맨 앞 `@` 제외)을 서버 환경변수로 설정합니다 -- **두 값 모두 서버를 재시작해야 반영됩니다**(long-polling 루프가 기동 시점에만 뜹니다). ③ 대시보드가 안내하는 봇 링크를 열어 `/start`를 보냅니다. ④ 봇이 6자리 인증 코드를 답장으로 보내면 대시보드 알림 설정 패널에 그 코드를 입력합니다 — **코드는 10분 안에 입력해야 하며(`CODE_TTL_MINUTES`), 유효 시간을 넘기면 다시 `/start`부터 반복**합니다(코드는 메모리에만 보관하므로 서버 재시작 시에도 사라집니다).

**Gmail 연결 절차**: ① Google 계정에 2단계 인증을 켭니다(앱 비밀번호는 2단계 인증이 켜져 있어야 발급됩니다). ② Google 계정 설정에서 앱 비밀번호(16자)를 발급받아 **공백을 제거하고** `SMTP_PASSWORD`에 넣습니다. ③ `SMTP_HOST`(`smtp.gmail.com`)·`SMTP_PORT`(587)·`SMTP_USER`·`SMTP_PASSWORD`·`SMTP_FROM_EMAIL` 5개를 채웁니다 -- **`SMTP_FROM_EMAIL`은 `SMTP_USER`와 같은 주소여야 합니다**(Gmail SMTP는 인증 계정과 다른 발신 주소를 거부합니다). ④ 대시보드에서 "연결하기"를 누르면 인증 메일이 발송되고, 메일의 확인 링크를 눌러야 연결이 완료됩니다(대기 상태, `pending`). **이 대기 상태는 5분이 지나면 서버에서 자동으로 만료되어 미연결로 돌아갑니다** — 5분 안에 메일의 링크를 눌러야 합니다. 만료 판정은 조회 시점에 이루어지며(`src/notifications/settings_store.py`의 `PENDING_TTL_SECONDS`), 이미 연결 완료(`verified: true`)된 채널은 만료 대상이 아니라 재시작·재접속 후에도 계속 유지됩니다.

인증 메일의 확인 링크는 기본적으로 **그 요청을 받은 API 서버 자신의 주소**(`request.base_url`)를 가리킵니다 — 프런트엔드(Next.js) 오리진을 기본값으로 쓰면 `/api/notify/gmail/verify`에 대응하는 Next 페이지가 없어 링크가 404로 갑니다. API 서버가 리버스 프록시·로드밸런서 뒤에 있어 `request.base_url`이 실제 공개 주소와 다르면 `NOTIFY_VERIFY_BASE_URL`을 명시적으로 설정합니다.

감시 디렉터리를 폴링하던 자동 수집 기능(`AUTO_INGEST_DIR`/`AUTO_INGEST_ENABLED`)은 삭제됐습니다(위 "자동화" 참고) — 지금 주기 자동화가 쓰는 데이터 소스는 SQL뿐입니다.

주기 자동화(`yield_dispatch.py`)와 "데이터베이스에서 불러오기" 버튼에서 SQL 데이터 소스를 쓰려면 아래 환경변수를 추가합니다 -- 접속 정보(host/port/db/user) 자체는 알림·자동화 설정 팝업에서 저장하며, 비밀번호만 서버 환경변수로 둡니다. 하나라도 비어 있거나 접속·조회에 실패하면 `yield_dispatch.py`는 그 회차를 그냥 건너뜁니다 — 내장 데이터로 자동 폴백하지 않습니다(내장 폴백은 이벤트 트리거 전용인 리프레시 파이프라인만 합니다. 위 "자동화" 참고).

```env
AUTO_INGEST_DB_DRIVER=
DB_PASSWORD=
AUTO_INGEST_QUERY=
AUTO_INGEST_CURSOR_COLUMN=
```

## 비밀값 취급

- API 키·토큰·Webhook URL을 코드에 직접 쓰지 않습니다
- 테스트에는 더미 값 또는 `monkeypatch`를 씁니다. 더미 값이라도 실제 서비스의 자격 증명과 같은 **형태**(예: Slack Webhook의 `T{8~10자}/B{8~10자}/{24자}` 구조)를 그대로 흉내 내지 않습니다 — GitHub Secret Scanning은 값의 진위가 아니라 구조를 보고 차단합니다
- 실제 값은 `.env`(로컬)와 Railway Variables(배포)에만 둡니다
- `.env`는 절대 커밋하지 않습니다 — `.gitignore`에 `.env`, `.env.*`(단 `.env.example`은 예외)가 있는지 새 환경변수를 추가할 때마다 확인합니다
- 커밋 전 자동 검사가 필요하면 아래로 pre-commit 훅을 설치합니다(`.git/hooks/`는 저장소에 포함되지 않으므로 각자 한 번씩 설치해야 합니다):

  ```powershell
  # PowerShell
  .\scripts\install-hooks.ps1
  ```

  ```bash
  # bash/WSL/git bash
  ./scripts/install-hooks.sh
  ```

  설치 후에는 커밋 내용에 Slack Webhook·Telegram 봇 토큰·`up_`/`sk-` 형태의 API 키가 포함되면 커밋 자체가 거부됩니다.

## API

- 상태 확인: `GET /health`, `GET /ready`, `GET /`
- 데이터셋: `GET /api/datasets`, `POST /api/datasets`(업로드), `DELETE /api/datasets/{id}`, `GET /api/datasets/{id}/download`, `GET /api/datasets/{id}/schema`
- CSV 검증·전처리(현재 UI에서는 호출하지 않는 독립 엔드포인트): `POST /api/validate`, `POST /api/preprocess`
- 학습: `POST /api/train`(동기), `POST /api/train/jobs`(비동기) + `GET /api/train/jobs/{job_id}`
- 모델: `GET /api/models`, `GET /api/models/{model_id}`, `GET /api/models/{model_id}/references`, `DELETE /api/models/{model_id}`, `GET /api/models/performance`, `GET /api/models/promotion-history`(승격 게이트 통과 여부와 무관한 학습 시도 이력), `GET /api/model/latest`
- 인자 스크리닝: `GET /api/screening/pareto`, `GET /api/screening/heatmap`(`dataset` 파라미터만 받는다 -- NG-1: 범주형(Config vs Y1~Y5) 보기와 그 전환용 `kind` 파라미터는 삭제됐다. 항상 R,D vs Y1~Y5 수치형이며 ε²(농도)·ρ(색상 방향)를 한 응답에 함께 낸다), `GET /api/screening/scatter`, `GET /api/screening/scatter/categorical`
- SPC 참조·수율 예측: `GET /api/control-ranges`(인자별 SPC 관리한계, 산점도 참조선·보고서 `control_limits`에 씀), `GET /api/alerts/ranking`(수율 예측 탭의 순위 목록 -- y 오름차순, 신뢰도 n/5, 타깃별 핵심 인자·권장사항), `GET /api/alarms/history`(발송된 알림 스냅샷 이력, 승격과 무관하게 불변 기록)
- 분석 보고서: `GET /api/analysis/report`(다운로드용 JSON, 현재 UI에는 다운로드 버튼 없음), `GET /api/analysis/context`(SUNI 챗봇 컨텍스트용), `GET /api/analysis/reliability`, `GET /api/training/preprocessing-comparison`
- 수율 예측(순위 도구): `GET /api/alerts/ranking`(수율 예측 탭이 쓰는 유일한 조회 -- 핵심 인자·신뢰도·권장사항을 함께 낸다)
- SUNI 챗봇: `POST /api/chat`(SSE 스트리밍, `mode: "report" | "chat"`)
- 탭 상태 저장(재접속 시 최근 결과 복원): `GET /api/state/latest`, `POST /api/state/training`, `POST /api/state/analysis`, `POST /api/state/alarms`
- 알림 채널: `GET /api/notify/settings`, `POST /api/notify/slack`(+`/test`), `POST /api/notify/telegram/verify`(+`/test`), `POST /api/notify/gmail`(+`/test`), `GET /api/notify/gmail/verify`, `POST /api/notify/conditions`, `DELETE /api/notify/{channel}`, `POST /api/notify/dispatch`(원인 분석 실행 직후 알람 발송), `POST /api/notify/yield-update/dispatch`(수율 예측 탭의 수동 "알림 전송" 버튼 -- AUC 게이트와 무관, 억제 규칙은 자동 발송과 공유)
- 즐겨찾기: `POST /api/favorites`, `GET /api/favorites`, `DELETE /api/favorites/{id}`
- 모니터링: `GET /api/monitoring/config-treemap`

**제거된 것**: `GET /api/analysis/measurement-expansion`(계측 확대 시뮬레이션 -- 모니터링 홈 재설계로 삭제, 근거는 `docs/decisions.md`). `YieldPredictionResponse.yield_summary`(예측 수율 평균·히스토그램·모드별 손실 -- 같은 재설계로 삭제). `FmeaTablePayload.items`/`no_qualifying_factor`/`measurement_shortage_wafers`/`correlation_shortage_wafers`(행별 FMEA 표 -- 블록①로 흡수, 응답에는 `mnar_rate_report`/`variance_decomposition`만 남음).
**추가된 것**: 조치 우선순위(회수 폭·비중·기대 회수·계측 카운트) -- `_action_priority_payload`가 `analysis.actionPriority`로 스냅샷에 싣는다. 같은 함수가 불량모드별 변동 기여(`mode_variance_share`, train.CSV 기준)도 함께 싣는다.

## Railway 배포

`railway.json`의 시작 명령:

```text
uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1
```

Healthcheck는 `/health`를 사용합니다. Railway 무료 티어(512MB RAM / 1vCPU)를 기준으로 업로드 상한을 20MB로 두고 `--workers 1`을 유지합니다(`config/upload_limits.yaml`). 배포 환경에는 최소 `FRONTEND_ORIGINS`, SUNI 챗봇을 쓸 경우 `UPSTAGE_API_KEY` 등을 설정해야 합니다. 로컬 검증은 Railway의 실제 RAM, OOM, 499/502 발생 여부를 대신하지 않으므로 운영 배포 후 별도 확인이 필요합니다.

### Railway 볼륨 설정 (필수)

Railway는 재시작·재배포마다 컨테이너 파일시스템을 초기화합니다. 볼륨을 붙이지 않으면 텔레그램·Slack·Gmail 연동, 챔피언 모델 지정, 분석 스냅샷, 즐겨찾기, 알림 발송 이력, 업로드한 데이터셋, 모델 아티팩트가 재시작마다 전부 사라집니다.

```text
Settings -> Volumes -> New Volume
Mount path: /app/var        (/app/data 는 사용하지 마세요 -- 저장소의 data/bundled/
                              내장 데이터(train.CSV 등)가 가려져 콜드 스타트가 실패합니다)
```

환경변수 설정은 별도로 필요 없습니다. Railway가 자동 주입하는 `RAILWAY_VOLUME_MOUNT_PATH`를 `api/settings.py`가 읽어 모델·runtime DB·아티팩트·학습 잡·업로드 데이터셋 5개 저장 경로를 전부 볼륨 안으로 전환합니다.

`MODEL_DIR`·`RUNTIME_DB_PATH`·`RUNTIME_ARTIFACT_DIR`·`TRAINING_JOB_ARTIFACT_DIR`·`DATASET_UPLOAD_DIR` 중 하나라도 Railway Variables에 명시적으로 설정되어 있으면 그 경로가 볼륨 자동 감지보다 우선합니다(명시적 설정 우선 원칙) — 볼륨을 붙였는데도 반영되지 않는다면 이 변수들부터 확인하고, 상대 경로이거나 로컬 기본값과 같다면 지우는 것을 권장합니다.

기동 로그에 볼륨 연결 여부와 5개 저장 경로, 환경변수로 덮어써진 경로가 1회 출력되므로(`storage: volume=...` 또는 `storage: 볼륨 미연결 ...`) 배포 후 반드시 로그로 확인하세요.

**볼륨에 저장되는 것**

- 모델 아티팩트(`models/`)
- runtime DB(연동 정보 · 챔피언 모델 지정 · 분석 스냅샷 · 즐겨찾기 · 알림 발송 이력)
- 학습 잡 산출물 · 업로드한 데이터셋

**볼륨이 없으면**: 재시작·재배포마다 위 항목이 전부 소실되어 텔레그램 연결부터 모델 학습까지 다시 해야 합니다.

## 검증

```powershell
python -m pytest -q
Set-Location frontend
npm.cmd run lint
npm.cmd run build
```

원본 학습 CSV와 실제 운영 환경이 없는 경우, 테스트 fixture 결과를 실제 모델 성능이나 Railway 안정성으로 표현하지 않습니다.
