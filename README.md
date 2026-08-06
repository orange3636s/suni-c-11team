# SUNI 11팀 — 반도체 공정 원인 분석 & 사전 알람

반도체 제조 공정 데이터에서 **Y1~Y5 불량률과 통계적으로 연관된 공정 인자를 특정하고, SPC 관리한계로 이탈 wafer를 탐지하며, 권장 구간으로 개선 방향을 제시하는** FastAPI + Next.js 애플리케이션입니다. Upstage Solar LLM 연동으로 이 분석 결과를 자연어 보고서·대화형 답변으로 변환하는 SUNI 챗봇을 포함합니다.

**이 프로젝트의 산출물은 수율 예측 정확도가 아니라 "어떤 step의 어떤 인자가 어떤 불량과 관계있는가"의 통계적 특정입니다.** Y1~Y5를 학습하는 회귀 모델이 포함되어 있지만, 이는 스크리닝 결과의 구현 정확성을 확인하는 회귀 테스트 기준값이지 제품 기능(수율 예측 서비스)이 아닙니다.

## 데이터

내장 데이터셋 4종(`data/bundled/`).

| 파일 | 행 × 열 | LOT | 비고 |
|---|---|---|---|
| train.CSV | 10,000 × 102 | L001~L400 | 기본 학습 데이터. `Lot_ID` 컬럼 있음 |
| test.CSV | 1,000 × 102 | L401~L440 | 홀드아웃. `Lot_ID` 컬럼 있음 |
| mentorship_dataset_final.CSV | 10,000 × 112 | L001~L400 | `_Config` 대신 `_EQ` 컬럼명 사용, `Lot_ID` 없음(`Lot_Wafer_ID`에서 파싱) |
| mentorship_dataset_v7_killing_event.csv | 10,000 × 189 | L001~L400 | `Lot_ID` 없음. 장비 관련 컬럼이 `Step{n}_EQ{m}`(EQ1/EQ2/EQ3…) 형태라 앱의 컬럼 파서가 인식하는 `_Config`/`_EQ` 단일 접미사 패턴과 달라 51개 전부 `unmapped`로 처리됨 — 스키마 인식 기준으로는 "Config 없음"이 맞지만, 원본 파일 자체에 장비 컬럼이 없는 것은 아님 |

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

타깃별로 독립된 `sklearn.ensemble.HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6, random_state=42)`를 학습합니다(`src/ml/pipeline.py`). 피처는 그 타깃에서 BH-FDR을 통과한 인자(들)의 원본값 + `_miss`(결측 여부) + U자 인자의 `_dev`(최적 중심으로부터의 절대편차)입니다. FDR 통과 인자가 없는 타깃은 모델을 만들지 않고 train 평균을 상수로 예측합니다.

`POST /api/train`은 업로드된 CSV 한 장을 받아 **그 파일 내부**에서 LOT 기준 85/15로 단일 분할(GroupShuffleSplit)하고, 남은 15%(내부 홀드아웃)로 평가합니다 — 별도로 준비된 test.CSV를 자동으로 붙여 평가하지 않습니다(응답 스키마의 `final_y_metrics.test`는 "내부 홀드아웃 분할"을 가리키는 키 이름이지, 번들 test.CSV를 의미하지 않습니다). 내장 train.CSV를 업로드해 학습했을 때의 내부 홀드아웃 성능(8,500행 학습 / 1,500행 평가):

| | Y1 | Y2 | Y3 | Y4 | Y5 | Y(최종) |
|---|---|---|---|---|---|---|
| R² | 0.085 | 0.031 | 0.533 | 0.020 | 0.182 | 0.175 |
| MAE | 1.152 | 1.906 | 0.910 | 1.100 | 0.303 | 2.739 |

**R² 값 자체는 성공 기준이 아닙니다.** 결측률이 구조적으로 높아 미계측 wafer는 예측할 정보가 없습니다. 이 수치는 구현이 깨지지 않았는지 확인하는 회귀 테스트 기준값입니다.

전처리 방식별 비교(모델 학습 화면의 `데이터 전처리` 카드, `scripts/benchmark.py`의 `GroupKFold(5)` out-of-fold 평가 기준, Y 최종 R²):

| 방식 | R² |
|---|---|
| A. 중앙값 대체 + 클리핑 | 0.114 |
| B. 전체 인자 + NaN 보존 | 0.146 |
| C. 선정 인자 + `_dev` + `_miss` (채택) | 0.177 |

전처리 선택이 알고리즘 선택보다 성능에 크게 기여합니다. `src/ml/ensemble.py`(GroupKFold 기반 후보 앙상블 선택)와 `src/ml/hybrid.py`의 `train_hybrid_multi_y`(HGBR-vs-RandomForest 비교), `src/ml/training.py`는 저장소에 남아 있지만 **`/api/train`에서는 호출되지 않고 테스트에서만 실행되는 미사용 모듈**입니다. LightGBM/XGBoost/CatBoost 등 다른 부스팅 라이브러리는 `requirements.txt`에 없어 이 저장소 안에서는 비교 자체가 재현되지 않습니다.

### 3. SPC 관리한계와 사전 알람

인자별로 **X 자신의 산포**에서 관리한계를 산출합니다(train.CSV 기준). Y를 참조하지 않습니다.

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

| 인자 | 타깃 | 권장 구간(train 기준) | 구간 내/전체 평균 대비 기대 개선 |
|---|---|---|---|
| Step28_R1 | Y1 | 54.7~61.5 | −18% |
| Step16_R1 | Y2 | 55.7~61.7 | −17% |
| Step1_D1 | Y3 | 3.0~9.0 | −20% |
| Step24_R1 | Y4 | 53.2~61.5 | −9% |
| Step18_R1 | Y5 | 52.4~59.7 | −27% |

test.CSV 기준 개선 권장 레코드는 254건이며(관리한계 이탈로 이미 알람에 잡힌 건은 중복 집계하지 않음), 등급이 강함·보통인 인자만 담습니다.

## 화면 구성

3개 탭. 좌측 접이식 사이드바 + 우측 SUNI AI 어시스턴트 패널. 첫 접속 시 두 패널 모두 펼쳐진 상태로 시작하며, 접힘/펼침 상태는 쿠키에 저장되어 다음 방문에도 유지됩니다.

- **모델 학습**(`/training`) — 데이터셋 선택/업로드, 학습 실행(비동기 Job 폴링), 타깃별 R²/MAE/ε² 통합 테이블, 데이터 전처리 방식 비교표, 전체 상관관계 히트맵, 인자 스크리닝 Pareto 차트 및 테이블
- **원인 분석**(`/root-cause`) — "원인 분석 실행" → 상관관계 히트맵 + 타깃(Y1~Y5)별 Pareto 상위 5개 + 산점도. 실행 완료 후 `JSON 보고서 저장` 버튼으로 p<0.05 인자·알람 전건을 담은 보고서를 다운로드할 수 있습니다(항상 전체 R+D+Config 인자 풀 기준)
- **사전 알람 로그**(`/alerts`) — 알람/정상/판정불가 집계 카드, 알람 목록(관리한계 이탈, `해설` 버튼으로 SUNI에게 질문), LOT별 알람 집계, 개선 권장 목록(권장 구간 이탈, `해설` 버튼)

데이터셋은 각 탭의 선택 UI에서 CSV를 업로드해 추가할 수 있고(`POST /api/datasets`), 내장 4종과 함께 목록에 표시됩니다.

산점도는 Spotfire 방식(원형 마커, Color By)의 커스텀 SVG 컴포넌트(`ScatterChart.tsx`)이며, SPC 관리한계선과 12구간 구간 평균 불량률 추세선·권장 구간 밴드를 함께 표시합니다. Pareto 차트(`ParetoChart.tsx`)와 상관관계 히트맵(`CorrelationHeatmap.tsx`)도 정밀한 클릭/호버 제어를 위해 커스텀 SVG로 구현했습니다. Config(범주형) 인자의 박스플롯만 Plotly를 사용합니다.

## LLM 연동 (SUNI 챗봇)

- Upstage Solar API(OpenAI 호환 스펙)를 스트리밍(SSE)으로 호출합니다(`api/routes/chat.py`)
- `report` 모드(예시: "분석 보고서 생성" 버튼, 메시지에 "보고서" 등 키워드 포함): 6개 섹션(요약/불량 유형별 소견/장비 구성 소견/관리 대역 제안/한계/확인 필요 사항)으로 구성된 공정 보고서를 생성합니다
- `chat` 모드(그 외 자유 입력): 3~5문장으로 답합니다. 알람·개선 권장 개별 건(예: 특정 wafer의 알람)에 대한 질문에도 해당 레코드를 근거로 답합니다
- 두 모드 모두 `/api/analysis/context`가 만드는 동일한 분석 결과 JSON(타깃별 1위 인자, 관리한계, 권장 구간, 챔버 교호작용, 알람·개선 권장 레코드, config 스크리닝, 한계)을 근거로 답하며, 시스템 프롬프트(`prompts/report_system.md`, `prompts/chat_system.md`)가 **숫자 생성 금지, 인과 표현 금지, "값을 조정하라" 같은 설정값 표현 금지**를 명시적으로 규정합니다
- `confidence`(신뢰도)는 LLM이 아니라 코드가 판정한 값을 그대로 따르게 합니다 — 판정 근거가 부족한 인자에 LLM이 임의로 관리 대역을 만들어내지 못하게 하는 장치입니다
- 원인 분석이 아직 실행되지 않은 상태의 질문은 LLM을 호출하지 않고 백엔드가 즉시 안내 메시지로 응답합니다
- 환경변수: `UPSTAGE_API_KEY`, `UPSTAGE_BASE_URL`(기본 `https://api.upstage.ai/v1`), `UPSTAGE_MODEL`(기본 `solar-pro3`). 키는 백엔드 프로세스에만 두며 프런트 번들에는 절대 노출하지 않습니다(`NEXT_PUBLIC_` 접두사 미사용)
- API 키가 설정되지 않은 환경에서는 `/api/chat`이 503과 안내 메시지를 반환하며 서버 자체는 정상 동작합니다

## 검토했으나 채택하지 않은 것

**"안 만든 것"이 아니라 "검증하고 배제한 것"입니다.**

- **LOT 단위 알람** — 최종 수율의 LOT 간/전체 분산 비(ICC 근사) 약 0.04로, 분산 대부분이 LOT 내부 wafer 간 차이입니다. test.CSV에서 알람 2건 이상 LOT의 평균 수율(89.41)이 알람 0건 LOT(89.06)보다 오히려 높아 LOT 단위 신호가 뚜렷하지 않습니다. wafer 단위로 설계했습니다.
- **관리한계 롤링 윈도우** — train 400 LOT을 100개씩 4구간으로 나눠 각각 관리한계를 산출해보면, 인자별로 구간 간 변동 폭이 대체로 1~2 수준(Step1_D1은 다소 크게 변동)으로 뚜렷한 시간 드리프트는 보이지 않았습니다.
- **Y의 Q1~Q3 구간에서 X 범위 역산** — 초기 구현이었으나 조건부 확률의 방향이 반대입니다(`P(X|Y)` 대신 `P(Y|X)`가 필요). X 자신의 분포 기반 SPC 관리한계로 교체 후 알람이 58장→19장으로 줄고 수율차는 −5.25%p→−6.14%p로 개선됐습니다.
- **알고리즘 교체(앙상블/RandomForest 비교)** — `src/ml/ensemble.py`, `src/ml/hybrid.py`에 구현이 남아 있지만 실제 학습 API 경로에서는 호출되지 않습니다. 피처가 타깃당 1~3개뿐이라 트리 앙상블이 찾을 상호작용이 거의 없고, 이득 대비 유지 비용이 크다고 판단했습니다. LightGBM/XGBoost/CatBoost 등과의 비교는 의존성에 포함돼 있지 않아 이 저장소에서 재현 가능한 형태로 존재하지 않습니다.
- **수율 예측 서비스(`/api/predict`) 및 SHAP 기여도 원인 분석** — 프로젝트 범위에서 제외했습니다. 통계적 원인 특정(ε² + BH-FDR + SPC)에 집중합니다. (`requirements.txt`에 `shap` 패키지가 남아 있지만 코드 어디에서도 import하지 않는 미사용 의존성입니다.)

## 한계

1. 이 분석은 **해당 인자가 계측된 wafer만** 대상으로 합니다. R은 전체의 15%, D는 5%만 계측됩니다. test.CSV 1,000장 중 상당수(선정 인자 기준 489장)는 선정 인자가 하나도 계측되지 않아 **판정 자체가 불가능**합니다.
2. ε²는 통계적 연관성이지 인과가 아닙니다. 공정 순서상 선행·후행 관계나 교락 인자는 반영되지 않았습니다.
3. Config에서 유의 인자가 검출되지 않은 것은 검출력 부족(MDE ε² 0.00261, 관측 최대 0.00271)이며, 영향이 없다는 뜻이 아닙니다.
4. 관리한계는 "평소와 다른가"를 판정하며 "수율이 좋은가"를 보장하지 않습니다.
5. `POST /api/train`의 성능 지표는 업로드 파일 내부 15% 홀드아웃 기준이며, 별도의 test.CSV로 일반화 성능을 확인한 것이 아닙니다.
6. 계측된 wafer와 미계측 wafer의 최종 수율 차이는 이 데이터셋에서 통계적으로 유의하지 않았습니다(R/D 계측이 하나도 없는 wafer 기준 t-검정 p=0.74). 계측 편향이 관측되지 않았다는 뜻이지, 다른 데이터셋에도 성립한다는 보장은 아닙니다.

## 구조

```text
api/            FastAPI 라우트(data/datasets/analysis/chat)·스키마·설정
src/analysis/   인자 스크리닝(ε², BH-FDR), 히트맵, SPC 관리한계, 권장 구간, 챗봇 컨텍스트용 통계, JSON 보고서
src/ml/         학습 파이프라인(HistGradientBoostingRegressor), 추론 메타데이터, 모델 저장
src/runtime/    데이터셋 레지스트리, 학습 Job, SQLite 이력 저장
frontend/       Next.js UI (app/training, app/root-cause, app/alerts, components/ai-panel)
prompts/        SUNI 챗봇 시스템 프롬프트(report_system.md, chat_system.md)
tests/          Python 테스트(pytest)
scripts/        오프라인 벤치마크(전처리 방식 비교)
config/         결측 스키마·알람 severity·전처리 정책 YAML
```

`src/analytics/`, `src/automation/`은 `__init__.py`만 있는 빈 자리표시자 패키지입니다. `src/ml/training.py`, `src/ml/ensemble.py`, `src/ml/hybrid.py`의 `train_hybrid_multi_y`는 저장소에 남아 있지만 API 경로에서는 사용되지 않고 테스트에서만 실행됩니다.

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

알림 연동(Slack/Telegram/Gmail)을 쓰려면 아래 환경변수도 추가합니다(`.env.example` 참고, 전부 선택값입니다). 설정하지 않으면 해당 채널은 UI에서 "연결하기"를 눌러도 실제 발송은 되지 않습니다.

```env
TELEGRAM_BOT_TOKEN=
TELEGRAM_BOT_USERNAME=
SMTP_HOST=
SMTP_PORT=587
SMTP_USER=
SMTP_PASSWORD=
SMTP_FROM_EMAIL=
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
- 모델: `GET /api/models`, `GET /api/models/{model_id}`, `GET /api/models/{model_id}/references`, `DELETE /api/models/{model_id}`, `GET /api/models/performance`, `GET /api/model/latest`
- 인자 스크리닝: `GET /api/screening/pareto`, `GET /api/screening/heatmap`, `GET /api/screening/scatter`, `GET /api/screening/scatter/categorical`
- SPC/알람/권장: `GET /api/control-ranges`, `GET /api/alarms`, `GET /api/alarms/summary`, `GET /api/recommendations`
- 분석 보고서: `GET /api/analysis/report`(다운로드용 JSON), `GET /api/analysis/context`(SUNI 챗봇 컨텍스트용, 같은 내용을 다른 응답 헤더로 제공)
- SUNI 챗봇: `POST /api/chat`(SSE 스트리밍, `mode: "report" | "chat"`)

## Railway 배포

`railway.json`의 시작 명령:

```text
uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1
```

Healthcheck는 `/health`를 사용합니다. Railway 무료 티어(512MB RAM / 1vCPU)를 기준으로 업로드 상한을 20MB로 두고 `--workers 1`을 유지합니다(`config/upload_limits.yaml`). 배포 환경에는 최소 `FRONTEND_ORIGINS`, SUNI 챗봇을 쓸 경우 `UPSTAGE_API_KEY` 등을 설정해야 합니다. 로컬 검증은 Railway의 실제 RAM, OOM, 499/502 발생 여부를 대신하지 않으므로 운영 배포 후 별도 확인이 필요합니다.

## 검증

```powershell
python -m pytest -q
Set-Location frontend
npm.cmd run lint
npm.cmd run build
```

원본 학습 CSV와 실제 운영 환경이 없는 경우, 테스트 fixture 결과를 실제 모델 성능이나 Railway 안정성으로 표현하지 않습니다.
