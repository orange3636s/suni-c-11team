# SUNI 11팀 — 반도체 공정 원인 분석 & 사전 알람

반도체 제조 공정 데이터에서 **Y1~Y5 불량률과 통계적으로 연관된 공정 인자를 특정하고, SPC 관리한계 기반으로 이탈 wafer를 사전 탐지하는** FastAPI + Next.js 애플리케이션입니다.

이 프로젝트의 산출물은 수율 예측 정확도가 아니라 **"어떤 step의 어떤 인자가 어떤 불량과 관계있는가"의 통계적 특정**입니다. Y1~Y5를 학습하는 회귀 모델이 포함되어 있지만, 이는 스크리닝 결과의 구현 정확성을 확인하는 회귀 테스트 기준값이지 제품 기능(수율 예측 서비스)이 아닙니다.

## 데이터 특성

| 항목 | 값 |
|---|---|
| train.CSV | 10,000행 = 400 LOT × 25 wafer (L001~L400) |
| test.CSV | 1,000행 = 40 LOT × 25 wafer (L401~L440) |
| 컬럼 | 102개 — Config 30, R 48, D 10, 식별자 3(`Lot_Wafer_ID`/`Lot_ID`/`Wafer_Slot`), 타깃 11(`Y1`~`Y5`, `Y6`~`Y10`, `Y`) |
| 결측률(평균) | R 85.0%, D 95.0%, Config 0% |
| 관계식 | `Y = 100 − (Y1+Y2+Y3+Y4+Y5)` |

**결측은 랜덤이 아니라 계측 샘플링 구조입니다.** R 인자 하나의 결측률이 85%라는 것은, LOT 25장 중 그 인자가 계측된 wafer가 평균 `25 × (1 − 0.85) ≈ 3.75`장뿐이라는 뜻입니다. 특정 R 인자(`Step28_R1`)의 LOT 내 표준편차(3.82)가 LOT 간(LOT 평균들의) 표준편차(2.95)보다 커서, LOT 평균으로 대체해도 정보가 거의 생기지 않습니다.

## 설계 원칙

각 항목에 근거 수치를 병기합니다.

- **결측을 채우지 않는다.** NaN을 유지하고 `{feature}_miss` 이진 태그를 별도 피처로 둡니다. 중앙값으로 대체하면 상관관계가 무너집니다 — `Step1_D1`–`Y3` Spearman ρ가 pairwise 기준 **0.843**에서, 전체 행에 중앙값 대체 후에는 **−0.041**로 바뀝니다.
- **양측 clipping을 하지 않는다.** R–Y 관계가 U자형인 경우가 많아 꼬리 값 자체가 신호입니다.
- **인자 선정은 편향보정 ε²(epsilon-squared) 기준.** 선형 Pearson r²은 U자 관계에서 좌우가 상쇄되어 과소평가합니다 — `Step24_R1`–`Y4`: 선형 r² **0.012** vs ε² **0.073**.
- **평가 분할은 LOT 단위 GroupShuffleSplit.** 같은 LOT의 wafer가 train/평가 양쪽에 걸치면 정보가 샙니다(`sklearn.model_selection.GroupShuffleSplit(groups=Lot_ID)`, 15% 홀드아웃, `random_state=42`). K-fold 교차검증은 오프라인 벤치마크 스크립트(`scripts/benchmark.py`, `GroupKFold(5)`)에서만 쓰이며, 실제 학습 API는 이 단일 분할만 사용합니다.
- **관리한계는 X축 SPC 방식.** Y의 분위수 구간에서 X 범위를 역산하지 않고, 인자 자신의 분포(mean/std/Q1/Q3)에서 산출합니다.

## 분석 파이프라인

### 1. 인자 스크리닝

88개 인자(R 48 + D 10 + Config 30) × 타깃 5개(Y1~Y5)에 대해 편향보정 ε²을 산출합니다. 연속형(R/D)은 분위수 8구간 ANOVA, 범주형(Config)은 카테고리 ANOVA를 사용합니다. 타깃별로 BH-FDR(Benjamini-Hochberg)을 적용해 q값을 계산합니다. 기여율(`contribution_pct`) 분모는 **해당 타깃의 전체 88개 인자 풀**입니다.

train.CSV 기준 타깃별 1위 인자:

| 타깃 | 1위 인자 | ε² | 기여율 | p값 | 관측수 |
|---|---|---|---|---|---|
| Y1 | Step28_R1 | 0.192 | 63.9% | <0.001 | 1,492 |
| Y2 | Step16_R1 | 0.159 | 63.1% | <0.001 | 1,470 |
| Y3 | Step1_D1 | 0.660 | 92.6% | <0.001 | 479 |
| Y4 | Step24_R1 | 0.073 | 41.2% | <0.001 | 1,512 |
| Y5 | Step18_R1 | 0.287 | 76.7% | <0.001 | 1,479 |

**Config 30개는 어떤 타깃에서도 BH-FDR을 통과하지 못했습니다** (가장 높은 ε²도 0.003 수준). 장비(Config 카테고리)당 표본이 평균 278장(표준편차 49) 수준이라 ε² 0.01 미만 효과는 검출력이 부족합니다. 이는 "장비 영향이 없다"가 아니라 "현재 표본으로 검출되지 않는다"는 뜻입니다.

화면(원인 분석 탭, 학습 탭)에서는 BH-FDR 게이트로 인자를 걸러내지 않고 **4단계 신뢰도 배지**(강함 p<0.01 / 보통 0.01≤p<0.05 / 약함 0.05≤p<0.20 / 참고 p≥0.20)로 모든 인자를 표시합니다. q값은 계산되어 툴팁에 노출되지만 화면 표시 여부를 걸러내는 필터로는 쓰이지 않습니다. **단, 사전 알람 생성과 JSON 보고서 포함 여부는 p<0.05(강함·보통 등급)로 제한됩니다.**

### 2. 모델 학습 및 검증

타깃별로 독립된 `sklearn.ensemble.HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6, random_state=42)`를 학습합니다(`src/ml/pipeline.py`). 피처는 그 타깃에서 BH-FDR을 통과한 인자(들)의 원본값 + `_miss`(결측 여부) + U자 인자의 `_dev`(최적 중심으로부터의 절대편차)입니다. FDR 통과 인자가 없는 타깃은 모델을 만들지 않고 train 평균을 상수로 예측합니다.

`POST /api/train`은 업로드된 CSV 한 장을 받아 **그 파일 내부**에서 LOT 기준 85/15로 단일 분할(GroupShuffleSplit)하고, 남은 15%(내부 홀드아웃)로 평가합니다 — 별도로 준비된 test.CSV를 자동으로 붙여 평가하지 않습니다. 내장 train.CSV를 업로드해 학습했을 때의 내부 홀드아웃 성능(8,500행 학습 / 1,500행 평가):

| | Y1 | Y2 | Y3 | Y4 | Y5 | Y(최종) |
|---|---|---|---|---|---|---|
| R² | 0.085 | 0.031 | 0.533 | 0.020 | 0.182 | 0.175 |
| MAE | 1.152 | 1.906 | 0.910 | 1.100 | 0.303 | 2.739 |

**R² 값 자체는 성공 기준이 아닙니다.** 결측률이 구조적으로 높아 미계측 wafer는 예측할 정보가 없습니다. 이 수치는 구현이 깨지지 않았는지 확인하는 회귀 테스트 기준값입니다.

전처리 방식별 비교(`scripts/benchmark.py`, train.CSV 전체에 대한 `GroupKFold(5)` out-of-fold 평가, Y 최종 R²):

| 방식 | Y R² |
|---|---|
| A. 중앙값 대체 + R 1~99pct 클리핑 + Config 빈도 인코딩 | 0.118 |
| B. 전체 R+D 피처, NaN 보존(Config 제외) | 0.153 |
| C. 선정 인자 + `_dev` + `_miss` (현재 파이프라인) | 0.173 |

전처리 선택이 알고리즘 선택보다 성능에 크게 기여합니다. 다른 부스팅 라이브러리(LightGBM/XGBoost/CatBoost)와의 비교는 이 저장소의 의존성에 포함되어 있지 않아 재현 가능한 형태로 존재하지 않습니다 — 알고리즘 교체 효과를 주장하는 근거로 사용하지 않습니다.

`src/ml/ensemble.py`(GroupKFold 기반 후보 앙상블 선택)와 `src/ml/hybrid.py`의 `train_hybrid_multi_y`(HGBR-vs-RandomForest 비교, "학습 후 즉시 디스크 저장·메모리 해제" 샤드 패턴), `src/ml/training.py`는 저장소에 남아 있지만 **`/api/train`에서는 호출되지 않고 테스트에서만 실행되는 미사용 모듈**입니다.

### 3. SPC 관리한계 및 사전 알람

인자별로 **X 자신의 산포**에서 관리한계를 산출합니다(train.CSV 기준). Y를 참조하지 않습니다.

```
UCL(LCL) = Q3(Q1) ± 1.5 × IQR   ← 알람 기준(IQR×1.5)
mean ± 3σ, mean ± 6σ            ← 참조선(알람에는 쓰이지 않음)
```

단조 증가/감소 인자(예: `Step1_D1`)는 "나쁜 쪽" 한 방향에서만 알람이 발생합니다(단측). 관리한계선의 값이 해당 인자의 관측 범위 [min, max]를 벗어나면 그 선은 그리지 않습니다(축을 늘리거나 화살표로 표시하지 않음).

| 인자 | 정상 범위(train 기준) | 비고 |
|---|---|---|
| Step28_R1 | 49.9 ~ 69.9 | |
| Step16_R1 | 51.8 ~ 68.8 | |
| Step1_D1 | ~ 18.0 | 단조증가, 단측 상한 |
| Step24_R1 | 48.1 ~ 69.5 | |
| Step18_R1 | 47.6 ~ 68.4 | |

test.CSV(1,000장)에 적용한 결과: **알람 wafer 19장(1.9%), 알람군 최종 수율(83.15)이 무알람군(89.29)보다 6.14%p 낮습니다.** (알람은 wafer×인자 조합 단위로도 집계되며, 한 wafer가 두 인자에서 동시에 이탈하면 알람 레코드는 2건이 됩니다 — 개별 레코드 20건 / 고유 wafer 19장.)

기준 선택 근거(동일 5개 인자, test.CSV 적용):

| 기준 | 알람 wafer | 수율차 |
|---|---|---|
| Q1 ~ Q3 | 279장 (27.9%) | −2.39%p |
| IQR 1.0배 | 42장 (4.2%) | −4.11%p |
| **IQR 1.5배 (채택)** | **19장 (1.9%)** | **−6.14%p** |
| ±3σ | 15장 (1.5%) | −6.05%p |
| ±6σ | 0장 | 전부 데이터 범위 밖 |

±3σ·±6σ·Q1/Q3는 산점도 참조선으로만 표시하고 알람 판정에는 사용하지 않습니다.

## 화면 구성

3개 탭. 좌측 접이식 사이드바 + 우측 SUNI AI 어시스턴트 패널(챗봇 백엔드는 아직 연결되지 않은 스텁입니다). 첫 접속 시 두 패널 모두 펼쳐진 상태로 시작하며, 접힘/펼침 상태는 쿠키에 저장되어 다음 방문에도 유지됩니다.

- **모델 학습** (`/training`) — 데이터셋 선택/업로드, 학습 실행(비동기 Job 폴링), 타깃별 R²/MAE/ε², 전처리 방식 비교표, 인자 스크리닝 Pareto 차트 및 테이블(BH-FDR 탈락 인자도 펼쳐볼 수 있음)
- **원인 분석** (`/root-cause`) — "원인 분석 실행" → 상관관계 히트맵 + 타깃(Y1~Y5) × 종류(전체/R/D/Config) 20개 조합의 Pareto + 산점도. 실행 완료 후 `JSON 보고서 저장` 버튼으로 p<0.05 인자·알람 전건을 담은 보고서를 다운로드할 수 있습니다(항상 전체 인자 기준, 화면의 R/D/Config 선택과 무관).
- **사전 알람 로그** (`/alerts`) — 알람/정상/판정불가 집계 카드, 알람 테이블(원인 분석 산점도로 딥링크), LOT별 알람 집계

데이터셋은 각 탭의 선택 UI에서 CSV를 업로드해 추가할 수 있고(`POST /api/datasets`), 내장 train.CSV/test.CSV와 함께 목록에 표시됩니다.

산점도는 Spotfire 방식(원형 마커, Color By)의 커스텀 SVG 컴포넌트(`ScatterChart.tsx`)이며, SPC 관리한계선(평균/Q1·Q3/IQR×1.5/±3σ/±6σ, 데이터 범위 밖인 선은 숨김)과 12구간 구간 평균 불량률 추세선을 함께 표시합니다. Pareto 차트(`ParetoChart.tsx`)와 상관관계 히트맵(`CorrelationHeatmap.tsx`)도 동일한 이유(정밀한 클릭/호버 제어)로 Plotly 대신 커스텀 SVG로 구현되어 있습니다. Config(범주형) 인자의 박스플롯만 Plotly를 사용합니다.

## 검토했으나 채택하지 않은 것

- **LOT 단위 알람** — 최종 수율의 LOT 간/전체 분산 비(ICC 근사) 약 0.04로, 분산 대부분이 LOT 내부 wafer 간 차이입니다. test.CSV에서 알람 2건 이상 LOT의 평균 수율(89.41)이 알람 0건 LOT(89.06)보다 오히려 높아 LOT 단위 신호가 뚜렷하지 않습니다. wafer 단위로 설계했습니다.
- **관리한계 롤링 윈도우** — train 400 LOT을 100개씩 4구간으로 나눠 각각 관리한계를 산출해보면, 인자별로 구간 간 변동 폭이 대체로 1~2 수준(Step1_D1은 다소 크게 변동)으로 뚜렷한 시간 드리프트는 보이지 않았습니다. 업로드되는 실 운영 데이터에서는 필요할 수 있습니다.
- **Y의 Q1~Q3 구간에서 X 범위 역산** — 초기 구현이었으나 조건부 확률의 방향이 반대입니다(`P(X|Y)` 대신 `P(Y|X)`가 필요). X 자신의 분포 기반 SPC 관리한계로 교체 후 알람이 58장→19장으로 줄고 수율차는 −5.25%p→−6.14%p로 개선됐습니다.
- **알고리즘 교체(앙상블/RandomForest 비교)** — `src/ml/ensemble.py`, `src/ml/hybrid.py`에 구현이 남아 있지만 실제 학습 API 경로에서는 호출되지 않습니다. 피처가 타깃당 1~3개뿐이라 트리 앙상블이 찾을 상호작용이 거의 없고, 이득 대비 유지 비용이 크다고 판단했습니다.
- **수율 예측 서비스(`/api/predict`) 및 SHAP 기여도 원인 분석** — 프로젝트 범위에서 제외했습니다. 통계적 원인 특정(ε² + BH-FDR + SPC)에 집중합니다.

## 한계

1. 이 분석은 **해당 인자가 계측된 wafer만** 대상으로 합니다. R은 전체의 15%, D는 5%만 계측됩니다. test.CSV 1,000장 중 상당수(선정 인자 기준 489장)는 선정 인자가 하나도 계측되지 않아 **판정 자체가 불가능**합니다.
2. ε²는 통계적 연관성이지 인과가 아닙니다. 공정 순서상 선행·후행 관계나 교락 인자는 반영되지 않았습니다.
3. Config에서 유의 인자가 검출되지 않은 것은 검출력 부족이며, 영향이 없다는 뜻이 아닙니다.
4. 관리한계는 "평소와 다른가"를 판정하며 "수율이 좋은가"를 보장하지 않습니다. 이 데이터에서 잘 작동하는 것은 U자 관계의 최적점이 분포 중심 근처에 있기 때문입니다.
5. `POST /api/train`의 성능 지표는 업로드 파일 내부 15% 홀드아웃 기준이며, 별도의 test.CSV로 일반화 성능을 확인한 것이 아닙니다.

## 구조

```text
api/            FastAPI 라우트(data/datasets/analysis)·스키마·설정
src/analysis/   인자 스크리닝(ε², BH-FDR), 히트맵, SPC 관리한계, 산점도, JSON 보고서
src/ml/         학습 파이프라인(HistGradientBoostingRegressor), 추론 메타데이터, 모델 저장
src/runtime/    데이터셋 레지스트리, 학습 Job, SQLite 이력 저장
frontend/       Next.js UI (app/training, app/root-cause, app/alerts)
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

`frontend/.env.local.example`을 `frontend/.env.local`로 복사합니다. 실제로 프런트엔드가 읽는 값은 `NEXT_PUBLIC_API_BASE_URL` 하나뿐입니다.

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

기본 주소는 Next.js `http://localhost:3000`, FastAPI `http://127.0.0.1:8000`, Swagger `http://127.0.0.1:8000/docs`입니다.

## API

- 상태 확인: `GET /health`, `GET /ready`, `GET /`
- 데이터셋: `GET /api/datasets`, `POST /api/datasets`(업로드), `DELETE /api/datasets/{id}`, `GET /api/datasets/{id}/download`, `GET /api/datasets/{id}/schema`
- CSV 검증·전처리(현재 UI에서는 호출하지 않는 독립 엔드포인트): `POST /api/validate`, `POST /api/preprocess`
- 학습: `POST /api/train`(동기), `POST /api/train/jobs`(비동기) + `GET /api/train/jobs/{job_id}`
- 모델: `GET /api/models`, `GET /api/models/{model_id}`, `GET /api/models/{model_id}/references`, `DELETE /api/models/{model_id}`, `GET /api/models/performance`, `GET /api/model/latest`
- 인자 스크리닝: `GET /api/screening`, `GET /api/screening/pareto`, `GET /api/screening/heatmap`, `GET /api/screening/scatter`, `GET /api/screening/scatter/categorical`
- SPC/알람: `GET /api/control-ranges`, `GET /api/alarms`, `GET /api/alarms/summary`
- 분석 보고서: `GET /api/analysis/report`(p<0.05 인자 + 알람 전건을 담은 JSON, `ensure_ascii=False`/부동소수점 4자리 반올림)

## Railway 배포

`railway.json`의 시작 명령:

```text
uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1
```

Healthcheck는 `/health`를 사용합니다. 배포 환경에는 최소 `FRONTEND_ORIGINS`와 영구 Volume 관련 경로(`RAILWAY_VOLUME_MOUNT_PATH`)를 설정해야 합니다. 로컬 검증은 Railway의 실제 RAM, OOM, 499/502 발생 여부를 대신하지 않으므로 운영 배포 후 별도 확인이 필요합니다.

## 검증

```powershell
python -m pytest -q
Set-Location frontend
npm.cmd run lint
npm.cmd run build
```

원본 학습 CSV와 실제 운영 환경이 없는 경우, 테스트 fixture 결과를 실제 모델 성능이나 Railway 안정성으로 표현하지 않습니다.
