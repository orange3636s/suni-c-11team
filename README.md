# 제조 공정 불량 예측 및 원인 분석 AI

반도체 제조 공정 데이터를 검증하고 전처리하여 수율 위험 예측과 불량
원인 후보 분석을 지원하는 프로젝트다. 기존 SOP 알람을 대체하지 않고
공정 엔지니어의 의사결정을 지원하는 것을 목표로 한다.

## 기술 구성

- 프런트엔드: Next.js, React, TypeScript, ESLint
- 백엔드: FastAPI
- 데이터 처리: pandas, NumPy
- 설정: YAML

## 프로젝트 구조

```text
frontend/   Next.js 프런트엔드
api/        FastAPI 백엔드
src/        데이터 검증 및 전처리
config/     데이터 스키마 및 전처리 설정
data/       공정 데이터
models/     모델 파일
tests/      Python 테스트
docs/       프로젝트 문서
workflows/  자동화 워크플로
```

## 환경변수 설정

프런트엔드에서 FastAPI 주소를 설정하려면 예제 파일을 복사해
`frontend/.env.local`을 만든다.

```bash
cd frontend
copy .env.local.example .env.local
```

macOS 또는 Linux에서는 다음 명령을 사용한다.

```bash
cp .env.local.example .env.local
```

기본 설정:

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

운영 환경에서 추가 CORS origin이 필요하면 백엔드 실행 환경에
쉼표로 구분한 값을 설정한다.

```env
FRONTEND_ORIGINS=https://example.com,https://dashboard.example.com
```

## 실행 방법

### 백엔드 실행

저장소 루트에서 다음 명령을 실행한다.

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload
```

### 프런트엔드 실행

Node.js 20.9 이상이 필요하다.

```bash
cd frontend
npm install
npm run dev
```

## 로컬 주소

- Next.js: http://localhost:3000
- 데이터 업로드 페이지: http://localhost:3000/upload
- 모델 학습 페이지: http://localhost:3000/training
- 수율 예측 페이지: http://localhost:3000/prediction
- 원인 분석 페이지: http://localhost:3000/root-cause
- 분석 보고서 페이지: http://localhost:3000/report
- 자동화 상태 페이지: http://localhost:3000/automation
- FastAPI: http://127.0.0.1:8000
- Swagger: http://127.0.0.1:8000/docs

## CSV 검증 및 전처리

Next.js의 `/upload` 페이지에서 CSV 파일을 선택하거나 드래그 앤 드롭한 뒤
검증 또는 전처리를 실행할 수 있다. 브라우저는 `NEXT_PUBLIC_API_BASE_URL`에
설정된 FastAPI로 파일을 전송하며, 업로드 파일은 서버 저장소에 영구 저장하지
않고 메모리에서 처리한다.

- 검증 API: `POST /api/validate`
- 전처리 API: `POST /api/preprocess`
- 요청 형식: `multipart/form-data`
- 파일 필드명: `file`
- 허용 형식: `.csv`
- 최대 파일 크기: 20MB
- 지원 인코딩: `utf-8-sig`, `utf-8`, `cp949`
- 전처리 미리보기: 최대 10행

로컬 실행 순서는 다음과 같다.

1. 저장소 루트에서 `pip install -r requirements.txt`로 Python 의존성을 설치한다.
2. 저장소 루트에서 `uvicorn api.main:app --reload`로 FastAPI를 실행한다.
3. `frontend/.env.local`에 `NEXT_PUBLIC_API_BASE_URL`을 설정한다.
4. `frontend` 디렉터리에서 `npm install` 후 `npm run dev`를 실행한다.
5. 브라우저에서 http://localhost:3000/upload 로 이동한다.

## 머신러닝 학습

`/training` 페이지에서 CSV와 목표 변수를 선택하면 검증과 기존 전처리를
먼저 실행한 뒤 회귀 모델을 비교한다. 기본 목표 변수는 최종 수율 `Y`이며,
동일한 구조로 `Y1`부터 `Y10`까지 선택할 수 있다.

- 학습 API: `POST /api/train`
- 요청 필드: `file`, `target`
- 기본 target: `Y`
- 지원 target: `Y`, `Y1`~`Y10`
- 모델 후보: `DummyRegressor`, `Ridge`, `RandomForestRegressor`,
  `HistGradientBoostingRegressor`
- 최적 모델 기준: Validation RMSE 오름차순, 동률이면 Validation R² 내림차순
- 평가 지표:
  - R²: 목표값 변동을 모델이 설명하는 비율
  - RMSE: 큰 오차에 더 큰 가중치를 주는 평균 오차
  - MAE: 예측값과 실제값 차이의 절댓값 평균

데이터는 먼저 전체의 20%를 Test로 분리하고, 나머지 80%에서 20%를
Validation으로 분리해 기본적으로 Train 64%, Validation 16%, Test 20%를
사용한다. `Lot_Wafer_ID`에서 Lot을 추출할 수 있으면 같은 Lot의 wafer가 서로
다른 데이터셋에 포함되지 않도록 그룹 단위로 분리한다. Lot 추출이 불가능하거나
그룹 수가 부족하면 `random_state=42`인 random split을 사용하고 학습 결과에
경고를 포함한다.

선택된 모델은 `models/`에 joblib 파일로 저장되며 같은 이름의 JSON
메타데이터가 함께 생성된다. 메타데이터에는 target, 모델명, feature 컬럼,
생성 시각, 데이터셋별 지표, 분리 방식 및 scikit-learn 버전이 포함된다.
업로드한 원본 CSV는 저장하지 않는다.

## 저장 모델 예측

학습은 CSV로 모델을 생성하고 평가하는 과정이며, 예측은 저장된 모델과
메타데이터를 불러와 새로운 CSV의 Wafer별 목표값을 계산하는 과정이다.
추론 CSV도 기존 검증과 전처리를 거치며, 메타데이터에 저장된 학습 feature
목록과 순서를 그대로 복원한다. 학습 당시 필요했던 feature 컬럼이 없으면
예측하지 않으며, 새로 추가된 R/D/EQ feature는 모델 입력에서 제외하고 경고로
알린다.

- 모델 목록 API: `GET /api/models`
- 예측 API: `POST /api/predict`
- 예측 CSV 다운로드 API: `POST /api/predict/download`
- 예측 페이지: http://localhost:3000/prediction
- 요청 필드: `file`, `model_id`, `warning_threshold`, `danger_threshold`
- 화면 응답 예측 행 제한: 최대 5,000행

기본 `Y` 위험 기준은 다음과 같다.

- 정상: `predicted_Y >= 95`
- 주의: `90 <= predicted_Y < 95`
- 위험: `predicted_Y < 90`

주의 기준은 위험 기준보다 커야 한다. 신규 CSV에 실제 target이 없어도
예측할 수 있으며, target이 있으면 R², RMSE, MAE와 행별 오차를 함께 계산한다.
원본 업로드 CSV는 예측 과정에서도 영구 저장하지 않는다.

## SHAP 기반 원인 후보 분석

`/root-cause` 페이지에서 저장된 모델과 CSV를 선택하면 실제 SHAP 값을
계산해 모델 예측에 영향을 준 feature를 보여준다. 학습 파이프라인의 전처리를
그대로 적용하고 변환된 feature 이름과 순서를 유지한다.

- 원인 분석 API: `POST /api/explain`
- 전체 중요도 CSV API: `POST /api/explain/download`
- 요청 필드: `file`, `model_id`, `max_rows`, `top_n`,
  `per_wafer_top_n`
- 기본 최대 분석 행: 500행
- 최대 허용 분석 행: 1,000행
- 샘플링 순서: 위험 → 주의 → 정상
- Tree 모델: `shap.Explainer` 또는 `TreeExplainer`
- 선형 모델: `shap.Explainer` 또는 `LinearExplainer`
- SHAP을 사용할 수 없는 모델: 모델 독립형 feature perturbation 방식

`Y`는 수율이 낮아지는 방향을 위험 기여로, `Y1`~`Y10`은 값이 높아지는
방향을 위험 기여로 계산한다. 결과에는 전체 feature 중요도, Step별 집계,
R/D/EQ 유형별 집계, 설비 집계 및 Wafer별 악화·개선 기여 변수가 포함된다.
`mean_abs_shap`과 `mean_harmful_contribution`은 분석 행 전체의 feature별
평균이며, Step·유형별 필드는 해당 그룹에 속한 feature 평균값들의 합계다.

SHAP 결과는 인과관계를 증명하지 않고 모델 예측의 기여도를 설명한다.
Test R²가 낮거나 Validation/Test 성능 차이가 큰 모델은 화면과 API 응답에
품질 경고를 표시하므로 공정 조치 전에 반드시 현장 검증이 필요하다.

## 자동 분석 보고서

`/report` 페이지에서는 CSV와 저장 모델만 선택하면 별도의 예측 또는 원인
분석 페이지 실행 없이 다음 흐름을 자동으로 수행한다.

```text
CSV 검사 → 전처리 → 모델 로드 → 수율 예측 → 위험 분류
→ SHAP 분석 → 주요 후보 요약 → 규칙 기반 보고서 생성
```

예측과 SHAP 계산은 기존 추론 및 설명 모듈을 재사용하며 외부 LLM API를
사용하지 않는다.

- 보고서 JSON API: `POST /api/report`
- HTML 다운로드 API: `POST /api/report/download`
- 요청 필드: `file`, `model_id`, `warning_threshold`,
  `danger_threshold`, `max_rows`, `top_n`
- 기본 SHAP 분석 행: 500행
- 위험 Wafer 목록: 위험 → 주의 → 정상 순서이며 같은 등급에서는 예측값
  오름차순

Executive Summary에는 전체 Wafer 수, 평균 예측 수율, 정상·주의·위험 수,
주의·위험 비율, 모델 Test 지표, 실제 SHAP 분석 행 수와 샘플링 여부가
포함된다. Key Findings는 계산된 위험 분류, 상위 Step, R/D/EQ 유형,
feature 및 LOT 집계만 사용해 3~7개의 규칙 기반 문장으로 생성한다.

Wafer 식별자에서 Lot을 추출할 수 있으면 LOT별 Wafer 수, 평균 예측값,
위험 분류 수, 위험 비율과 상위 SHAP 후보를 함께 제공한다. 식별자 일부에서
Lot을 추출하지 못하면 LOT 섹션을 생략하고 경고에 이유를 기록한다.

추천 조치는 설비 제어값 변경 지시가 아니라 엔지니어 검토 우선순위다.
위험 LOT, 상위 Step, 반복 탐지된 R/D/EQ 유형과 모델 품질을 기준으로
공정 로그·장비 이력·레시피 변경 내역 확인 또는 추가 학습을 권고한다.

다운로드 HTML에는 CSS를 문서 내부에 포함하므로 외부 CDN이나 인터넷 연결
없이 브라우저에서 열고 인쇄할 수 있다. 보고서의 SHAP 후보는 모델 예측
기여도이며 실제 불량의 직접 원인이나 인과관계를 확정하지 않는다.

## 현재 구현 범위

- CSV 업로드와 프론트엔드 파일 형식·크기 검사
- 기존 Python 로직을 이용한 데이터 검증
- 기존 Python 로직을 이용한 결측치 처리와 이상치 보정
- 검증 결과, 처리 전후 통계, 전처리 데이터 미리보기
- Lot 그룹 기반 Train/Validation/Test 분리
- 회귀 모델 비교, 평가 및 최적 모델 저장
- 모델 학습 결과와 모델별 비교 화면
- 저장 모델 목록 조회와 신규 CSV 수율 예측
- Wafer 위험 분류, 예측 결과 필터·검색·정렬 및 CSV 다운로드
- 실제 SHAP 기반 전체·공정 단계·파라미터 유형·Wafer별 원인 후보 분석
- 원인 분석 중요도 차트 및 CSV 다운로드
- 예측·위험 분류·SHAP 결과를 결합한 규칙 기반 자동 분석 보고서
- 위험 Wafer 및 LOT 집계와 독립형 HTML 보고서 다운로드
- n8n용 통합 분석 API와 import 가능한 위험 분기 워크플로우

실제 Slack credential 연결과 Vercel·Render 배포는 아직 완료하지 않았다.

## n8n 통합 자동화

`POST /api/analyze`는 CSV 검증과 전처리부터 모델 예측, 위험 분류, SHAP
분석 및 규칙 기반 보고서 요약까지 한 번에 실행한다. 기존 predict, explain,
report 내부 함수를 재사용하며 업로드 CSV를 영구 저장하지 않는다.

- 통합 분석 API: `POST /api/analyze`
- 요청 형식: `multipart/form-data`
- 요청 필드: `file`, `model_id`, `warning_threshold`,
  `danger_threshold`, `max_rows`, `top_n`, `per_wafer_top_n`,
  `include_report`
- n8n Webhook path: `manufacturing-ai-analysis`
- 워크플로우 파일:
  `workflows/n8n_manufacturing_ai_workflow.json`
- 상세 설정 문서: `docs/N8N_WORKFLOW_GUIDE.md`

n8n의 FastAPI 주소는 워크플로우에 하드코딩하지 않고 다음 환경변수로
설정한다.

```env
FASTAPI_BASE_URL=http://127.0.0.1:8000
```

n8n을 Docker에서 실행하면 일반적으로
`http://host.docker.internal:8000`을 사용한다. 환경변수에는 기본 서버
주소만 넣으며 `/api/analyze`는 HTTP Request 노드가 추가한다.

Alert 분기 정책:

- 위험 Wafer가 하나 이상이면 `danger`, 알림 필요
- 위험은 없고 주의 Wafer가 하나 이상이면 `warning`, 알림 필요
- 위험과 주의가 모두 없으면 `normal`, 알림 없음

Test R² 저하, Test 지표 누락, DummyRegressor 및 SHAP fallback은 모델 품질
경고에 포함하지만 공정 위험 Alert를 강제로 발생시키지는 않는다.

Slack 노드에는 credential이나 운영 채널이 저장되어 있지 않다. n8n에서
`Slack Alert - Danger`, `Slack Alert - Warning` 노드에 credential을
연결하고 `SLACK_CHANNEL_ID` 또는 실제 채널을 설정해야 한다. Slack 전송
실패가 분석 전체를 실패시키지 않도록 두 노드에 실패 계속 처리를 적용했다.

Next.js `/automation` 페이지에서는 FastAPI 상태와
`NEXT_PUBLIC_N8N_WEBHOOK_URL` 설정 여부를 확인할 수 있다.

```env
NEXT_PUBLIC_N8N_WEBHOOK_URL=http://localhost:5678/webhook/manufacturing-ai-analysis
```

현재 n8n 워크플로우와 통합 API는 구현되어 있지만 실제 Slack 알림에는
credential 설정이 필요하다. 운영 전에는 HTTPS Webhook URL, API 인증,
n8n 실행 데이터 보존 정책과 Vercel·Render 또는 별도 배포 환경 설정이
추가로 필요하다.

## 운영 배포 구성

운영 배포는 다음 구조를 기준으로 한다.

```text
Vercel Next.js (frontend/)
  → Render FastAPI (저장소 루트, api.main:app)
  → models/ 기반 예측·SHAP·보고서

n8n Cloud 또는 별도 n8n
  → Render POST /api/analyze
  → n8n Slack credential을 이용한 알림
```

백엔드 환경변수는 `.env.example`, 프런트 예시는
`frontend/.env.local.example`을 참고한다. 실제 값은 배포 플랫폼의
환경변수 화면에 설정하고 저장소에 넣지 않는다.

- FastAPI: `APP_ENV`, `FRONTEND_ORIGINS`, `MODEL_DIR`,
  `MAX_UPLOAD_SIZE_MB`, `LOG_LEVEL`
- Next.js: `NEXT_PUBLIC_API_BASE_URL`, `NEXT_PUBLIC_N8N_WEBHOOK_URL`
- n8n: `FASTAPI_BASE_URL`, 필요 시 `SLACK_CHANNEL_ID`

Render의 Root Directory는 저장소 루트이고 시작 명령은
`uvicorn api.main:app --host 0.0.0.0 --port $PORT`이다. Vercel의 Root
Directory는 `frontend`이다. 자세한 절차는 다음 문서를 따른다.

- `docs/VERCEL_DEPLOYMENT_GUIDE.md`
- `docs/RENDER_DEPLOYMENT_GUIDE.md`
- `docs/N8N_WORKFLOW_GUIDE.md`
- `docs/SLACK_INTEGRATION_GUIDE.md`
- `docs/DEPLOYMENT_CHECKLIST.md`

## 모델 파일 운영 정책

현재 초기 모델 `.joblib`은 약 540KB이고 JSON 메타데이터에는 모델 설정과
feature 목록이 들어 있다. 파일 크기가 작고 현재 확인된 비밀값은 없어
이 초기 모델 한 쌍만 배포용 저장소에 포함하는 정책이다. 새 학습 모델은
`.gitignore`로 기본 제외한다.

Render의 로컬 파일 시스템은 영구 모델 저장소로 간주하지 않는다. 운영
중 새로 학습한 모델은 재배포나 인스턴스 재생성 때 유실될 수 있다.
장기적으로는 Persistent Disk 또는 외부 Object Storage를 연결해야 한다.
모델을 새로 저장소에 포함하기 전에는 크기, 학습 데이터 유출 가능성,
직렬화 파일 신뢰성을 다시 검토한다.
