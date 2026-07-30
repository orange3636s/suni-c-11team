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
CORS_ALLOWED_ORIGINS=https://example.com,https://dashboard.example.com
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

## 현재 구현 범위

- CSV 업로드와 프론트엔드 파일 형식·크기 검사
- 기존 Python 로직을 이용한 데이터 검증
- 기존 Python 로직을 이용한 결측치 처리와 이상치 보정
- 검증 결과, 처리 전후 통계, 전처리 데이터 미리보기
- Lot 그룹 기반 Train/Validation/Test 분리
- 회귀 모델 비교, 평가 및 최적 모델 저장
- 모델 학습 결과와 모델별 비교 화면

저장된 모델을 이용한 수율 예측, SHAP 기반 원인 분석, n8n 및 Slack 알림
기능은 아직 구현하지 않았다.
