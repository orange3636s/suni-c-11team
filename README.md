# 제조 공정 불량 예측 & 원인 분석 AI

반도체 제조 공정 CSV에서 Y1~Y5 불량률을 학습하고, 예측값으로 최종 수율 Y를 계산한 뒤 Lot·Wafer·R·D·Config 관점의 원인 후보를 분석하는 FastAPI/Next.js 애플리케이션입니다. 분석 결과는 모델의 통계적 관계와 SHAP 기여도이며 인과관계를 확정하지 않습니다.

## 핵심 모델 규칙

- 학습 Target: Y1, Y2, Y3, Y4, Y5만 사용
- 최종 수율: `clip(100 - sum(max(predicted_Y1..Y5, 0)), 0, 100)`
- Y: 학습 Target이 아니라 최종 성능 평가용
- Y6~Y10: 기술 통계와 Lot/Wafer/관계 분석용이며 모델 Artifact를 만들지 않음
- 분할: `random_state=42`, Lot 기준 Train 70% / Validation 15% / Test 15%; 수율 분위수 5개에서 시작해 필요 시 2개까지 축소하거나 안전한 Lot fallback 적용
- Config: 문자열을 분해하지 않고 Train 빈도만으로 Frequency Encoding; 결측은 Train 최빈값, 미지 Category는 0, 출력은 `float32`
- R: Train 중앙값 결측 대체 후 Train 1~99 percentile 양측 clipping
- D: Train 중앙값 결측 대체 후 Train 상위 99.9 percentile만 clipping; 0 보존
- 모델 선택: HistGradientBoostingRegressor를 먼저 학습하고, Validation 성능 조건을 만족하지 못한 Target만 RandomForestRegressor를 비교
- 선택 기준: Validation RMSE; 차이가 1% 이내면 더 작은 모델을 우선
- 위험 기준: 정상 `Y >= 90`, 주의 `85 <= Y < 90`, 위험 `Y < 85`

학습은 Y1부터 Y5까지 순차 실행하며 각 Target 모델은 디스크 shard로 저장한 뒤 메모리에서 해제합니다. 추론도 Target 모델을 하나씩 로드·예측·해제합니다. Test 데이터는 최종 선택 이후 한 번만 평가합니다.

## 구조

```text
api/          FastAPI 라우트·스키마·설정
src/ml/       자동 학습, 추론, 설명
src/analytics Lot 및 관계·통계 분석
src/runtime/  이력 저장, 초기화, 일회성 Migration
frontend/     Next.js UI
tests/        Python 테스트
```

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

`frontend/.env.local.example`을 `frontend/.env.local`로 복사하고 필요하면 다음 값을 변경합니다.

```env
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
```

기본 주소는 Next.js `http://localhost:3000`, FastAPI `http://127.0.0.1:8000`, Swagger `http://127.0.0.1:8000/docs`입니다.

## 주요 기능과 API

- CSV 검증·전처리: `POST /api/validate`, `POST /api/preprocess`
- 비동기 자동 학습: `POST /api/train/jobs`, `GET /api/train/jobs/{job_id}`
- 모델 목록·상세·삭제: `GET /api/models`, `GET /api/models/{model_id}`, `DELETE /api/models/{model_id}`
- 수율 예측·CSV 다운로드: `POST /api/predict`, `POST /api/predict/download`
- 원인 분석: `POST /api/explain`, `POST /api/explain/relationships`
- 예측/원인 분석 이력: `/api/predictions/history`, `/api/analyses/history`
- 전체 이력 초기화 요약·실행: `GET /api/admin/history/reset/summary`, `POST /api/admin/history/reset`
- 상태 확인: `GET /health`

전체 이력 초기화는 JSON 본문에 정확한 확인 문구를 요구하며 Secret은 사용하지 않습니다. 동일 IP에서 10분당 3회로 제한되고 실행 중인 작업이 있으면 409를 반환합니다. 원본 CSV, 환경변수, 소스 코드, 알람 및 자동화 이력은 초기화 대상이 아닙니다.

원인 분석 화면은 Target 분석, Lot별 원인, Wafer 상세, 공정 관계의 네 탭으로 구성됩니다. Lot 순위는 실제 Y가 있으면 실제 평균 Y, 없으면 예측 평균 Y를 사용합니다. Config는 원본 문자열 Category 그대로 표시하며 표본 5개 미만도 `표본 부족`으로 노출하되 공식 순위, Pareto, 유의성 판단에서는 제외합니다.

분석 보고서 생성·다운로드 기능과 Model Status UI는 제공하지 않습니다.

## 저장소와 Migration

Railway Volume을 `/data`에 연결하면 `RAILWAY_VOLUME_MOUNT_PATH`를 기준으로 모델, SQLite 이력, 분석 Artifact, 학습 Job 임시 파일 경로를 구성합니다. 신규 Pipeline 첫 운영 시작 시 Migration registry를 사용해 Legacy 모델과 이전 Report Artifact/DB marker를 한 번만 정리하며, 완료 ID를 저장해 재시작 시 반복하지 않습니다. 신규 모델과 예측·원인 분석 이력은 보존합니다.

## Railway

`railway.json`의 단일 worker 시작 명령은 다음과 같습니다.

```text
uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1
```

Healthcheck는 `/health`를 사용합니다. 배포 환경에는 최소 `FRONTEND_ORIGINS`와 영구 Volume 관련 경로를 설정해야 합니다. 로컬 검증은 Railway의 실제 RAM, OOM, 499/502 발생 여부를 대신하지 않으므로 운영 배포 후 별도 확인해야 합니다.

## 검증

```powershell
python -m pytest -q
Set-Location frontend
npm.cmd run lint
npm.cmd run build
```

원본 학습 CSV와 실제 운영 환경이 없는 경우 테스트 fixture 결과를 실제 모델 성능이나 Railway 안정성으로 표현하지 않습니다.
