# n8n 제조 공정 AI 자동화 가이드

## 전체 아키텍처

```text
CSV + model_id
  → n8n Webhook
  → 입력 검증
  → FastAPI POST /api/analyze
  → 예측·위험 분류·SHAP·보고서 요약
  → 위험 여부 분기
  → Slack 알림(선택)
  → Webhook JSON 응답
```

역할은 다음과 같이 구분한다.

- n8n: Webhook 수신, 입력 검증, FastAPI 호출, 위험 분기, Slack 전달
- FastAPI: CSV 검사·전처리, 모델 로드, 예측, SHAP 분석, 보고서 생성
- Next.js: 사용자가 직접 사용하는 분석 화면과 자동화 설정 상태 표시

CSV 원본은 FastAPI나 n8n 워크플로우가 프로젝트 폴더에 영구 저장하지
않는다.

## 워크플로우 가져오기

1. n8n에서 **Workflows → Import from File**을 선택한다.
2. `workflows/n8n_manufacturing_ai_workflow.json`을 선택한다.
3. `FASTAPI_BASE_URL` 환경변수를 설정한다.
4. Slack을 사용할 경우 두 Slack 노드에 credential과 채널을 연결한다.
5. 테스트 Webhook으로 확인한 후 워크플로우를 활성화한다.

워크플로우 이름은 `제조 공정 예측 및 불량분석 AI 자동화`이다.

## Webhook URL

- Path: `manufacturing-ai-analysis`
- Method: `POST`
- 파일 필드: `data`
- 테스트 URL: `http://localhost:5678/webhook-test/manufacturing-ai-analysis`
- 활성화 후 URL: `http://localhost:5678/webhook/manufacturing-ai-analysis`

n8n 주소와 포트가 다르면 해당 환경의 Webhook URL을 사용한다.

## FASTAPI_BASE_URL

n8n 실행 환경에 기본 서버 주소만 설정한다.

```env
FASTAPI_BASE_URL=http://127.0.0.1:8000
```

끝에 `/api/analyze`를 붙이지 않는다. HTTP Request 노드가 자동으로
`/api/analyze`를 추가한다.

n8n을 Docker에서 실행하고 FastAPI를 Windows 호스트에서 실행한다면
컨테이너의 `127.0.0.1`은 호스트를 가리키지 않는다. 일반적으로 다음 값을
사용한다.

```env
FASTAPI_BASE_URL=http://host.docker.internal:8000
```

운영 환경에서는 실제 HTTPS FastAPI 기본 주소로 교체하되 워크플로우 JSON에
운영 URL을 직접 하드코딩하지 않는다.

## 요청 필드

n8n Webhook은 다음 multipart 필드를 받는다.

- `data`: CSV binary 파일
- `model_id`: `GET /api/models`에서 확인한 모델 ID
- `warning_threshold`: 선택, 기본 95
- `danger_threshold`: 선택, 기본 90
- `max_rows`: 선택, 기본 500, 최대 1,000
- `top_n`: 선택, 기본 20, 최대 100
- `per_wafer_top_n`: 선택, 기본 5
- `include_report`: 선택, 기본 true

입력 검증 노드는 CSV, model_id, 숫자 형식 및
`warning_threshold > danger_threshold` 조건을 검사한다. 실패하면 FastAPI를
호출하지 않고 오류 응답을 반환한다.

## Slack 연결

워크플로우 파일에는 Slack token이나 credential ID가 포함되어 있지 않다.

1. n8n의 **Credentials**에서 Slack OAuth2 또는 Slack API credential을 만든다.
2. `Slack Alert - Danger`와 `Slack Alert - Warning` 노드에 같은 credential을
   선택한다.
3. 환경변수 `SLACK_CHANNEL_ID`를 설정하거나 각 노드에서 채널을 선택한다.
4. Slack 앱에 메시지 작성 권한과 대상 채널 접근 권한이 있는지 확인한다.

두 Slack 노드는 `continueOnFail`이 활성화되어 있다. Slack 전송 실패가
FastAPI 분석 결과 자체를 실패로 바꾸지는 않는다.

## FastAPI 직접 테스트

저장소 루트에서 FastAPI를 실행한다.

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

PowerShell에서 직접 통합 API를 호출하는 예시:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/analyze" `
  -F "file=@tests/fixtures/training_sample.csv;type=text/csv" `
  -F "model_id=YOUR_MODEL_ID" `
  -F "warning_threshold=95" `
  -F "danger_threshold=90" `
  -F "max_rows=500" `
  -F "top_n=20" `
  -F "per_wafer_top_n=5" `
  -F "include_report=true"
```

`YOUR_MODEL_ID`는 `GET http://127.0.0.1:8000/api/models` 결과의
`model_id`로 교체한다.

## n8n Webhook 테스트

n8n 편집 화면에서 Webhook 노드의 **Listen for test event**를 누른 다음:

```powershell
curl.exe -X POST "http://localhost:5678/webhook-test/manufacturing-ai-analysis" `
  -F "data=@tests/fixtures/training_sample.csv;type=text/csv" `
  -F "model_id=YOUR_MODEL_ID" `
  -F "warning_threshold=95" `
  -F "danger_threshold=90" `
  -F "max_rows=500" `
  -F "top_n=20"
```

워크플로우를 활성화한 후에는 `/webhook-test/` 대신 `/webhook/` URL을
사용한다.

## HTML 보고서

`POST /api/analyze`는 분석 요약과 다음 안내 경로를 반환한다.

```json
{
  "report": {
    "included": true,
    "report_id": "...",
    "download_endpoint": "/api/report/download"
  }
}
```

서버가 업로드 CSV나 생성 HTML을 영구 저장하지 않으므로
`download_endpoint`는 가짜 파일 URL이 아니라 호출 안내다. n8n에서 HTML이
필요하면 동일 CSV와 옵션으로 `POST /api/report/download`를 다시 호출한다.

## 오류 대응

- `binary.data CSV 파일이 필요합니다`: Webhook multipart 파일 필드가
  `data`인지 확인한다.
- `model_id가 필요합니다`: body의 model_id와 저장 모델 목록을 확인한다.
- threshold 오류: 숫자 형식과 주의 기준이 위험 기준보다 큰지 확인한다.
- FastAPI 연결 실패: `FASTAPI_BASE_URL`, 방화벽, Docker host 주소를 확인한다.
- feature 누락: 학습 데이터와 신규 CSV의 R/D/EQ feature 구성을 비교한다.
- SHAP 지연: 먼저 `max_rows`를 낮춰 확인하고 FastAPI 로그를 점검한다.
- Slack 실패: credential 권한, 채널 ID, Slack 앱의 채널 참여 여부를 확인한다.

외부 응답에는 FastAPI의 `detail` 또는 정리된 오류만 전달하며 Python
traceback을 포함하지 않는다.

## 성능 주의사항

SHAP 분석은 모델과 feature 수에 따라 수초 이상 걸릴 수 있다. 워크플로우의
FastAPI HTTP timeout은 10분으로 설정되어 있지만, 기본 `max_rows=500`을
유지하고 필요할 때만 최대 1,000까지 늘린다. 가짜 진행률은 표시하지 않는다.

## 보안 주의사항

- Slack token, credential ID, 운영 URL을 저장소에 기록하지 않는다.
- 운영 Webhook에는 HTTPS, 접근 제어, 요청 크기 제한을 적용한다.
- Webhook URL은 비밀값처럼 관리하고 불필요하게 공개하지 않는다.
- 업로드 CSV에 민감 정보가 포함될 수 있으므로 n8n 실행 데이터 보존 정책을
  별도로 설정한다.
- FastAPI의 CORS는 브라우저 정책이며 n8n 서버 간 호출의 인증을 대신하지
  않는다. 운영 환경에서는 API 인증이나 사설 네트워크 구성을 추가한다.
