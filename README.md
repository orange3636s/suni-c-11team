# SUNI — 제조 공정 불량 예측 & 원인 분석 AI

## 1. 프로젝트 소개

반도체 공정 데이터에서 불량 원인 인자를 찾고, wafer별 검토 우선순위를 제시합니다. train 10,000 wafer로 학습하고 Y가 없는 배치에 대해 예측합니다.

**이 도구가 하는 것**

- 어느 공정 인자를 어느 범위로 관리해야 하는지
- 어느 wafer를 먼저 검토해야 하는지
- 그 판단을 얼마나 믿을 수 있는지

**이 도구가 하지 않는 것**

- 수율 절대값 예측 (R² 0.18 — 순위 판별용입니다)
- 장비·챔버별 순위 (Config 인자 150건 검정 중 BH-FDR 통과 0건)
- 랏 단위 관리 (ICC(1,1) 0.005)
- 폐기 자동 판정

## 2. 빠른 시작 (로컬)

요구: Python 3.12, Node 20 이상

```powershell
# 백엔드
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m uvicorn api.main:app --reload

# 프런트엔드 (별도 터미널)
cd frontend
npm install
npm run dev
```

http://localhost:3000 접속.

첫 실행 시 저장된 스냅샷이 없으면 서버가 내장 `train.CSV`로 자동 학습하고 `test_remove_y.CSV`를 분석해 첫 화면을 채웁니다. 수십 초 걸리며 화면에 진행 표시가 나옵니다.

## 3. 화면 구성

- **모니터링 홈** — 조치 우선순위, 조치 가능 범위, 데이터 한계 세 블록
- **Config별 트리맵** — 장비 구성별 불량률, Y1~Y5 다섯 트리맵을 세로로 배치
- **원인 분석** — 파레토, 산점도, 박스플롯, 상관관계 히트맵
- **수율 예측** — wafer별 예측 수율과 검토 순위
- **알림 기록** — 발송된 알림 원문
- **즐겨찾기** — 저장한 차트

모든 화면에 SUNI 챗봇 패널이 떠 있습니다. 현재 분석 결과를 근거로 답하거나(`chat` 모드) 6개 섹션 구조의 보고서를 작성합니다(`report` 모드).

사이드바 하단 네 버튼:

- **모델 학습** — 학습 데이터를 정합니다(수동 업로드 전용)
- **모델 분석** — 등록된 데이터로 네 화면(모니터링 홈·Config별 트리맵·원인 분석·수율 예측)을 한 번에 갱신합니다 — 새로고침 역할
- **알림·자동화 설정** — refresh time마다 수율 예측만 계산해 알림을 발송하도록 설정합니다
- **화면 모드** — 라이트/다크/시스템

## 4. 핵심 개념

- **Y = 100 − (Y1+Y2+Y3+Y4+Y5)** — 최종 수율과 다섯 불량 모드
- **핵심 인자** — 각 모드의 기여율 10% 이상인 인자
- **권장 구간** — 손실이 가장 낮은 인자 범위
- **회수 폭 / 기대 회수** — 조치 시 되찾을 수 있는 양
- **신뢰도 n/5** — 예측 근거가 있는 모드 수

상세 산식은 5장, 알고리즘은 6장 참고.

## 5. 지표와 공식

```
Adjusted R²  = 1 - (1 - R²) x (n - 1) / (n - p - 1)
기여율        = AdjR2_k / sum(AdjR2)
회수 폭       = 구간 밖 평균 손실 - 구간 안 평균 손실  (train.CSV 기준)
비중          = 이 타깃의 평균 손실 / 5개 타깃 평균 손실 합계
기대 회수     = 회수 폭 x 비중 / 100
신뢰도        = 기여율 10% 이상 인자가 사용된 타깃 수 / 5
MNAR 배수     = P(계측 | 손실 상위 10%) / P(계측)
```

모델 성능 지표: Adjusted R² · RMSE · MAE · MSE · AUC. AUC는 하위 10% 식별 기준(이진 라벨 = 손실 상위 10% 여부, 예측 점수 = 낮은 예측 Y일수록 양성).

단위 규칙:

| 단위 | 대상 |
|---|---|
| % | Y · Y1~Y5 · 계측률 · 기여율 · 이탈률 |
| %p | 회수 폭 · 기대 회수 · RMSE · MAE |
| 없음 | Adjusted R² · MSE · AUC |

## 6. 알고리즘

```
인자 선정      Adjusted R² + BH-FDR (alpha 0.05)
               구간 수는 Sturges 자동, ceil(1 + log2(n)), [5, 15] 클램프
곡선 적합      1차/2차 자동 판정 (F 검정, p < 0.01 AND 2차 계수 > 0 이면 2차)
권장 구간      SPC(관측 분위) vs ML(결정트리) 비교 후 채택
예측           LightGBM · 모드별(Y1~Y5) 예측 후 합산해 Y 산출
```

Config(장비 구성) 효과와 랏(Lot) 효과는 사용자가 자주 묻는 질문이라 근거를 남겨둡니다.

- **Config 효과**: 30개 Config 컬럼 × 5개 타깃 = 150건 검정에서 BH-FDR(α=0.05) 통과 0건, 관측된 최대 Adjusted R² 0.0027 (train.CSV 기준)
- **랏 효과**: ICC(1,1) = 0.005. 랏 간 분산 비중 4.49%는 랏당 25장일 때 무효과 기대값(1/25=4.0%)과 거의 같은 수준

## 7. 자동화와 알림

"자동화 사용"이 켜져 있는 동안 refresh time마다:

1. SQL에서 최신 배치를 조회 (SQL 미설정이면 그 주기는 건너뜁니다 — 발송 없음)
2. 활성 모델로 수율 예측만 계산 (모니터링 홈·Config별 트리맵·원인 분석은 갱신하지 않음)
3. 연결된 채널로 발송

**알림 양식 예시** (Telegram)

```
[SUNI] 수율 예측 갱신 (2026-08-12 09:00)
소스: train.CSV

예측 수율이 낮은 WF TOP 10
LOT_WF_ID  Y       신뢰도
LOT23-W07  61.40%  4/5
LOT19-W22  64.80%  5/5
LOT23-W03  68.10%  3/5

Y1 불량률 높은 순
LOT_WF_ID  Y1      핵심인자
LOT23-W07  12.30%  Step28_R1 (86.2%)
LOT19-W22  9.80%   Step28_R1 (71.5%)
LOT23-W03  7.10%   Step16_R1 (62.0%)

(Y2~Y5도 같은 형식으로 이어집니다)

Y1 불량이 전체 손실의 71%를 차지합니다.

예측 수율의 절대값은 정확도가 낮습니다. 검토 우선순위로 활용하세요.
```

Gmail은 같은 내용을 HTML 표로, Slack은 텍스트로 보냅니다.

**채널별 연결 절차**

| 채널 | 절차 | 소요 |
|---|---|---|
| Telegram | BotFather에서 봇 토큰 발급 → 환경변수 설정 → 서버 재시작 → 대시보드에서 `/start` → 인증 코드 입력 | 10분 (코드 유효시간) |
| Gmail | 구글 계정 2단계 인증 → 앱 비밀번호 16자 발급 → SMTP 5개 환경변수 설정 → 인증 메일의 링크 클릭 | 5분 (인증 대기시간) |
| Slack | 설정 패널에 Incoming Webhook URL을 직접 입력 | 즉시 |

## 8. 데이터 요구사항

**학습 데이터**

- `Step{n}_R{m}` · `Step{n}_D{m}` · `Step{n}_Config` — 결측 허용
- `Y` · `Y1~Y5` — 결측 0건 필수

**분석 데이터**

- Step 컬럼은 학습 데이터와 동일한 이름 규칙
- `Y` · `Y1~Y5` 일부 또는 전부 결측 허용

**업로드 한도**: 150MB, 200,000행. 20,000행을 넘으면 스크리닝·히트맵·트리맵·권장구간 계산은 로트 단위 표본을 쓰고(수율 예측·모델 추론은 항상 전량), 화면에 표본 사용 고지가 표시됩니다.

**내장 데이터**

- `data/bundled/train.CSV` — 10,000 wafer, 학습용
- `data/bundled/test_remove_y.CSV` — 1,000 wafer, Y 전량 결측

## 9. 배포

- **백엔드**: Railway (RAILPACK 빌더, `railway.json`의 시작 명령 `uvicorn api.main:app --host 0.0.0.0 --port $PORT --workers 1`)
- **프런트엔드**: Vercel (Root Directory `frontend`)

**Railway 볼륨 설정 (필수)**

Railway는 재시작·재배포마다 컨테이너 파일시스템을 초기화합니다. 볼륨을 붙이지 않으면 텔레그램·Slack·Gmail 연동, 챔피언 모델 지정, 분석 스냅샷, 즐겨찾기, 알림 발송 이력, 업로드한 데이터셋, 모델 아티팩트가 재시작마다 전부 사라집니다.

```
Settings -> Volumes -> New Volume
Mount path: /app/var
```

`/app/data`를 마운트 경로로 쓰지 마세요 — 저장소의 내장 데이터(`data/bundled/train.CSV` 등)가 가려져 콜드 스타트가 실패합니다.

**LightGBM 의존성**

```
RAILPACK_BUILD_APT_PACKAGES  = libgomp1
RAILPACK_DEPLOY_APT_PACKAGES = libgomp1
```

## 10. 환경변수

| 변수 | 용도 | 필수 | 재시작 |
|---|---|---|---|
| `FRONTEND_ORIGINS` | CORS 허용 origin (기본값 localhost:3000) | 배포 시 | 필요 |
| `MODEL_DIR` | 모델 저장 경로 | 아니오 (기본값 있음) | 필요 |
| `MAX_UPLOAD_SIZE_MB` | 업로드 크기 제한 | 아니오 | 필요 |
| `UPSTAGE_API_KEY` / `UPSTAGE_BASE_URL` / `UPSTAGE_MODEL` | SUNI 챗봇 LLM 연결 | 챗봇 사용 시 | 필요 |
| `TELEGRAM_BOT_TOKEN` | 봇 토큰 | 알림 시 | 필요 |
| `TELEGRAM_BOT_USERNAME` | 봇 사용자명 (@ 제외) | 알림 시 | 필요 |
| `SMTP_HOST` / `SMTP_PORT` / `SMTP_USER` / `SMTP_PASSWORD` | Gmail 발송 | 알림 시 | 필요 |
| `SMTP_FROM_EMAIL` | 발신자 주소 (SMTP_USER와 동일) | 알림 시 | 필요 |
| `NOTIFY_VERIFY_BASE_URL` | 인증 링크 기준 URL | 알림 시 | 필요 |
| `AUTO_INGEST_DB_DRIVER` / `DB_PASSWORD` / `AUTO_INGEST_QUERY` / `AUTO_INGEST_CURSOR_COLUMN` | SQL 자동 갱신 (host/port/db/user는 설정 패널에 저장) | 자동화 시 | 필요 |

Slack은 환경변수가 아니라 설정 패널에서 Webhook URL을 직접 입력해 연결합니다. 전체 목록과 기본값은 `.env.example` 참고.

## 11. 데이터의 한계

- **결측** — R 인자 평균 계측률 15.0%(결측 85.0%), D 인자 평균 계측률 5.0%(결측 95.0%) (train.CSV 기준, 48개 R컬럼/10개 D컬럼)
- **Config 효과** — 150건 검정(30 Config × 5 타깃) 중 BH-FDR 통과 0건, 관측 최대 Adjusted R² 0.0027
- **랏 효과** — ICC(1,1) 0.005. 랏 간 분산 비중 4.49%는 랏당 25장 기준 무효과 기대값(1/25=4.0%) 수준
- **계측 편향** — MNAR. 예: Y3의 핵심 인자 Step1_D1은 손실 상위 10% wafer에서 계측률이 전체 대비 9.58배 높음
- **예측 정확도** — 전체 Y Adjusted R² 0.176 (test n=1,500, RMSE 3.37%p, MAE 2.72%p) — 절대값이 아니라 순위 판별용
- **순위 정확도** — 하위 10% 식별 AUC 0.735
- **wafer 좌표** — 없음. WMAP·Contour·Zonal 분석 불가
- **시간축** — 없음. 추세·드리프트 분석 불가

## 12. 개발

```powershell
python -m pytest -q
Set-Location frontend
npm.cmd run build
```

```
프로젝트 구조
├── api/            FastAPI 라우트·스키마
├── src/
│   ├── analysis/   스크리닝·권장구간·리포트·챗봇용 통계
│   ├── ml/         전처리·학습·추론 파이프라인
│   ├── notifications/  Slack·Telegram·Gmail 발송
│   ├── automation/ SQL 자동 갱신·수율 예측 발송 파이프라인
│   └── runtime/    런타임 스토어(SQLite)·데이터셋 레지스트리
├── frontend/       Next.js 대시보드
├── config/         스키마·임계값·업로드 한도 YAML
├── prompts/        SUNI 챗봇 시스템 프롬프트
├── docs/
│   ├── decisions.md      폐기된 설계와 근거
│   └── validation/       데이터·챗봇 검증 기록
└── tests/
```
