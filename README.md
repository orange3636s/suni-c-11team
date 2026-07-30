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
- FastAPI: http://127.0.0.1:8000
- Swagger: http://127.0.0.1:8000/docs
