# Render FastAPI 배포 가이드

FastAPI는 저장소 루트에서 실행하며 `render.yaml`을 사용할 수 있다.
실제 계정 접속과 배포는 운영자가 수행한다.

## Web Service 생성

1. Render에 로그인하고 **New → Web Service**를 선택한다.
2. 이 프로젝트의 GitHub 저장소를 연결한다.
3. **Root Directory**는 비워 두어 저장소 루트를 사용한다.
4. Runtime은 Python을 선택한다.
5. Build Command는 `pip install -r requirements.txt`로 설정한다.
6. Start Command는 다음과 같이 설정한다.

   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port $PORT
   ```

7. 저장소 루트의 `.python-version`에 따라 Python 3.12.13이
   선택되는지 빌드 로그에서 확인한다.
8. 다음 환경변수를 설정한다.

   | 이름 | 운영값 |
   | --- | --- |
   | `APP_ENV` | `production` |
   | `FRONTEND_ORIGINS` | `https://your-project.vercel.app` |
   | `MODEL_DIR` | `models` |
   | `MAX_UPLOAD_SIZE_MB` | `20` |
   | `LOG_LEVEL` | `INFO` |

   `FRONTEND_ORIGINS`에는 Preview URL 등 필요한 origin을 쉼표로 구분해
   추가한다. `*`는 사용하지 않는다.

9. Health Check Path를 `/health`로 설정하고 배포한다.

`render.yaml`에는 요금제 필드를 넣지 않았으므로 서비스 생성 시 필요한
인스턴스 유형을 운영자가 선택한다.

## 배포 확인

- `GET https://your-api.onrender.com/health`: 상태, 서비스명, 환경, 버전,
  모델 디렉터리 준비 여부 확인
- `GET https://your-api.onrender.com/ready`: 모델 디렉터리가 쓰기
  가능한지 확인. 실패 시 HTTP 503
- `GET https://your-api.onrender.com/docs`: Swagger UI 확인
- `GET https://your-api.onrender.com/api/models`: 초기 모델 목록 확인
- `POST /api/analyze`: 운영 전 샘플 CSV로 통합 분석 확인

응답에는 전체 내부 경로나 환경변수 값이 노출되지 않는다. 오류 원인은
Render의 **Logs**에서 확인한다.

## 모델 파일과 저장소 수명

저장소에 포함된 약 540KB의 초기 `.joblib`과 JSON 메타데이터는 배포
이미지에서 바로 사용할 수 있다. 운영 중 `/api/train`으로 만든 모델은
Render 인스턴스의 로컬 파일 시스템에 저장되므로 재배포, 재생성, 장애
복구 시 사라질 수 있다. 이번 구성에는 DB, Object Storage, Persistent
Disk를 추가하지 않는다.

장기 운영에서는 Render Persistent Disk 또는 외부 Object Storage를
검토하고 `MODEL_DIR`을 해당 저장 경로에 맞춘다. 모델에는 학습 데이터의
민감 정보가 포함될 가능성이 있으므로 외부 반출과 저장소 포함 전에
보안 검토가 필요하다.

## 운영 주의사항

- SHAP와 모델 학습은 일반 API보다 오래 걸리고 메모리를 많이 사용할 수
  있다. n8n timeout과 인스턴스 자원을 함께 확인한다.
- 서비스가 유휴 상태에서 정지되는 인스턴스 유형은 첫 요청이 느릴 수
  있다.
- 재배포는 저장소 자동 배포 또는 Render Dashboard의 Manual Deploy로
  수행한다.
- Python 기본 버전에 의존하지 말고 빌드 로그에서 3.12.13을 확인한다.
- 파일 저장 실패가 발생하면 `/ready`, `MODEL_DIR`, 파일 시스템 권한을
  확인한다.
