# 배포 체크리스트

## FastAPI / Render

- [ ] 저장소 루트에서 `requirements.txt` 설치 성공
- [ ] Python 3.12.13 사용 확인
- [ ] `GET /health` HTTP 200 및 환경값 확인
- [ ] `GET /ready` HTTP 200 및 모델 디렉터리 준비 확인
- [ ] Swagger `/docs` 접근 확인
- [ ] Vercel origin만 CORS 허용
- [ ] `MODEL_DIR=models` 설정
- [ ] `MAX_UPLOAD_SIZE_MB` 제한 확인
- [ ] `GET /api/models` 초기 모델 표시
- [ ] 샘플 CSV로 `POST /api/analyze` 성공
- [ ] 외부 응답에 traceback과 내부 절대 경로 없음

## Next.js / Vercel

- [ ] Root Directory가 `frontend`
- [ ] `npm run lint` 성공
- [ ] `npm run build` 성공
- [ ] `NEXT_PUBLIC_API_BASE_URL` 설정
- [ ] 필요한 경우 `NEXT_PUBLIC_N8N_WEBHOOK_URL` 설정
- [ ] 로고 이미지 표시
- [ ] 모든 기존 페이지 접근
- [ ] 브라우저 Console/Network 오류 없음

## n8n

- [ ] workflow JSON import 성공
- [ ] `FASTAPI_BASE_URL`이 운영 Render HTTPS origin
- [ ] URL 끝에 `/api/analyze`를 중복 입력하지 않음
- [ ] Test Webhook으로 CSV 전송 성공
- [ ] workflow 활성화 후 Production Webhook 확인
- [ ] danger 분기 확인
- [ ] warning 분기 확인
- [ ] normal 분기 확인

## Slack

- [ ] n8n credential 연결
- [ ] 앱의 메시지 작성 및 채널 접근 권한 확인
- [ ] Danger 테스트 알림 수신
- [ ] Warning 테스트 알림 수신
- [ ] Slack 실패 시 분석 응답 유지 확인

## 보안과 운영

- [ ] 저장소와 workflow JSON에 토큰·credential ID 없음
- [ ] 실제 `.env`, `.env.local`이 제외됨
- [ ] 예제 환경변수 파일은 포함됨
- [ ] 업로드 크기 제한 확인
- [ ] CORS wildcard 미사용
- [ ] 내부 traceback 미노출
- [ ] 운영 중 생성 모델의 재배포 유실 가능성 공유
- [ ] n8n 실행 데이터 보존 정책 검토
