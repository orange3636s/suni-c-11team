# Vercel 프런트엔드 배포 가이드

이 문서는 `frontend/`의 Next.js 애플리케이션을 Vercel에 배포하고
Render의 FastAPI 및 n8n Webhook과 연결하는 절차다. 실제 URL이나
credential은 저장소에 기록하지 않고 Vercel 프로젝트 환경변수로 관리한다.

## 최초 배포

1. Vercel에 로그인하고 **Add New → Project**를 선택한다.
2. 이 프로젝트가 있는 GitHub 저장소를 Import한다.
3. **Root Directory**를 `frontend`로 지정한다.
4. **Framework Preset**이 `Next.js`인지 확인한다.
5. **Build Command**는 `npm run build`를 사용한다.
6. **Output Directory**는 비워 두어 Next.js 기본값을 사용한다.
7. 프로젝트의 **Settings → Environment Variables**에 다음 값을 추가한다.

   | 이름 | 값 예시 | 적용 환경 |
   | --- | --- | --- |
   | `NEXT_PUBLIC_API_BASE_URL` | `https://your-api.onrender.com` | Production, Preview |
   | `NEXT_PUBLIC_N8N_WEBHOOK_URL` | `https://your-n8n.example/webhook/manufacturing-ai-analysis` | 필요한 환경 |

8. Deploy를 실행하고 빌드 로그가 성공인지 확인한다.
9. 발급된 HTTPS URL에서 `/`, `/training`, `/prediction`,
   `/root-cause`, `/alerts`, `/automation`을 확인한다.
10. Render 환경변수 `FRONTEND_ORIGINS`에 Vercel의 정확한 origin을
    추가한다. 경로와 끝 슬래시는 넣지 않는다.

`NEXT_PUBLIC_*` 값은 브라우저 번들에 포함되는 공개 설정이다. 토큰, API
키, Slack credential을 넣으면 안 된다. API URL이 운영 빌드에서 빠지면
화면 요청은 설정 누락을 명시하는 오류를 반환한다.

## 자동 배포와 재배포

연결된 Git 저장소의 배포 대상 브랜치에 변경사항이 push되면 Vercel이
새 빌드를 자동 실행할 수 있다. 이 문서는 그 흐름만 설명하며 저장소
명령이나 실제 배포를 실행하지 않는다.

환경변수를 바꾸면 기존 정적 빌드에는 반영되지 않는다. **Deployments**에서
최신 배포의 **Redeploy**를 실행하거나 새 저장소 변경으로 다시 빌드한다.

## 오류 확인

- 빌드 실패: Deployments의 Build Logs에서 TypeScript, ESLint, 패키지
  설치 오류를 확인한다.
- API 연결 실패: 브라우저 개발자 도구의 Network/Console과
  `NEXT_PUBLIC_API_BASE_URL`을 확인한다.
- CORS 오류: Render의 `FRONTEND_ORIGINS`에 현재 Vercel origin이 정확히
  들어 있는지 확인한다. 여러 주소는 쉼표로 구분한다.
- 404/잘못된 프로젝트: Root Directory가 반드시 `frontend`인지 확인한다.
- 로고 오류: `frontend/public/sk-hynix-logo.png`가 배포 결과에 포함됐는지
  확인한다.
