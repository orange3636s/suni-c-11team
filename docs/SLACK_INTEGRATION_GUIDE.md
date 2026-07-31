# n8n Slack 알림 연결 가이드

Slack 알림은 FastAPI가 직접 보내지 않고 n8n의 Danger/Warning 분기
뒤에서 전송한다. 토큰과 credential ID는 코드나 workflow JSON에 넣지
않고 n8n Credentials에만 저장한다.

## 연결 절차

1. 알림을 받을 Slack workspace와 채널을 준비한다.
2. n8n의 **Credentials → New Credential**에서 Slack을 선택한다.
3. 관리 정책에 따라 Slack OAuth2 방식 또는 Bot Token 방식을 선택한다.
4. 앱에 메시지 작성 권한을 부여하고 대상 채널에 앱을 초대한다.
5. `workflows/n8n_manufacturing_ai_workflow.json`을 가져온다.
6. `Slack Alert - Danger` 노드에 credential과 알림 채널을 연결한다.
7. `Slack Alert - Warning` 노드에도 credential과 알림 채널을 연결한다.
8. 각 노드를 Execute Step으로 실행해 테스트 메시지가 도착하는지
   확인한다.
9. 위험·주의 샘플을 Webhook으로 전송해 두 분기가 의도대로 동작하는지
   확인한다.

Bot Token 방식이면 토큰을 n8n credential 입력 화면에만 넣는다. OAuth
방식이면 n8n의 Redirect URL을 Slack 앱 설정에 정확히 등록한다. 채널
선택 목록이 비어 있으면 앱 권한, workspace, 채널 참여 상태를 확인한다.

## 실패 격리

두 Slack 노드는 `continueOnFail`이 활성화되어 있다. Slack API 오류,
권한 부족, 잘못된 채널 때문에 알림이 실패해도 앞에서 완료된 FastAPI
분석 결과와 Webhook 응답 경로는 유지된다. n8n 실행 기록에서 Slack 노드
오류를 별도로 확인하고 재전송 여부를 판단한다.

credential을 JSON에 넣지 않는 이유는 저장소 유출, 로그 노출, 환경 간
오사용과 토큰 회전의 어려움을 막기 위해서다. 토큰이 노출됐다면 즉시
Slack에서 폐기·재발급하고 n8n credential을 갱신한다.

