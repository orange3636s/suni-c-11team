"use client";

import { X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import { formatLastRun } from "@/lib/timeFormat";
import { useFocusTrap } from "@/lib/useFocusTrap";
import {
  connectGmail,
  connectSlack,
  disconnectNotificationChannel,
  saveAutomationSettings,
  testAutomationConnection,
  testGmail,
  testSlack,
  testTelegram,
  verifyTelegramCode,
} from "@/lib/api";
import type { NotificationSettingsSummary } from "@/types/data";

export default function SettingsPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { notifications, setNotifications } = useAnalysisState();
  const panelRef = useRef<HTMLDivElement>(null);
  useFocusTrap(panelRef, open);

  useEffect(() => {
    if (!open) return;
    function handleKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="settingsPanelBackdrop" onClick={onClose} role="presentation">
      <div
        ref={panelRef}
        className="settingsPanel"
        onClick={(event) => event.stopPropagation()}
        role="dialog"
        aria-modal="true"
        aria-label="알림·자동화 설정"
        tabIndex={-1}
      >
        <div className="settingsPanelHeader">
          <h2>알림·자동화 설정</h2>
          <button type="button" className="settingsPanelClose" onClick={onClose} aria-label="닫기">
            <X size={16} strokeWidth={1.5} />
          </button>
        </div>
        <div className="settingsPanelBody">
          <section className="settingsSection">
            <h3>채널 연결</h3>
            <p className="settingsSectionDesc">수율 예측 알림을 받을 채널을 연결합니다.</p>
            <div className="notifyChannelList">
              <SlackCard summary={notifications} onUpdate={setNotifications} />
              <TelegramCard summary={notifications} onUpdate={setNotifications} />
              <GmailCard summary={notifications} onUpdate={setNotifications} />
            </div>
          </section>

          <AutomationSection summary={notifications} onUpdate={setNotifications} />
        </div>
      </div>
    </div>
  );
}

type ChannelProps = {
  summary: NotificationSettingsSummary;
  onUpdate: (next: NotificationSettingsSummary) => void;
};

function TestResultNote({ result }: { result: { ok: boolean; error: string | null } | null }) {
  if (!result) return null;
  return (
    <p className={`notifyTestResult ${result.ok ? "ok" : "error"}`}>
      {result.ok ? "테스트 메시지를 보냈습니다. 채널을 확인하세요." : result.error || "발송에 실패했습니다."}
    </p>
  );
}

function SlackCard({ summary, onUpdate }: ChannelProps) {
  const { slack } = summary;
  const [expanded, setExpanded] = useState(false);
  const [webhookUrl, setWebhookUrl] = useState("");
  const [channel, setChannel] = useState("");
  const [busy, setBusy] = useState<"test" | "connect" | null>(null);
  const [error, setError] = useState("");
  const [testResult, setTestResult] = useState<{ ok: boolean; error: string | null } | null>(null);

  async function handleTest() {
    // 이미 연결된 채널이면 폼의 webhookUrl(빈 값)이 아니라 서버에
    // 저장된 값으로 테스트한다 -- 연결 요약에는 마스킹된 값만 있어
    // 원본을 다시 보낼 수 없다.
    if (!slack.connected && !webhookUrl.trim()) {
      setError("Webhook URL을 먼저 입력하세요.");
      return;
    }
    setError("");
    setBusy("test");
    setTestResult(null);
    try {
      setTestResult(await testSlack(slack.connected ? undefined : webhookUrl.trim()));
    } catch {
      setTestResult({ ok: false, error: "테스트 발송 요청에 실패했습니다." });
    } finally {
      setBusy(null);
    }
  }

  async function handleConnect() {
    if (!webhookUrl.trim()) {
      setError("Webhook URL을 입력하세요.");
      return;
    }
    setError("");
    setBusy("connect");
    try {
      onUpdate(await connectSlack(webhookUrl.trim(), channel.trim() || null));
      setExpanded(false);
      setWebhookUrl("");
      setChannel("");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "연결에 실패했습니다.");
    } finally {
      setBusy(null);
    }
  }

  async function handleDisconnect() {
    if (!window.confirm("Slack 연결을 해제할까요? 다시 연결하려면 Webhook URL을 처음부터 다시 입력해야 합니다.")) return;
    try {
      onUpdate(await disconnectNotificationChannel("slack"));
    } catch {
      // unhandled rejection이 아니라 눈에 보이는 오류로 -- 실패해도
      // 화면은 "연결됨" 그대로라 사용자가 재시도할 수 있어야 한다.
      setError("연결 해제에 실패했습니다. 다시 시도해 주세요.");
    }
  }

  return (
    <div className="notifyChannelCard">
      <div className="notifyChannelHeaderRow">
        <span className="notifyChannelName">Slack</span>
        {slack.connected ? (
          <span className="notifyChannelStatus connected">연결됨</span>
        ) : (
          <button type="button" className="notifyConnectButton" onClick={() => setExpanded((v) => !v)}>
            연결하기
          </button>
        )}
      </div>
      {slack.connected && (
        <>
          <div className="notifyChannelConnectedRow">
            <span className="notifyChannelTarget">{slack.target || slack.webhook_masked}</span>
            <button type="button" className="notifyInlineButton" onClick={handleTest} disabled={busy === "test"}>
              {busy === "test" ? "발송 중…" : "테스트 발송"}
            </button>
            <button type="button" className="notifyInlineButton danger" onClick={handleDisconnect}>
              해제
            </button>
          </div>
          {error && <p className="notifyFieldError">{error}</p>}
          <TestResultNote result={testResult} />
        </>
      )}
      {!slack.connected && expanded && (
        <div className="notifyConnectForm">
          <ol className="notifyConnectSteps">
            <li>
              <strong>Slack 워크스페이스에서</strong>
              <span>Incoming Webhook 앱 추가</span>
            </li>
            <li>
              <strong>알림 받을 채널 선택</strong>
              <span>#eng-yield 등</span>
            </li>
            <li>
              <strong>발급된 URL 복사</strong>
              <span>hooks.slack.com/services/…</span>
            </li>
          </ol>
          <label className="notifyFieldLabel">
            Webhook URL
            <input
              type="text"
              value={webhookUrl}
              onChange={(event) => setWebhookUrl(event.target.value)}
              placeholder="https://hooks.slack.com/services/…"
            />
          </label>
          <label className="notifyFieldLabel">
            채널 (선택)
            <input type="text" value={channel} onChange={(event) => setChannel(event.target.value)} placeholder="#eng-yield" />
          </label>
          {error && <p className="notifyFieldError">{error}</p>}
          <TestResultNote result={testResult} />
          <div className="notifyFormActions">
            <button type="button" className="notifyInlineButton" onClick={handleTest} disabled={busy !== null}>
              {busy === "test" ? "발송 중…" : "테스트 발송"}
            </button>
            <button type="button" className="notifyPrimaryButton" onClick={handleConnect} disabled={busy !== null}>
              {busy === "connect" ? "연결 중…" : "연결"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

function TelegramCard({ summary, onUpdate }: ChannelProps) {
  const { telegram } = summary;
  const [expanded, setExpanded] = useState(false);
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState<"test" | "verify" | null>(null);
  const [error, setError] = useState("");
  const [testResult, setTestResult] = useState<{ ok: boolean; error: string | null } | null>(null);
  // 봇 username은 백엔드가 단일 소스다 -- 프런트 환경변수나
  // 하드코딩 폴백을 두지 않는다. 서버가 TELEGRAM_BOT_USERNAME을 설정하지
  // 않았으면 그대로 null이고, 그 사실을 화면에 드러낸다(죽은 링크를
  // 보여주지 않는다).
  const botUsername = summary.telegram_bot_username;

  async function handleVerify() {
    if (!code.trim()) {
      setError("봇이 보낸 6자리 코드를 입력하세요.");
      return;
    }
    setError("");
    setBusy("verify");
    try {
      onUpdate(await verifyTelegramCode(code.trim()));
      setExpanded(false);
      setCode("");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "인증에 실패했습니다.");
    } finally {
      setBusy(null);
    }
  }

  async function handleTest() {
    setBusy("test");
    setTestResult(null);
    try {
      setTestResult(await testTelegram());
    } catch {
      setTestResult({ ok: false, error: "테스트 발송 요청에 실패했습니다." });
    } finally {
      setBusy(null);
    }
  }

  async function handleDisconnect() {
    if (!window.confirm("Telegram 연결을 해제할까요? 다시 연결하려면 봇에게 /start를 다시 보내야 합니다.")) return;
    onUpdate(await disconnectNotificationChannel("telegram"));
  }

  return (
    <div className="notifyChannelCard">
      <div className="notifyChannelHeaderRow">
        <span className="notifyChannelName">Telegram</span>
        {telegram.connected ? (
          <span className="notifyChannelStatus connected">연결됨</span>
        ) : (
          <button type="button" className="notifyConnectButton" onClick={() => setExpanded((v) => !v)}>
            연결하기
          </button>
        )}
      </div>
      {telegram.connected && (
        <div className="notifyChannelConnectedRow">
          <span className="notifyChannelTarget">{telegram.target || telegram.chat_id_masked}</span>
          <button type="button" className="notifyInlineButton" onClick={handleTest} disabled={busy === "test"}>
            {busy === "test" ? "발송 중…" : "테스트 발송"}
          </button>
          <button type="button" className="notifyInlineButton danger" onClick={handleDisconnect}>
            해제
          </button>
        </div>
      )}
      {telegram.connected && <TestResultNote result={testResult} />}
      {!telegram.connected && expanded && botUsername === null && (
        // 봇 username이 서버에 설정돼 있지 않다 -- 존재하지 않는
        // 봇으로 향하는 죽은 링크를 보여주는 대신, 설정 누락임을 알리고
        // 코드 입력도 막는다(보낼 곳이 없다). 오류색은 쓰지 않는다.
        <p className="notifyChannelMuted">
          텔레그램 봇이 설정되지 않았습니다. 서버 환경변수 TELEGRAM_BOT_USERNAME을 확인하세요.
        </p>
      )}
      {!telegram.connected && expanded && botUsername !== null && (
        <div className="notifyConnectForm">
          <p className="notifyChannelWarning">
            ⚠ 사용자 이름으로는 발송할 수 없습니다.
            <br />
            텔레그램 정책상 봇이 먼저 대화를 시작할 수 없습니다.
          </p>
          <ol className="notifyConnectSteps">
            <li>
              <strong>아래 봇 링크 열기</strong>
              <span>@{botUsername}</span>
            </li>
            <li>
              <strong>봇에게 /start 전송</strong>
            </li>
            <li>
              <strong>받은 인증 코드 입력</strong>
              <span>6자리</span>
            </li>
          </ol>
          <div className="notifyFormActions">
            <a
              className="notifyInlineButton"
              href={`https://t.me/${botUsername}`}
              target="_blank"
              rel="noreferrer noopener"
            >
              봇 열기
            </a>
            <input
              type="text"
              className="notifyCodeInput"
              value={code}
              onChange={(event) => setCode(event.target.value)}
              placeholder="인증 코드 6자리"
              maxLength={6}
            />
          </div>
          {/* 코드는 메모리에 10분만 보관되고 서버 재시작 시
              사라진다 -- 사용자가 시간을 넘기거나 재시작 직후 재도전하는
              이유를 알 수 있게 미리 알린다. */}
          <p className="notifyChannelNote">봇이 보낸 6자리 코드를 10분 이내에 입력하세요.</p>
          {error && <p className="notifyFieldError">{error}</p>}
          <button type="button" className="notifyPrimaryButton" onClick={handleVerify} disabled={busy !== null}>
            {busy === "verify" ? "확인 중…" : "연결"}
          </button>
        </div>
      )}
    </div>
  );
}

function GmailCard({ summary, onUpdate }: ChannelProps) {
  const { gmail } = summary;
  const [expanded, setExpanded] = useState(false);
  const [email, setEmail] = useState("");
  const [busy, setBusy] = useState<"test" | "connect" | null>(null);
  const [error, setError] = useState("");
  const [testResult, setTestResult] = useState<{ ok: boolean; error: string | null } | null>(null);
  // "인증 메일 발송됨" 안내는 서버의 영속 `pending`
  // 상태가 아니라 이 세션에서 방금 발송했는지만 나타내는 로컬 상태다.
  // `pending`을 폼 렌더 조건에 쓰면 메일을 못 받거나 주소를 잘못
  // 입력했을 때 다시 시도할 방법이 없어진다 -- 패널을 닫았다 열면(이
  // 컴포넌트가 언마운트/재마운트되며) 자동으로 초기화된다. 서버·
  // localStorage에는 절대 저장하지 않는다.
  const [justSent, setJustSent] = useState(false);

  async function handleConnect() {
    if (!email.trim()) {
      setError("수신 이메일을 입력하세요.");
      return;
    }
    setError("");
    setBusy("connect");
    try {
      onUpdate(await connectGmail(email.trim()));
      setJustSent(true);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "연결에 실패했습니다.");
    } finally {
      setBusy(null);
    }
  }

  async function handleTest() {
    setBusy("test");
    setTestResult(null);
    try {
      setTestResult(await testGmail());
    } catch {
      setTestResult({ ok: false, error: "테스트 발송 요청에 실패했습니다." });
    } finally {
      setBusy(null);
    }
  }

  async function handleDisconnect() {
    if (!window.confirm("메일 연결을 해제할까요? 다시 연결하려면 인증 메일을 새로 받아야 합니다.")) return;
    try {
      onUpdate(await disconnectNotificationChannel("gmail"));
    } catch {
      // unhandled rejection이 아니라 눈에 보이는 오류로.
      setError("연결 해제에 실패했습니다. 다시 시도해 주세요.");
    }
  }

  return (
    <div className="notifyChannelCard">
      <div className="notifyChannelHeaderRow">
        <span className="notifyChannelName">Gmail</span>
        {gmail.connected ? (
          <span className="notifyChannelStatus connected">연결됨</span>
        ) : gmail.pending ? (
          <>
            <span className="notifyChannelStatus pending">대기 중</span>
            {/* pending일 때도 폼을 다시 열 수 있어야
                한다 -- 메일을 못 받았거나 주소를 잘못 입력한 경우의 유일한
                복구 경로다. */}
            <button type="button" className="notifyConnectButton" onClick={() => setExpanded((v) => !v)}>
              주소 변경
            </button>
          </>
        ) : (
          <button type="button" className="notifyConnectButton" onClick={() => setExpanded((v) => !v)}>
            연결하기
          </button>
        )}
      </div>
      {gmail.connected && (
        <>
          <div className="notifyChannelConnectedRow">
            <span className="notifyChannelTarget">{gmail.email}</span>
            <button type="button" className="notifyInlineButton" onClick={handleTest} disabled={busy === "test"}>
              {busy === "test" ? "발송 중…" : "테스트 발송"}
            </button>
            <button type="button" className="notifyInlineButton danger" onClick={handleDisconnect}>
              해제
            </button>
          </div>
          {error && <p className="notifyFieldError">{error}</p>}
          <TestResultNote result={testResult} />
        </>
      )}
      {justSent && !gmail.connected && (
        <p className="notifyChannelPendingNote">{email.trim() || gmail.email}로 인증 메일이 발송되었습니다. 메일의 링크를 눌러야 연결이 완료됩니다.</p>
      )}
      {!gmail.connected && expanded && (
        <div className="notifyConnectForm">
          <label className="notifyFieldLabel">
            수신 이메일
            <input
              type="email"
              value={email}
              onChange={(event) => {
                setEmail(event.target.value);
                // 주소를 고치기 시작하면 직전 발송 안내는 유효하지 않다
                // -- 남겨두면 새 주소를 입력 중인데 바뀌기 전 주소로
                // 보냈다는 문구가 그대로 떠 있어 혼동을 준다.
                setJustSent(false);
              }}
              placeholder="name@company.com"
            />
          </label>
          <p className="notifyChannelNote">인증 메일이 발송됩니다. 메일의 링크를 눌러야 연결이 완료됩니다.</p>
          {error && <p className="notifyFieldError">{error}</p>}
          <TestResultNote result={testResult} />
          <button type="button" className="notifyPrimaryButton" onClick={handleConnect} disabled={busy !== null}>
            {busy === "connect" ? "발송 중…" : justSent ? "인증 메일 다시 보내기" : "인증 메일 발송"}
          </button>
        </div>
      )}
    </div>
  );
}

// "자동화" 섹션 -- refreshIntervalMinutes마다 서버에서 최신 CSV를
// 받아 수율 예측만 계산해 알림을 보낸다(SD-2). 비밀번호 입력칸은 없다 --
// 서버 환경변수(DB_PASSWORD)로만 받는다("하지 말 것").
function AutomationSection({ summary, onUpdate }: ChannelProps) {
  const { automation } = summary;
  const [enabled, setEnabled] = useState(automation.enabled);
  const [sqlHost, setSqlHost] = useState(automation.sql_host);
  const [sqlPort, setSqlPort] = useState(automation.sql_port);
  const [sqlDb, setSqlDb] = useState(automation.sql_db);
  const [sqlUser, setSqlUser] = useState(automation.sql_user);
  const [refreshMinutes, setRefreshMinutes] = useState(String(automation.refresh_interval_minutes || 60));
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [testBusy, setTestBusy] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; error: string | null } | null>(null);

  async function handleSave() {
    setSaving(true);
    setError("");
    setMessage("");
    try {
      const minutes = Number(refreshMinutes) || 60;
      onUpdate(
        await saveAutomationSettings({
          enabled,
          sql_host: sqlHost,
          sql_port: sqlPort,
          sql_db: sqlDb,
          sql_user: sqlUser,
          refresh_interval_minutes: minutes,
        }),
      );
      setMessage("저장했습니다.");
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "저장하지 못했습니다.");
    } finally {
      setSaving(false);
    }
  }

  async function handleTest() {
    setTestBusy(true);
    setTestResult(null);
    try {
      setTestResult(await testAutomationConnection());
    } catch {
      setTestResult({ ok: false, error: "연결 테스트 요청에 실패했습니다." });
    } finally {
      setTestBusy(false);
    }
  }

  return (
    <section className="settingsSection">
      <h3>자동화</h3>
      <p className="settingsSectionDesc">
        화면 없이 수율 예측만 계산해 알림을 보냅니다 -- 모니터링·트리맵·원인 분석은 계산하지 않습니다. 비밀번호는 서버
        환경변수(DB_PASSWORD)로 설정합니다.
      </p>
      <label className="scatterViewToggleBtn notifyAutomationEnableRow" style={{ display: "inline-flex", alignItems: "center", gap: "6px" }}>
        <input type="checkbox" checked={enabled} onChange={(event) => setEnabled(event.target.checked)} />
        자동화 사용
      </label>
      <div className="trainingSqlRow">
        <label className="notifyFieldLabel">
          서버 주소
          <input type="text" value={sqlHost} onChange={(event) => setSqlHost(event.target.value)} placeholder="db.internal" />
        </label>
        <label className="notifyFieldLabel trainingPortField">
          포트
          <input type="text" value={sqlPort} onChange={(event) => setSqlPort(event.target.value)} placeholder="5432" />
        </label>
      </div>
      <div className="trainingSqlRow">
        <label className="notifyFieldLabel">
          DB명
          <input type="text" value={sqlDb} onChange={(event) => setSqlDb(event.target.value)} placeholder="suni_prod" />
        </label>
        <label className="notifyFieldLabel">
          사용자명
          <input type="text" value={sqlUser} onChange={(event) => setSqlUser(event.target.value)} placeholder="suni_reader" />
        </label>
      </div>
      <label className="notifyFieldLabel">
        Refresh Time (분마다 최신 데이터를 받아 수율 예측 갱신)
        <input type="number" min={1} value={refreshMinutes} onChange={(event) => setRefreshMinutes(event.target.value)} placeholder="60" />
      </label>
      {/* 자동 발송 시점은 이 주기 하나로만 결정되므로(별도 "발송 시점"
          설정이 없다), 무엇이 자동으로 일어나는지 여기서 바로 설명한다. */}
      <p className="settingsSectionDesc">
        Refresh Time마다 데이터를 불러와 수율 예측을 계산하고 알림을 발송합니다.
      </p>
      <p className="settingsSectionDesc">
        마지막 실행{" "}
        {automation.last_run_at
          ? `${formatLastRun(automation.last_run_at)} · ${
              automation.last_run_status === "skipped"
                ? "건너뜀"
                : automation.last_run_status === "error"
                  ? "오류"
                  : `알림 ${automation.last_run_sent_count ?? 0}건 발송`
            }`
          : "없음"}
      </p>
      <div className="notifyFormActions">
        <button type="button" className="button secondary" onClick={() => void handleTest()} disabled={testBusy}>
          {testBusy ? "테스트 중…" : "연결 테스트"}
        </button>
        <button type="button" className="button primary" onClick={() => void handleSave()} disabled={saving}>
          {saving ? "저장 중…" : "저장"}
        </button>
      </div>
      {testResult && (
        <p className={`notifyTestResult ${testResult.ok ? "ok" : "error"}`}>
          {testResult.ok ? "연결에 성공했습니다." : testResult.error || "연결에 실패했습니다."}
        </p>
      )}
      {error && <p className="notifyFieldError">{error}</p>}
      {message && <p className="notifyTestResult ok">{message}</p>}
    </section>
  );
}
