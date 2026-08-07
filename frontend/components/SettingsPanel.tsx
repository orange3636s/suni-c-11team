"use client";

import { X } from "lucide-react";
import { useEffect, useState } from "react";
import { useAnalysisState } from "@/components/AnalysisStateProvider";
import {
  connectGmail,
  connectSlack,
  disconnectNotificationChannel,
  saveNotificationConditions,
  testGmail,
  testSlack,
  testTelegram,
  verifyTelegramCode,
} from "@/lib/api";
import type { NotificationGrade, NotificationSettingsSummary, NotificationTiming } from "@/types/data";

const GRADE_OPTIONS: NotificationGrade[] = ["심각", "위험", "주의"];

export default function SettingsPanel({ open, onClose }: { open: boolean; onClose: () => void }) {
  const { notifications, setNotifications } = useAnalysisState();

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
      <div className="settingsPanel" onClick={(event) => event.stopPropagation()} role="dialog" aria-modal="true" aria-label="알림 설정">
        <div className="settingsPanelHeader">
          <h2>알림 설정</h2>
          <button type="button" className="settingsPanelClose" onClick={onClose} aria-label="닫기">
            <X size={18} />
          </button>
        </div>
        <div className="settingsPanelBody">
          <section className="settingsSection">
            <h3>알림 받기</h3>
            <p className="settingsSectionDesc">심각·위험 등급 알람이 발생하면 선택한 채널로 발송합니다.</p>
            <div className="notifyChannelList">
              <SlackCard summary={notifications} onUpdate={setNotifications} />
              <TelegramCard summary={notifications} onUpdate={setNotifications} />
              <GmailCard summary={notifications} onUpdate={setNotifications} />
            </div>
          </section>

          <section className="settingsSection">
            <h3>발송 조건</h3>
            <p className="settingsSectionDesc">무엇을 언제 보낼지 정합니다.</p>
            <ConditionsForm summary={notifications} onUpdate={setNotifications} />
          </section>
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
    if (!webhookUrl.trim()) {
      setError("Webhook URL을 먼저 입력하세요.");
      return;
    }
    setError("");
    setBusy("test");
    setTestResult(null);
    try {
      setTestResult(await testSlack(webhookUrl.trim()));
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
    onUpdate(await disconnectNotificationChannel("slack"));
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
        <div className="notifyChannelConnectedRow">
          <span className="notifyChannelTarget">{slack.target || slack.webhook_masked}</span>
          <button type="button" className="notifyInlineButton" onClick={handleTest} disabled={busy === "test"}>
            {busy === "test" ? "발송 중…" : "테스트 발송"}
          </button>
          <button type="button" className="notifyInlineButton danger" onClick={handleDisconnect}>
            해제
          </button>
        </div>
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
  const botUsername = process.env.NEXT_PUBLIC_TELEGRAM_BOT_USERNAME || "suni_alarm_bot";

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
      {!telegram.connected && expanded && (
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
  // 버그 수정 (지시서 W): "인증 메일 발송됨" 안내는 서버의 영속 `pending`
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
    onUpdate(await disconnectNotificationChannel("gmail"));
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
            {/* 버그 수정 (지시서 W): pending일 때도 폼을 다시 열 수 있어야
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
        <div className="notifyChannelConnectedRow">
          <span className="notifyChannelTarget">{gmail.email}</span>
          <button type="button" className="notifyInlineButton" onClick={handleTest} disabled={busy === "test"}>
            {busy === "test" ? "발송 중…" : "테스트 발송"}
          </button>
          <button type="button" className="notifyInlineButton danger" onClick={handleDisconnect}>
            해제
          </button>
        </div>
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
                // 주소를 고치기 시작하면 이전 발송 안내는 더 이상 유효하지
                // 않다 -- 남겨두면 새 주소를 입력 중인데 옛 주소로 보냈다는
                // 문구가 그대로 떠 있어 혼동을 준다.
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

function ConditionsForm({ summary, onUpdate }: ChannelProps) {
  const { conditions } = summary;
  const [saving, setSaving] = useState(false);

  async function persist(grades: NotificationGrade[], timing: NotificationTiming) {
    setSaving(true);
    try {
      onUpdate(await saveNotificationConditions({ grades, timing }));
    } finally {
      setSaving(false);
    }
  }

  function toggleGrade(grade: NotificationGrade) {
    const has = conditions.grades.includes(grade);
    const next = has ? conditions.grades.filter((g) => g !== grade) : [...conditions.grades, grade];
    void persist(next, conditions.timing);
  }

  return (
    <div className="notifyConditions">
      <div className="notifyConditionsRow">
        <span className="notifyConditionsLabel">발송 대상 등급</span>
        <div className="notifyGradeToggles">
          {GRADE_OPTIONS.map((grade) => (
            <button
              key={grade}
              type="button"
              className={`notifyGradeToggle ${conditions.grades.includes(grade) ? "active" : ""}`}
              onClick={() => toggleGrade(grade)}
              disabled={saving}
              aria-pressed={conditions.grades.includes(grade)}
            >
              {conditions.grades.includes(grade) ? "✓ " : ""}
              {grade}
            </button>
          ))}
        </div>
      </div>
      <div className="notifyConditionsRow">
        <span className="notifyConditionsLabel">발송 시점</span>
        <div className="notifyTimingToggles">
          <button
            type="button"
            className={`notifyTimingToggle ${conditions.timing === "on_analysis" ? "active" : ""}`}
            onClick={() => void persist(conditions.grades, "on_analysis")}
            disabled={saving}
          >
            분석 실행 직후
          </button>
          <button
            type="button"
            className={`notifyTimingToggle ${conditions.timing === "daily_9am" ? "active" : ""}`}
            onClick={() => void persist(conditions.grades, "daily_9am")}
            disabled={saving}
          >
            매일 오전 9시
          </button>
        </div>
      </div>
      <p className="notifyReliabilityGateNote">
        신뢰도 낮은 데이터셋은 발송하지 않습니다
        <br />
        분석 신뢰도가 낮음 등급이면 알람이 발생해도 발송을 건너뜁니다.
        <br />
        근거 없는 알림이 반복되면 신뢰를 잃습니다.
      </p>
    </div>
  );
}
