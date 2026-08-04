"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { usePanelState } from "@/components/PanelStateProvider";
import SuniAvatar from "@/components/SuniAvatar";
import { type ChatErrorKind, type ChatHistoryTurn, type ChatMode, streamChat } from "@/lib/api";

type MessageStatus = "streaming" | "done" | "error";

type ChatMessage = {
  id: string;
  from: "suni" | "user";
  text: string;
  kind?: ChatMode; // set on suni messages only
  status?: MessageStatus;
  errorKind?: ChatErrorKind;
};

const WELCOME_ID = "welcome";
const INITIAL_MESSAGES: ChatMessage[] = [
  { id: WELCOME_ID, from: "suni", text: "무엇을 도와드릴까요?", status: "done" },
];

type ChipIconKind = "bell" | "chart" | "help" | "warning";
type ExampleChip = { icon: ChipIconKind; text: string };

// Every chip here must be answerable from the context JSON alone
// (targets[]/alarms.records/recommendations.records/config_screening/
// limitations) -- never a question that needs physical-mechanism or
// corrective-action knowledge the data can't provide (the system prompt
// forbids causal/action phrasing, so a chip that invites it would just
// produce a refusal). Row 1 asks about results (bell/chart icons), row 2
// asks about method and limits (help/warning icons) -- kept as two
// semantically distinct rows so their opposite scroll directions read as
// two different kinds of question, not an arbitrary split.
const MARQUEE_ROW_1: ExampleChip[] = [
  { icon: "bell", text: "알람이 가장 많은 인자는?" },
  { icon: "chart", text: "Y2에 영향이 큰 인자는?" },
  { icon: "chart", text: "개선 여지가 가장 큰 인자는?" },
  { icon: "bell", text: "알람 wafer의 수율은 얼마나 낮나요?" },
  { icon: "chart", text: "가장 신뢰도 높은 인자는?" },
];
const MARQUEE_ROW_2: ExampleChip[] = [
  { icon: "help", text: "관리한계는 어떻게 정했나요?" },
  { icon: "help", text: "권장 구간은 무엇인가요?" },
  { icon: "help", text: "Step16_R1을 어느 범위로 관리해야 하나요?" },
  { icon: "warning", text: "판정불가 wafer가 왜 많나요?" },
  { icon: "warning", text: "이 분석의 한계는?" },
];
const REPORT_KEYWORD_PATTERN = /보고서|리포트|report|요약해줘|정리해줘/i;
// Reveals streamed text one character per tick regardless of how large the
// underlying network chunk was (spec §5-3: "한 글자씩 이어 붙인다").
const TYPEWRITER_INTERVAL_MS = 14;
const AUTO_SCROLL_THRESHOLD_PX = 40;
const HISTORY_MESSAGES = 4; // last 2 user/suni turns
// Trailing spacer so the last real message always lands in the mask's fully
// opaque zone, never the fade -- must be >= the fade band's own length
// (48px) or the last line would still dim on auto-scroll.
const MESSAGE_LIST_BOTTOM_SPACER_PX = 56;

export default function AiPanel({
  open,
  onToggle,
}: {
  open: boolean;
  onToggle: () => void;
}) {
  const { analysisDataset, pendingChatRequest, clearPendingChatRequest, setAiPanelOpen } = usePanelState();
  const reducedMotion = usePrefersReducedMotion();
  const [messages, setMessages] = useState<ChatMessage[]>(INITIAL_MESSAGES);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [lastRequest, setLastRequest] = useState<{ message: string; mode: ChatMode } | null>(null);

  const messagesRef = useRef<HTMLDivElement | null>(null);
  const bodyRef = useRef<HTMLDivElement | null>(null);
  const chipFloatRef = useRef<HTMLDivElement | null>(null);
  const cancelRef = useRef<{ cancel: () => void } | null>(null);
  const queueRef = useRef<string[]>([]);
  const doneRef = useRef(false);
  const timerRef = useRef<number | null>(null);
  const messagesStateRef = useRef<ChatMessage[]>(messages);

  useEffect(() => {
    messagesStateRef.current = messages;
  }, [messages]);

  // Measures the floating chip block's real rendered height so the message
  // list can reserve exactly that much room (spec: "하드코딩하지 마라") --
  // a fixed guess would drift the moment the block's own content wraps
  // differently (font scaling, browser zoom, translation-length changes).
  useEffect(() => {
    const el = chipFloatRef.current;
    const body = bodyRef.current;
    if (!el || !body) return;
    const observer = new ResizeObserver((entries) => {
      const height = entries[0]?.contentRect.height;
      if (height != null) body.style.setProperty("--chip-float-height", `${height}px`);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    return () => {
      cancelRef.current?.cancel();
      if (timerRef.current !== null) window.clearInterval(timerRef.current);
    };
  }, []);

  useEffect(() => {
    if (!autoScroll) return;
    const el = messagesRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, autoScroll]);

  function handleMessagesScroll() {
    const el = messagesRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    setAutoScroll(distanceFromBottom < AUTO_SCROLL_THRESHOLD_PX);
  }

  function stopTypewriter() {
    if (timerRef.current !== null) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
  }

  function finalizeMessage(id: string, status: MessageStatus, errorText?: string, errorKind?: ChatErrorKind) {
    setMessages((current) =>
      current.map((message) =>
        message.id === id
          ? { ...message, status, errorKind, ...(errorText !== undefined ? { text: errorText } : {}) }
          : message,
      ),
    );
    setStreaming(false);
  }

  function startTypewriter(id: string) {
    stopTypewriter();
    timerRef.current = window.setInterval(() => {
      if (queueRef.current.length > 0) {
        const char = queueRef.current.shift() as string;
        setMessages((current) =>
          current.map((message) => (message.id === id ? { ...message, text: message.text + char } : message)),
        );
        return;
      }
      if (doneRef.current) {
        stopTypewriter();
        finalizeMessage(id, "done");
      }
    }, TYPEWRITER_INTERVAL_MS);
  }

  const send = useCallback(
    (rawText: string, modeOverride?: ChatMode) => {
      const text = rawText.trim();
      if (!text || streaming) return;

      const mode: ChatMode = modeOverride ?? (REPORT_KEYWORD_PATTERN.test(text) ? "report" : "chat");
      const userId = `user-${Date.now()}`;
      const suniId = `suni-${Date.now()}`;

      const history: ChatHistoryTurn[] = messagesStateRef.current
        .filter((message) => message.id !== WELCOME_ID && message.status !== "error")
        .slice(-HISTORY_MESSAGES)
        .map((message) => ({ role: message.from === "user" ? "user" : "assistant", content: message.text }));

      setMessages((current) => [
        ...current,
        { id: userId, from: "user", text },
        { id: suniId, from: "suni", text: "", kind: mode, status: "streaming" },
      ]);
      setDraft("");
      setLastRequest({ message: text, mode });
      setAutoScroll(true);

      queueRef.current = [];
      doneRef.current = false;
      setStreaming(true);
      startTypewriter(suniId);

      // Whether 원인 분석 has run is frontend-only session state
      // (analysisDataset) -- an empty dataset tells the backend to judge
      // and answer that itself (spec 재지시: "분석이 실행되지 않은 상태면
      // 백엔드가 판단해 안내 메시지를 응답한다"), so every send() always
      // reaches /api/chat the same way regardless of entry point (typed
      // text or an example chip).
      cancelRef.current = streamChat(
        { message: text, mode, dataset: analysisDataset ?? "", history },
        {
          onDelta: (chunk) => {
            queueRef.current.push(...chunk.split(""));
          },
          onDone: () => {
            doneRef.current = true;
          },
          onError: (message, errorKind) => {
            doneRef.current = true;
            queueRef.current = [];
            stopTypewriter();
            finalizeMessage(suniId, "error", message, errorKind);
          },
        },
      );
    },
    // startTypewriter/finalizeMessage close over refs and setMessages only
    // -- stable in effect, safe to omit; including them would recreate
    // `send` (and the textarea's onKeyDown handler) on every keystroke.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [analysisDataset, streaming],
  );

  // Another page (알람/개선 권장 목록's "해설" button) asked SUNI to open
  // and answer a question on its behalf -- open the panel if it's
  // collapsed and send immediately; if it was already open, this just
  // adds the message (spec: "패널이 이미 열려 있으면 그대로 두고 메시지만
  // 추가한다").
  useEffect(() => {
    if (!pendingChatRequest) return;
    const timer = window.setTimeout(() => {
      setAiPanelOpen(true);
      send(pendingChatRequest.message, pendingChatRequest.mode);
      clearPendingChatRequest();
    }, 0);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pendingChatRequest]);

  function retry() {
    if (!lastRequest) return;
    send(lastRequest.message, lastRequest.mode);
  }

  function copyReport(text: string) {
    void navigator.clipboard.writeText(text);
  }

  function saveReport(text: string) {
    const now = new Date();
    const stamp =
      [now.getFullYear(), String(now.getMonth() + 1).padStart(2, "0"), String(now.getDate()).padStart(2, "0")].join("") +
      "_" +
      [String(now.getHours()).padStart(2, "0"), String(now.getMinutes()).padStart(2, "0")].join("");
    const filename = `suni_report_${analysisDataset ?? "train"}_${stamp}.md`;
    const blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  return (
    <aside className={`aiPanel ${open ? "" : "collapsed"}`} aria-label="SUNI AI 어시스턴트">
      <div className="aiPanelSurface">
        <button
          type="button"
          className="shellLogoBlock"
          onClick={onToggle}
          aria-label={open ? "SUNI 접기" : "SUNI 펼치기"}
          aria-expanded={open}
        >
          <SuniAvatar size={open ? 28 : 32} />
          {open && <span className="shellLogoBlockTitle">SUNI AI 어시스턴트</span>}
          <span className="shellLogoBlockChevron" aria-hidden="true">
            <ChevronIcon direction={open ? "right" : "left"} />
          </span>
        </button>

        {open && (
          <>
            <div className="aiPanelBody" ref={bodyRef}>
              <div className="aiPanelMessages" ref={messagesRef} onScroll={handleMessagesScroll}>
              {messages.map((message, index) => {
                const showAvatar = message.from === "suni" && messages[index - 1]?.from !== "suni";
                const isReport = message.from === "suni" && message.kind === "report";
                const isTyping = message.from === "suni" && message.status === "streaming" && message.text === "";
                const canRetry = message.errorKind === "timeout" || message.errorKind === "other";
                return (
                  <div key={message.id}>
                    <div className={`aiPanelRow aiPanelRow-${message.from}`}>
                      {message.from === "suni" && (
                        <span className="aiPanelBubbleAvatar">{showAvatar && <SuniAvatar size={28} />}</span>
                      )}
                      <div
                        className={[
                          "aiPanelBubble",
                          `aiPanelBubble-${message.from}`,
                          isReport ? "aiPanelBubble-report" : "",
                          message.status === "error" ? "aiPanelBubble-errorState" : "",
                        ]
                          .filter(Boolean)
                          .join(" ")}
                      >
                        {message.from === "suni" && message.kind && message.status !== "error" && (
                          <span className="aiPanelAiBadge">AI 생성</span>
                        )}
                        {isTyping ? (
                          <span className="aiPanelTypingDots" aria-label="답변 준비 중">
                            <span />
                            <span />
                            <span />
                          </span>
                        ) : message.from === "suni" ? (
                          <div className="aiPanelMarkdown">
                            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.text}</ReactMarkdown>
                          </div>
                        ) : (
                          message.text
                        )}
                        {message.status === "error" && canRetry && (
                          <button type="button" className="aiPanelRetryButton" onClick={retry}>
                            다시 시도
                          </button>
                        )}
                        {message.status === "done" && message.kind === "report" && (
                          <div className="aiPanelReportActions">
                            <button type="button" onClick={() => copyReport(message.text)}>
                              복사
                            </button>
                            <button type="button" onClick={() => saveReport(message.text)}>
                              저장
                            </button>
                          </div>
                        )}
                      </div>
                    </div>
                    {message.id === WELCOME_ID && (
                      <div className="aiPanelChipsWrap">
                        <button
                          type="button"
                          className="aiPanelChip aiPanelChipPrimary"
                          disabled={streaming}
                          onClick={() => send("분석 보고서 생성", "report")}
                        >
                          분석 보고서 생성
                        </button>
                      </div>
                    )}
                  </div>
                );
              })}
              {/* Keeps the last real message out of the mask's fade band
                  (spec §8-5) -- without this, scrollTop=scrollHeight lands
                  the final line exactly in the fully-transparent zone. */}
              <div aria-hidden="true" style={{ height: MESSAGE_LIST_BOTTOM_SPACER_PX, flex: "0 0 auto" }} />
              </div>

              {/* Floating block: marquee rows only -- "분석 보고서 생성" stays
                  attached to the welcome bubble above and scrolls with the
                  conversation (spec A-1/A-2). Rendered unconditionally
                  (never tied to message count or a collapse toggle) so it
                  can never disappear as the conversation grows. */}
              <div className="aiPanelChipFloat" ref={chipFloatRef}>
                <MarqueeRow
                  chips={MARQUEE_ROW_1}
                  direction="left"
                  disabled={streaming}
                  reducedMotion={reducedMotion}
                  onSend={(text) => send(text)}
                />
                <MarqueeRow
                  chips={MARQUEE_ROW_2}
                  direction="right"
                  disabled={streaming}
                  reducedMotion={reducedMotion}
                  onSend={(text) => send(text)}
                />
              </div>
            </div>

            <div className="aiPanelInputArea">
              <form
                className="aiPanelInputRow"
                onSubmit={(event) => {
                  event.preventDefault();
                  send(draft);
                }}
              >
                <textarea
                  rows={1}
                  value={draft}
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault();
                      send(draft);
                    }
                  }}
                  placeholder="메시지를 입력하세요"
                  aria-label="SUNI에게 메시지 보내기"
                />
                <button type="submit" className="aiPanelSendButton" disabled={!draft.trim() || streaming} aria-label="전송">
                  <SendIcon />
                </button>
              </form>
              {!analysisDataset && <p className="aiPanelCaption">원인 분석을 실행하면 답변할 수 있습니다</p>}
              <p className="aiPanelFooterCaption">AI가 생성한 내용입니다. 수치는 분석 결과를 참조하세요.</p>
            </div>
          </>
        )}
      </div>
    </aside>
  );
}

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => typeof window !== "undefined" && window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );
  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    function handleChange(event: MediaQueryListEvent) {
      setReduced(event.matches);
    }
    query.addEventListener("change", handleChange);
    return () => query.removeEventListener("change", handleChange);
  }, []);
  return reduced;
}

// Continuous marquee: the chip list is rendered twice back-to-back and the
// track animates to exactly -50% of its own width, so the seam between the
// two copies is invisible -- the moment copy 1 scrolls fully offscreen,
// copy 2 is sitting exactly where copy 1 started. `prefers-reduced-motion`
// renders only one copy (no seam to hide) and falls back to native
// horizontal scroll instead of the transform loop.
function MarqueeRow({
  chips,
  direction,
  disabled,
  reducedMotion,
  onSend,
}: {
  chips: ExampleChip[];
  direction: "left" | "right";
  disabled: boolean;
  reducedMotion: boolean;
  onSend: (text: string) => void;
}) {
  const items = reducedMotion ? chips : [...chips, ...chips];
  return (
    <div className="aiPanelMarquee">
      <div className={`aiPanelMarqueeTrack ${direction === "right" ? "reverse" : ""}`}>
        {items.map((chip, index) => (
          <button
            key={`${chip.text}-${index}`}
            type="button"
            className="aiPanelMarqueeChip"
            disabled={disabled}
            onClick={() => onSend(chip.text)}
          >
            <ChipIcon kind={chip.icon} />
            {chip.text}
          </button>
        ))}
      </div>
    </div>
  );
}

function ChipIcon({ kind }: { kind: ChipIconKind }) {
  switch (kind) {
    case "bell":
      return (
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" />
          <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" />
        </svg>
      );
    case "chart":
      return (
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M3 3v18h18" />
          <path d="M18 17V9" />
          <path d="M13 17V5" />
          <path d="M8 17v-3" />
        </svg>
      );
    case "help":
      return (
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <circle cx="12" cy="12" r="10" />
          <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
          <path d="M12 17h.01" />
        </svg>
      );
    case "warning":
      return (
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
          <path d="M12 9v4" />
          <path d="M12 17h.01" />
        </svg>
      );
  }
}

function SendIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="m5 12 14-7-7 14-2-5-5-2Z" />
    </svg>
  );
}

function ChevronIcon({ direction }: { direction: "left" | "right" }) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d={direction === "left" ? "m15 6-6 6 6 6" : "m9 6 6 6-6 6"} />
    </svg>
  );
}
