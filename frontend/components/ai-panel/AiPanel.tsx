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
// Second position is the alarm question deliberately (사전 알람 로그가 주요
// 기능이므로 노출도가 높아야 한다) -- every chip here must be answerable
// from the context JSON alone (alarms.records/targets[]), never a question
// that needs physical-mechanism knowledge the data can't provide.
const EXAMPLE_QUERIES = ["알람이 가장 많은 인자는?", "Y2에 영향이 큰 인자는?", "관리한계는 어떻게 정했나요?"];
const REPORT_KEYWORD_PATTERN = /보고서|리포트|report|요약해줘|정리해줘/i;
// Reveals streamed text one character per tick regardless of how large the
// underlying network chunk was (spec §5-3: "한 글자씩 이어 붙인다").
const TYPEWRITER_INTERVAL_MS = 14;
const AUTO_SCROLL_THRESHOLD_PX = 40;
const HISTORY_MESSAGES = 4; // last 2 user/suni turns

export default function AiPanel({
  open,
  onToggle,
}: {
  open: boolean;
  onToggle: () => void;
}) {
  const { analysisDataset, pendingChatRequest, clearPendingChatRequest, setAiPanelOpen } = usePanelState();
  const [messages, setMessages] = useState<ChatMessage[]>(INITIAL_MESSAGES);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [lastRequest, setLastRequest] = useState<{ message: string; mode: ChatMode } | null>(null);

  const messagesRef = useRef<HTMLDivElement | null>(null);
  const cancelRef = useRef<{ cancel: () => void } | null>(null);
  const queueRef = useRef<string[]>([]);
  const doneRef = useRef(false);
  const timerRef = useRef<number | null>(null);
  const messagesStateRef = useRef<ChatMessage[]>(messages);

  useEffect(() => {
    messagesStateRef.current = messages;
  }, [messages]);

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
                      <div className="aiPanelChipRow">
                        <button
                          type="button"
                          className="aiPanelChip aiPanelChipPrimary"
                          disabled={streaming}
                          onClick={() => send("분석 보고서 생성", "report")}
                        >
                          분석 보고서 생성
                        </button>
                        {EXAMPLE_QUERIES.map((query) => (
                          <button
                            key={query}
                            type="button"
                            className="aiPanelChip"
                            disabled={streaming}
                            onClick={() => send(query)}
                          >
                            {query}
                          </button>
                        ))}
                      </div>
                    )}
                  </div>
                );
              })}
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
