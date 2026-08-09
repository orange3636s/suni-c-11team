"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { usePanelState } from "@/components/PanelStateProvider";
import SuniAvatar from "@/components/SuniAvatar";
import { type ChatErrorKind, type ChatHistoryTurn, type ChatMode, streamChat } from "@/lib/api";
import { useFocusTrap } from "@/lib/useFocusTrap";
import { useIsMobileLayout, useIsTabBarLayout } from "@/lib/useMediaQuery";

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

type ChipIconKind = "chart" | "help" | "alert" | "info";
type ExampleChip = { icon: ChipIconKind; text: string };

// HC그룹: 프리셋을 네 갈래(해석/기준/한계/기능)로 나눈다 -- 이전에는
// 데이터 해석 질문만 있어서 판정 기준·한계·기능을 물어봐도 되는지
// 사용자가 몰랐다. 카테고리마다 아이콘을 고정 배정한다(해석 chart,
// 기준 help, 한계 alert, 기능 info) -- 두 마퀴 행에 두 카테고리씩
// 묶어 방향(좌/우)이 서로 다른 질문 성격을 나타내게 한다. 마퀴는
// 뷰포트 폭만큼만 동시에 보이므로(320px 패널 기준 한 행에 2~3개) 총
// 16개를 다 넣어도 "한 번에 6~8개만 보이게" 조건을 자연히 만족한다.
// 여기 있는 모든 문항은 SUNI 시스템 프롬프트(prompts/chat_system.md)가
// 실제로 답할 수 있는 근거(입력 JSON 또는 GC그룹에서 추가한 배경
// 지식/기능 안내 절)를 갖고 있다 -- 답할 수 없는 프리셋은 넣지 않는다.
const MARQUEE_ROW_1: ExampleChip[] = [
  // 데이터 해석
  { icon: "chart", text: "가장 신뢰도 높은 인자는?" },
  { icon: "chart", text: "권장 구간은 무엇인가요?" },
  { icon: "chart", text: "Step16_R1을 어느 쪽으로 조정해야 하나요?" },
  { icon: "chart", text: "지금 가장 위험한 wafer는?" },
  { icon: "chart", text: "Y1~Y5 중 어디에 집중해야 하나요?" },
  // 판정 기준
  { icon: "help", text: "심각·위험·주의는 어떻게 나뉘나요?" },
  { icon: "help", text: "민감도를 올리면 무엇이 달라지나요?" },
  { icon: "help", text: "상관성 강함·보통 기준이 뭔가요?" },
  { icon: "help", text: "미분류는 왜 생기나요?" },
];
const MARQUEE_ROW_2: ExampleChip[] = [
  // 한계·신뢰도
  { icon: "alert", text: "이 분석의 한계는?" },
  { icon: "alert", text: "예측 구간이 왜 이렇게 넓나요?" },
  { icon: "alert", text: "계측률이 낮으면 어떤 문제가 있나요?" },
  { icon: "alert", text: "왜 알람이 없나요?" },
  // 기능 안내 (GC그룹에서 챗봇 프롬프트에 기능 안내 절을 추가한 뒤에만
  // 넣는다 -- prompts/chat_system.md "## 대시보드 기능 안내" 참고)
  { icon: "info", text: "이 화면은 무엇을 보여주나요?" },
  { icon: "info", text: "자동 갱신은 어떻게 동작하나요?" },
  { icon: "info", text: "알림은 언제 발송되나요?" },
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
  // ≥1024px: unchanged floating circle<->320px panel (spec §B-5 row 1).
  // 768~1023px: overlay drawer, 768px 이하로는 못 내려간다 -- see CSS.
  // ≤767px: full-screen drawer.
  const isTabBarLayout = useIsTabBarLayout();
  const isMobileLayout = useIsMobileLayout();
  const layout: "desktop" | "overlay" | "fullscreen" = isMobileLayout ? "fullscreen" : isTabBarLayout ? "overlay" : "desktop";
  // G-4: 오버레이/전체화면일 때만 진짜 모달이다 -- 데스크톱은 항상 떠 있는
  // 도킹 패널이라 포커스를 가두면 평소 페이지 탐색이 막힌다.
  const isDrawer = layout !== "desktop";
  const panelRef = useRef<HTMLElement | null>(null);
  useFocusTrap(panelRef, isDrawer && open);
  const [messages, setMessages] = useState<ChatMessage[]>(INITIAL_MESSAGES);
  const [draft, setDraft] = useState("");
  const [streaming, setStreaming] = useState(false);
  const [autoScroll, setAutoScroll] = useState(true);
  const [lastRequest, setLastRequest] = useState<{ message: string; mode: ChatMode } | null>(null);

  // Locks background scroll while the drawer covers the screen (spec
  // §B-5: "열리면 뒤 본문 스크롤을 잠근다") -- desktop's fixed-width side
  // panel never scrolls the body regardless, so this only ever applies
  // for overlay/fullscreen.
  useEffect(() => {
    if (layout === "desktop" || !open) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [layout, open]);

  // G-4: 오버레이/전체화면일 때만 모달로 동작한다(데스크톱은 항상 떠 있는
  // 도킹 패널이라 Esc로 닫히면 안 된다) -- SettingsPanel/TrainingPanel과
  // 같은 조건으로 Esc를 처리한다.
  useEffect(() => {
    if (layout === "desktop" || !open) return;
    function handleKey(event: KeyboardEvent) {
      if (event.key === "Escape") onToggle();
    }
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [layout, open, onToggle]);

  // `open`'s default comes from a cookie written by (and defaulting true
  // for) the desktop side panel, which used to be easy to ignore -- as a
  // full-screen/overlay takeover that same default would otherwise cover
  // the entire first paint on every fresh mobile visit before the user
  // has done anything. Closes it exactly once, the first time `layout`
  // is actually confirmed non-desktop (not on the transient SSR-safe
  // "desktop" guess useMediaQuery starts every render with -- see that
  // hook's own comment) while still sitting on the inherited default;
  // `autoClosedOnce` marks completion only once the close has actually
  // happened, so a user reopening it afterward is never refought.
  const autoClosedOnce = useRef(false);
  useEffect(() => {
    if (autoClosedOnce.current) return;
    if (layout !== "desktop" && open) {
      autoClosedOnce.current = true;
      onToggle();
    }
  }, [layout, open, onToggle]);

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

  // HC-2: 대화가 시작되면 프리셋 영역을 접는다 -- 이미 질문한 뒤에는
  // 그 자리를 차지할 이유가 없다. welcome 메시지 하나만 있는 상태를
  // "아직 대화를 시작하지 않음"으로 본다.
  const conversationStarted = messages.some((message) => message.id !== WELCOME_ID);

  // Measures the floating chip block's real rendered height so the message
  // list can reserve exactly that much room (spec: "하드코딩하지 마라") --
  // a fixed guess would drift the moment the block's own content wraps
  // differently (font scaling, browser zoom, translation-length changes).
  // 대화가 시작돼 이 블록이 언마운트되면 관찰할 대상이 없다 -- 이전에
  // 측정된 높이를 그대로 CSS 변수에 남겨두면 메시지 목록이 더 이상
  // 존재하지 않는 프리셋 블록만큼 여백을 계속 예약해 공간을 낭비한다.
  useEffect(() => {
    const body = bodyRef.current;
    if (!body) return;
    if (conversationStarted) {
      body.style.setProperty("--chip-float-height", "0px");
      return;
    }
    const el = chipFloatRef.current;
    if (!el) return;
    const observer = new ResizeObserver((entries) => {
      const height = entries[0]?.contentRect.height;
      if (height != null) body.style.setProperty("--chip-float-height", `${height}px`);
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, [conversationStarted]);

  useEffect(() => {
    return () => {
      cancelRef.current?.cancel();
      if (timerRef.current !== null) window.clearInterval(timerRef.current);
    };
  }, []);

  // D-5: 데이터셋이 바뀌면 대화 히스토리를 이어가지 않는다 -- 최근 4턴이
  // 그대로 다음 요청에 실려 나가므로(HISTORY_MESSAGES), train에서 나눈
  // 문답이 test 재분석 후 질문의 맥락으로 딸려가 답이 오염된다. 최초
  // 마운트 값은 건드리지 않는다(그때는 "바뀐" 게 아니다).
  const previousAnalysisDatasetRef = useRef(analysisDataset);
  useEffect(() => {
    if (previousAnalysisDatasetRef.current === analysisDataset) return;
    previousAnalysisDatasetRef.current = analysisDataset;
    cancelRef.current?.cancel();
    stopTypewriter();
    queueRef.current = [];
    doneRef.current = false;
    setStreaming(false);
    setMessages(INITIAL_MESSAGES);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [analysisDataset]);

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

  // D-4: 에러 시 이미 받은 내용을 지우지 않는다 -- `flushText`는 아직
  // 타자기 효과로 화면에 안 찍힌 채 큐에 남아 있던 문자들(진짜로 받은
  // 응답의 일부)이고, `errorText`가 있으면 텍스트를 통째로 교체하는
  // 대신 그 뒤에 덧붙인다. 성공 종료(`finalizeMessage(id, "done")`)는
  // 둘 다 안 넘기므로 기존 동작 그대로다.
  function finalizeMessage(id: string, status: MessageStatus, errorText?: string, errorKind?: ChatErrorKind, flushText?: string) {
    setMessages((current) =>
      current.map((message) => {
        if (message.id !== id) return message;
        const withFlush = flushText ? message.text + flushText : message.text;
        const withError = errorText !== undefined ? `${withFlush}${withFlush ? "\n\n" : ""}⚠ ${errorText}` : withFlush;
        return { ...message, status, errorKind, text: withError };
      }),
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
            // F-3: `chunk.split("")`은 UTF-16 서로게이트 쌍을 반으로
            // 쪼개 이모지 등이 스트리밍 중 깨져 보인다 -- 코드 포인트
            // 단위로 순회하는 `Array.from`을 쓴다.
            queueRef.current.push(...Array.from(chunk));
          },
          onDone: () => {
            doneRef.current = true;
          },
          onError: (message, errorKind) => {
            doneRef.current = true;
            stopTypewriter();
            // D-4: 큐에 남아 있던(아직 타자기로 안 찍힌) 문자도 실제로
            // 받은 응답이다 -- 버리지 않고 먼저 합친 뒤 에러를 덧붙인다.
            const remaining = queueRef.current.join("");
            queueRef.current = [];
            finalizeMessage(suniId, "error", message, errorKind, remaining);
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

  // Overlay/fullscreen: the header's own SUNI button is the only way to
  // open (spec §B-5) -- there is no floating circle to render while
  // closed, unlike the desktop panel which stays visible collapsed.
  if (layout !== "desktop" && !open) return null;

  return (
    <>
      {/* Overlay (768~1023px) gets a dismiss-on-tap backdrop; full-screen
          (≤767px) covers everything itself, no backdrop needed. */}
      {layout === "overlay" && <div className="aiPanelBackdrop" onClick={onToggle} aria-hidden="true" />}
      <aside
        ref={panelRef}
        className={`aiPanel ${layout === "desktop" && !open ? "collapsed" : ""} ${layout === "overlay" ? "aiPanelOverlay" : ""} ${layout === "fullscreen" ? "aiPanelFullscreen" : ""}`}
        aria-label="SUNI AI 어시스턴트"
        role={isDrawer ? "dialog" : undefined}
        aria-modal={isDrawer ? true : undefined}
        tabIndex={isDrawer ? -1 : undefined}
      >
        <div className="aiPanelSurface">
          {isDrawer ? (
            <div className="aiPanelMobileHeader">
              <button type="button" className="aiPanelBackButton" onClick={onToggle} aria-label="닫기">
                <BackIcon />
              </button>
              <span className="aiPanelMobileTitle">SUNI AI 어시스턴트</span>
              {/* 지시서 N-7: 어느 데이터셋 기준으로 답하는지 헤더에 항상 보인다. */}
              <span className="aiPanelCtx">CTX {analysisDataset ?? "미실행"}</span>
            </div>
          ) : (
            <button
              type="button"
              className="shellLogoBlock"
              onClick={onToggle}
              aria-label={open ? "SUNI 접기" : "SUNI 펼치기"}
              aria-expanded={open}
            >
              <SuniAvatar size={open ? 28 : 32} />
              {open && (
                <span className="shellLogoBlockTitleGroup">
                  <span className="shellLogoBlockTitle">SUNI AI 어시스턴트</span>
                  <span className="aiPanelCtx">CTX {analysisDataset ?? "미실행"}</span>
                </span>
              )}
              <span className="shellLogoBlockChevron" aria-hidden="true">
                <ChevronIcon direction={open ? "right" : "left"} />
              </span>
            </button>
          )}

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
                  conversation (spec A-1/A-2). HC-2: 대화가 시작되면(사용자가
                  한 번이라도 메시지를 보내면) 이 블록은 더 이상 그리지
                  않는다 -- 이미 질문한 뒤에는 프리셋이 자리를 차지할
                  이유가 없다. */}
              {!conversationStarted && (
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
              )}
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
    </>
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
        {items.map((chip, index) => {
          // F-3: 뒤 절반은 이음매를 감추기 위한 시각적 복제본이다 -- Tab
          // 키로 같은 칩을 두 번 지나가지 않도록 포커스/스크린리더에서
          // 뺀다.
          const isDuplicate = !reducedMotion && index >= chips.length;
          return (
            <button
              key={`${chip.text}-${index}`}
              type="button"
              className="aiPanelMarqueeChip"
              disabled={disabled}
              onClick={() => onSend(chip.text)}
              aria-hidden={isDuplicate || undefined}
              tabIndex={isDuplicate ? -1 : undefined}
            >
              <ChipIcon kind={chip.icon} />
              {chip.text}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function ChipIcon({ kind }: { kind: ChipIconKind }) {
  switch (kind) {
    // 해석
    case "chart":
      return (
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M3 3v18h18" />
          <path d="M18 17V9" />
          <path d="M13 17V5" />
          <path d="M8 17v-3" />
        </svg>
      );
    // 기준
    case "help":
      return (
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <circle cx="12" cy="12" r="10" />
          <path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3" />
          <path d="M12 17h.01" />
        </svg>
      );
    // 한계·신뢰도
    case "alert":
      return (
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z" />
          <path d="M12 9v4" />
          <path d="M12 17h.01" />
        </svg>
      );
    // 기능 안내
    case "info":
      return (
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <circle cx="12" cy="12" r="10" />
          <path d="M12 16v-5" />
          <path d="M12 8h.01" />
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

function BackIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="m15 6-6 6 6 6" />
    </svg>
  );
}
