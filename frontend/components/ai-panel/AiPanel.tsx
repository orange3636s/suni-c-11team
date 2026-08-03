"use client";

import { useState } from "react";
import SuniAvatar from "@/components/SuniAvatar";

// Stub for the future live SUNI assistant. Swap this implementation to call
// the real backend once one exists -- callers already await a Promise<string>.
async function sendMessage(message: string): Promise<string> {
  void message;
  throw new Error("SUNI is not connected yet.");
}
void sendMessage; // referenced by future wiring, unused for now

type ChatMessage = {
  id: string;
  from: "suni" | "user";
  text: string;
};

const INITIAL_MESSAGES: ChatMessage[] = [
  { id: "welcome", from: "suni", text: "무엇을 도와드릴까요?" },
];

export default function AiPanel({
  open,
  onToggle,
}: {
  open: boolean;
  onToggle: () => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>(INITIAL_MESSAGES);
  const [draft, setDraft] = useState("");

  function submit() {
    const text = draft.trim();
    if (!text) return;
    setMessages((current) => [
      ...current,
      { id: `user-${Date.now()}`, from: "user", text },
    ]);
    setDraft("");
  }

  if (!open) {
    return (
      <aside className="aiPanel collapsed">
        <button
          type="button"
          className="shellCircleButton"
          onClick={onToggle}
          aria-label="SUNI 채팅 열기"
          title="SUNI"
          aria-expanded={false}
        >
          <SuniAvatar size={34} />
          <span className="shellCircleBadge" aria-hidden="true">
            <ChatBadgeIcon />
          </span>
        </button>
      </aside>
    );
  }

  return (
    <aside className="aiPanel" aria-label="SUNI 어시스턴트">
      <div className="aiPanelHeader">
        <SuniAvatar size={32} />
        <div className="aiPanelHeaderText">
          <strong>SUNI</strong>
          <span>어시스턴트</span>
        </div>
        <button
          type="button"
          className="aiPanelCollapseButton"
          onClick={onToggle}
          aria-label="SUNI 패널 접기"
          aria-expanded={true}
        >
          <ChevronRightIcon />
        </button>
      </div>

      <div className="aiPanelMessages">
        {messages.map((message, index) => {
          const showAvatar = message.from === "suni" && messages[index - 1]?.from !== "suni";
          return (
            <div key={message.id} className={`aiPanelRow aiPanelRow-${message.from}`}>
              {message.from === "suni" && (
                <span className="aiPanelBubbleAvatar">{showAvatar && <SuniAvatar size={28} />}</span>
              )}
              <div className={`aiPanelBubble aiPanelBubble-${message.from}`}>{message.text}</div>
            </div>
          );
        })}
      </div>

      <div className="aiPanelInputArea">
        <form
          className="aiPanelInputRow"
          onSubmit={(event) => {
            event.preventDefault();
            submit();
          }}
        >
          <input
            type="text"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="메시지를 입력하세요"
            aria-label="SUNI에게 메시지 보내기"
          />
          <button type="submit" className="aiPanelSendButton" disabled={!draft.trim()} aria-label="전송">
            <SendIcon />
          </button>
        </form>
        <p className="aiPanelCaption">SUNI는 아직 연결되지 않았습니다</p>
      </div>
    </aside>
  );
}

function ChatBadgeIcon() {
  return (
    <svg viewBox="0 0 24 24" width="9" height="9" fill="currentColor">
      <path d="M4 4h16a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H9l-4.4 3.3A.6.6 0 0 1 3 18.8V16H4a1 1 0 0 1-1-1V5a1 1 0 0 1 1-1Z" />
    </svg>
  );
}

function ChevronRightIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="m9 6 6 6-6 6" />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="m5 12 14-7-7 14-2-5-5-2Z" />
    </svg>
  );
}
