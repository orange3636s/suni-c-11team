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
