"use client";

import { useState, type RefObject } from "react";
import { exportNodeAsPng } from "@/lib/chartExport";

const DEFAULT_TITLE = "이미지로 저장 (PNG)";

type SelfCaptureProps = {
  /** 캡처할 카드 루트 DOM 노드 -- 버튼이 클릭 시점에 스스로 캡처한다
   * (산점도·박스플롯·파레토). */
  nodeRef: RefObject<HTMLElement | null>;
  buildOptions: () => { filename: string; captionText: string };
  onClick?: never;
  busy?: never;
};

type ControlledProps = {
  nodeRef?: never;
  buildOptions?: never;
  /** 캡처 전에 별도 준비가 필요한 호출부(예: 히트맵의 "화면 밖에 전체 행
   * 사본을 잠깐 마운트")가 캡처 로직을 직접 들고 있을 때 쓴다. */
  onClick: () => void;
  busy: boolean;
};

type ChartExportButtonProps = (SelfCaptureProps | ControlledProps) & { title?: string };

/** 파레토·산점도·박스플롯·상관 히트맵 네 차트 카드가 공유하는 이미지 저장
 * 버튼 -- 위치(제목 옆)·크기·스타일을 이 컴포넌트 하나로 통일한다. 각자
 * 버튼을 따로 그리면 스타일이 다시 갈라진다. */
export default function ChartExportButton(props: ChartExportButtonProps) {
  const [selfBusy, setSelfBusy] = useState(false);
  const isControlled = props.onClick != null;

  async function handleSelfCapture() {
    const node = props.nodeRef?.current;
    if (!node || selfBusy) return;
    setSelfBusy(true);
    try {
      await exportNodeAsPng(node, props.buildOptions!());
    } catch (error) {
      console.warn("차트 이미지 저장 실패", error);
    } finally {
      setSelfBusy(false);
    }
  }

  const busy = isControlled ? props.busy : selfBusy;
  const handleClick = isControlled ? props.onClick : () => void handleSelfCapture();

  return (
    <button
      type="button"
      className="chartExportButton"
      onClick={handleClick}
      disabled={busy}
      title={props.title ?? DEFAULT_TITLE}
    >
      ⬇ 이미지 저장
    </button>
  );
}
