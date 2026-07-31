"use client";

import type { DragEvent, KeyboardEvent } from "react";
import { useRef, useState } from "react";

type CsvUploadPanelProps = {
  id: string;
  file: File | null;
  onFileSelect: (file?: File) => void;
  disabled?: boolean;
  compact?: boolean;
  title?: string;
  description?: string;
};

export default function CsvUploadPanel({
  id,
  file,
  onFileSelect,
  disabled = false,
  compact = false,
  title = "CSV 파일을 드래그하거나 클릭하여 선택하세요.",
  description = "최대 20MB · 업로드 파일은 서버에 영구 저장되지 않습니다.",
}: CsvUploadPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  function openFilePicker() {
    if (!disabled) inputRef.current?.click();
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDragging(false);
    if (!disabled) onFileSelect(event.dataTransfer.files?.[0]);
  }

  function handleKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openFilePicker();
    }
  }

  return (
    <div
      className={`csvUploadPanel ${compact ? "compact" : ""} ${
        isDragging ? "dragging" : ""
      } ${disabled ? "disabled" : ""}`}
      onClick={openFilePicker}
      onDragEnter={(event) => {
        event.preventDefault();
        if (!disabled) setIsDragging(true);
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onKeyDown={handleKeyDown}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      aria-label={file ? `${file.name} 선택됨. 다른 CSV 선택` : title}
    >
      <input
        ref={inputRef}
        id={id}
        className="visuallyHidden"
        type="file"
        accept=".csv,text/csv"
        disabled={disabled}
        onChange={(event) => {
          onFileSelect(event.target.files?.[0]);
          event.target.value = "";
        }}
      />
      <span className="csvUploadIcon" aria-hidden="true">CSV</span>
      <div className="csvUploadCopy">
        <strong>{file ? file.name : title}</strong>
        <span>{file ? "다른 CSV를 선택하려면 클릭하세요." : description}</span>
      </div>
      <span className="csvUploadButton" aria-hidden="true">
        파일 선택
      </span>
    </div>
  );
}
