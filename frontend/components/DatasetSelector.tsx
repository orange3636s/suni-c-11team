"use client";

import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { deleteDataset, getDatasets, uploadDataset } from "@/lib/api";
import type { DatasetSummary } from "@/types/data";

type DatasetSelectorProps = {
  label: string;
  value: string;
  onChange: (datasetId: string) => void;
  onDatasetsLoaded?: (datasets: DatasetSummary[]) => void;
};

type MenuPosition = { left: number; width: number; top?: number; bottom?: number };

// Menu height isn't known until it renders, so an upward flip is done with
// `bottom` (anchored to the button's top edge) instead of computing a `top`
// that would need the height in advance -- it just grows upward on its own.
const MENU_MAX_HEIGHT = 360;

export default function DatasetSelector({ label, value, onChange, onDatasetsLoaded }: DatasetSelectorProps) {
  const [datasets, setDatasets] = useState<DatasetSummary[]>([]);
  const [open, setOpen] = useState(false);
  const [menuPos, setMenuPos] = useState<MenuPosition | null>(null);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const containerRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function refresh() {
    try {
      const response = await getDatasets();
      setDatasets(response.items);
      onDatasetsLoaded?.(response.items);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "데이터셋 목록을 불러오지 못했습니다.");
    }
  }

  useEffect(() => {
    const timer = window.setTimeout(() => void refresh(), 0);
    return () => window.clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // The menu itself lives in a portal outside `containerRef` (spec §1-4),
  // so an outside-click check needs to also exempt clicks landing inside
  // the portaled menu -- otherwise every option click would immediately
  // read as "outside" and close the menu before onSelect fires.
  useEffect(() => {
    if (!open) return;
    function isOutside(target: Node) {
      if (containerRef.current?.contains(target)) return false;
      if (menuRef.current?.contains(target)) return false;
      return true;
    }
    function handleOutside(event: MouseEvent) {
      if (isOutside(event.target as Node)) setOpen(false);
    }
    function handleKey(event: KeyboardEvent) {
      if (event.key === "Escape") setOpen(false);
    }
    // Simplest correct response to a scroll/resize while open: close,
    // rather than track and re-measure a moving anchor (spec §1-4 allows
    // either).
    function handleScrollOrResize() {
      setOpen(false);
    }
    document.addEventListener("mousedown", handleOutside);
    document.addEventListener("keydown", handleKey);
    window.addEventListener("scroll", handleScrollOrResize, true);
    window.addEventListener("resize", handleScrollOrResize);
    return () => {
      document.removeEventListener("mousedown", handleOutside);
      document.removeEventListener("keydown", handleKey);
      window.removeEventListener("scroll", handleScrollOrResize, true);
      window.removeEventListener("resize", handleScrollOrResize);
    };
  }, [open]);

  // The trigger button can move without a window resize or scroll event --
  // toggling the left/right panel animates the grid's column widths, which
  // shifts this button horizontally. Poll its rect while open and close
  // rather than let the portaled menu drift out of alignment with it.
  useEffect(() => {
    if (!open) return;
    let rafId: number;
    let lastRect = buttonRef.current?.getBoundingClientRect();
    function poll() {
      const rect = buttonRef.current?.getBoundingClientRect();
      if (rect && lastRect && (Math.abs(rect.left - lastRect.left) > 0.5 || Math.abs(rect.top - lastRect.top) > 0.5)) {
        setOpen(false);
        return;
      }
      lastRect = rect;
      rafId = requestAnimationFrame(poll);
    }
    rafId = requestAnimationFrame(poll);
    return () => cancelAnimationFrame(rafId);
  }, [open]);

  function openMenu() {
    const rect = buttonRef.current?.getBoundingClientRect();
    if (!rect) return;
    const width = Math.max(rect.width, 320);
    const spaceBelow = window.innerHeight - rect.bottom;
    const spaceAbove = rect.top;
    const flipUp = spaceBelow < MENU_MAX_HEIGHT && spaceAbove > spaceBelow;
    setMenuPos(
      flipUp
        ? { left: rect.left, width, bottom: window.innerHeight - rect.top + 6 }
        : { left: rect.left, width, top: rect.bottom + 6 },
    );
    setOpen(true);
  }

  const selected = datasets.find((item) => item.dataset_id === value);

  async function handleFileSelected(file: File | undefined) {
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      const result = await uploadDataset(file);
      if (!result.success) {
        setError(result.blocking_errors.join(" ") || "업로드가 거부되었습니다.");
        return;
      }
      await refresh();
      if (result.dataset_id) onChange(result.dataset_id);
      setOpen(false);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "업로드에 실패했습니다.");
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(datasetId: string, event: React.MouseEvent) {
    event.stopPropagation();
    if (!window.confirm("이 데이터셋을 삭제할까요? 연관된 분석 결과도 함께 정리됩니다.")) return;
    try {
      await deleteDataset(datasetId);
      await refresh();
      if (value === datasetId) {
        const fallback = datasets.find((item) => item.dataset_id !== datasetId);
        if (fallback) onChange(fallback.dataset_id);
      }
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "삭제에 실패했습니다.");
    }
  }

  const bundled = datasets.filter((item) => item.kind === "bundled");
  const uploaded = datasets.filter((item) => item.kind === "uploaded");

  return (
    <div className="fieldGroup">
      <span>{label}</span>
      <div className="datasetSelector" ref={containerRef}>
        <button
          ref={buttonRef}
          type="button"
          className="datasetSelectorButton"
          onClick={() => (open ? setOpen(false) : openMenu())}
          aria-haspopup="listbox"
          aria-expanded={open}
        >
          <span className="datasetSelectorButtonLabel" title={selected?.original_filename}>
            {selected ? selected.original_filename : "선택하세요"}
          </span>
          <ChevronIcon />
        </button>
        {open && menuPos && createPortal(
          <div
            ref={menuRef}
            className="datasetSelectorMenu"
            role="listbox"
            style={{ left: menuPos.left, width: menuPos.width, top: menuPos.top, bottom: menuPos.bottom }}
          >
            {bundled.map((item) => (
              <DatasetOption key={item.dataset_id} item={item} active={item.dataset_id === value} onSelect={() => { onChange(item.dataset_id); setOpen(false); }} />
            ))}
            {uploaded.length > 0 && <div className="datasetSelectorDivider" />}
            {uploaded.map((item) => (
              <div className="datasetSelectorItemRow" key={item.dataset_id}>
                <DatasetOption item={item} active={item.dataset_id === value} onSelect={() => { onChange(item.dataset_id); setOpen(false); }} />
                <button className="datasetSelectorDelete" type="button" onClick={(event) => void handleDelete(item.dataset_id, event)} aria-label={`${item.original_filename} 삭제`}>
                  <TrashIcon />
                </button>
              </div>
            ))}
            <div className="datasetSelectorDivider" />
            <button
              type="button"
              className="datasetSelectorItem datasetSelectorAdd"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
            >
              {uploading ? "업로드 중…" : "+ 파일 추가"}
            </button>
          </div>,
          document.body,
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,text/csv"
          className="visuallyHidden"
          onChange={(event) => {
            void handleFileSelected(event.target.files?.[0]);
            event.target.value = "";
          }}
        />
      </div>
      {error && <p className="errorMessage">{error}</p>}
    </div>
  );
}

function DatasetOption({
  item,
  active,
  onSelect,
}: {
  item: DatasetSummary;
  active: boolean;
  onSelect: () => void;
}) {
  const lotRange = item.lot_min && item.lot_max ? `${item.lot_min}~${item.lot_max}` : "LOT 정보 없음";
  const uploadedNote = item.uploaded_at ? new Intl.DateTimeFormat("ko-KR", { dateStyle: "medium", timeStyle: "short" }).format(new Date(item.uploaded_at)) : "내장";
  return (
    <button type="button" className={`datasetSelectorItem ${active ? "active" : ""}`} onClick={onSelect} role="option" aria-selected={active} style={{ flex: 1 }}>
      <strong className="datasetSelectorFilename" title={item.original_filename}>{item.original_filename}</strong>
      <small>{item.kind === "bundled" ? "내장" : "업로드"} · {item.row_count.toLocaleString()}행 · {lotRange} · {uploadedNote}</small>
      {item.warnings.length > 0 && <small style={{ color: "#b8720a" }}>⚠ {item.warnings[0]}</small>}
    </button>
  );
}

function ChevronIcon() {
  return (
    <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="m7 10 5 5 5-5" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2M6 7l1 13a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-13" />
    </svg>
  );
}
