"use client";

import Link from "next/link";
import { createPortal } from "react-dom";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { getModels } from "@/lib/api";
import type { ModelSummary } from "@/types/data";

const LAST_MODEL_KEY = "semiconductor-ai:last-model-id";

type ModelSelectorProps = {
  value: string;
  onValueChange: (modelId: string) => void;
  onModelsChange?: (models: ModelSummary[], warnings: string[]) => void;
  disabled?: boolean;
  ariaLabel?: string;
};

function statusLabel(model: ModelSummary): string {
  if (model.compatibility === "incompatible") return "호환되지 않음";
  if (model.compatibility === "legacy") return "이전 모델";
  if (model.compatibility === "unknown_schema") return "스키마 확인 필요";
  return model.model_type === "hybrid_multi_y" ? "Hybrid Multi-Y" : "호환 가능";
}

function metric(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "-";
}

function createdAtLabel(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("ko-KR");
}

function chooseInitialModel(models: ModelSummary[]): string {
  const queryModel = new URLSearchParams(window.location.search).get("model_id");
  const storedModel = window.sessionStorage.getItem(LAST_MODEL_KEY) ?? window.localStorage.getItem(LAST_MODEL_KEY);
  for (const candidate of [queryModel, storedModel]) {
    if (candidate && models.some((model) => model.model_id === candidate && model.compatibility !== "incompatible")) return candidate;
  }
  return models.find((model) => model.model_type === "hybrid_multi_y" && model.compatibility === "compatible")?.model_id
    ?? models.find((model) => model.target === "Y" && model.compatibility === "compatible")?.model_id
    ?? "";
}

export default function ModelSelector({ value, onValueChange, onModelsChange, disabled = false, ariaLabel = "저장 모델 선택" }: ModelSelectorProps) {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);
  const [popupStyle, setPopupStyle] = useState<React.CSSProperties>({});
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const loadModels = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const response = await getModels();
      const sorted = [...response.models].sort((left, right) => right.created_at.localeCompare(left.created_at));
      setModels(sorted);
      onModelsChange?.(sorted, response.warnings);
      if (!value) {
        const initialModel = chooseInitialModel(sorted);
        if (initialModel) onValueChange(initialModel);
      }
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "모델 목록을 불러오지 못했습니다.");
      onModelsChange?.([], []);
    } finally {
      setLoading(false);
    }
  }, [onModelsChange, onValueChange, value]);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadModels(), 0);
    return () => window.clearTimeout(timer);
    // Initial model discovery must not rerun when a parent callback identity changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const filtered = useMemo(() => {
    const keyword = search.trim().toLocaleLowerCase("ko");
    return models.filter((model) => !keyword || `${model.model_name} ${model.model_id} ${model.model_type ?? ""} ${model.created_at} ${statusLabel(model)}`.toLocaleLowerCase("ko").includes(keyword));
  }, [models, search]);
  const selected = models.find((model) => model.model_id === value);
  const missingSelectedModel = Boolean(value && !selected && !loading);

  const positionPopup = useCallback(() => {
    const rect = triggerRef.current?.getBoundingClientRect();
    if (!rect) return;
    const width = Math.min(Math.max(rect.width, 420), window.innerWidth - 24);
    setPopupStyle({ left: Math.max(12, Math.min(rect.left, window.innerWidth - width - 12)), top: rect.bottom + 6, width });
  }, []);

  useEffect(() => {
    if (!open) return;
    positionPopup();
    const close = (event: MouseEvent) => {
      const target = event.target as Node;
      if (!rootRef.current?.contains(target) && !(target as Element).closest?.(".modelSelectorPopup")) setOpen(false);
    };
    window.addEventListener("resize", positionPopup);
    window.addEventListener("scroll", positionPopup, true);
    document.addEventListener("mousedown", close);
    window.setTimeout(() => searchRef.current?.focus(), 0);
    return () => { window.removeEventListener("resize", positionPopup); window.removeEventListener("scroll", positionPopup, true); document.removeEventListener("mousedown", close); };
  }, [open, positionPopup]);

  function selectModel(modelId: string) {
    const model = models.find((item) => item.model_id === modelId);
    if (!model || model.compatibility === "incompatible") return;
    onValueChange(modelId);
    window.sessionStorage.setItem(LAST_MODEL_KEY, modelId);
    window.localStorage.setItem(LAST_MODEL_KEY, modelId);
    const url = new URL(window.location.href); url.searchParams.set("model_id", modelId); window.history.replaceState({}, "", url);
    setOpen(false); setSearch(""); triggerRef.current?.focus();
  }

  function handleKeyDown(event: React.KeyboardEvent) {
    if (event.key === "Escape") { setOpen(false); triggerRef.current?.focus(); return; }
    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
      event.preventDefault();
      const direction = event.key === "ArrowDown" ? 1 : -1;
      setActiveIndex((current) => Math.max(0, Math.min(filtered.length - 1, current + direction)));
    }
    if (event.key === "Enter" && filtered[activeIndex]) { event.preventDefault(); selectModel(filtered[activeIndex].model_id); }
  }

  const popup = open && typeof document !== "undefined" ? createPortal(
    <div className="modelSelectorPopup" style={popupStyle} role="dialog" aria-label={`${ariaLabel} 목록`} onKeyDown={handleKeyDown}>
      <div className="modelSelectorPopupTools">
        <input ref={searchRef} type="search" value={search} placeholder="모델명, ID, 유형, 생성일 검색" onChange={(event) => { setSearch(event.target.value); setActiveIndex(0); }} />
        <button type="button" className="button secondary compact" onClick={() => void loadModels()} disabled={loading}>{loading ? "조회 중" : "새로고침"}</button>
      </div>
      {error ? <div className="modelSelectorState error" role="alert">{error}</div> : !filtered.length ? (
        <div className="modelSelectorState"><p>{models.length ? "검색 결과가 없습니다." : "학습 모델이 없습니다."}</p>{!models.length && <Link href="/training">모델 학습으로 이동</Link>}</div>
      ) : (
        <div className="modelSelectorList" role="listbox">
          {filtered.map((model, index) => {
            const unavailable = model.compatibility === "incompatible";
            return <button type="button" role="option" aria-selected={model.model_id === value} className={`modelSelectorOption ${model.model_id === value || index === activeIndex ? "active" : ""}`} key={model.model_id} disabled={unavailable} onMouseEnter={() => setActiveIndex(index)} onClick={() => selectModel(model.model_id)}>
              <span className="modelSelectorOptionTop">
                <strong title={model.model_name}>{model.model_name}</strong>
                <span className={`modelCompatibility ${model.compatibility}`}>{statusLabel(model)}</span>
              </span>
              <code title={model.model_id}>{model.model_id}</code>
              <small>{model.model_type ?? "Single Model"} · R² {metric(model.test_metrics.r2)} · RMSE {metric(model.test_metrics.rmse)} · {createdAtLabel(model.created_at)}</small>
            </button>;
          })}
        </div>
      )}
    </div>, document.body) : null;

  return <div className="modelSelector" ref={rootRef}>
    <button ref={triggerRef} type="button" className={`modelSelectorTrigger ${selected || missingSelectedModel ? "selected" : ""}`} aria-label={ariaLabel} aria-haspopup="dialog" aria-expanded={open} disabled={disabled || loading} onClick={() => setOpen((current) => !current)}>
      <span>
        <strong title={selected?.model_name ?? (missingSelectedModel ? value : undefined)}>{selected?.model_name ?? (loading ? "모델 불러오는 중" : missingSelectedModel ? "삭제되었거나 목록에 없는 모델" : "모델을 선택하세요")}</strong>
        {(selected || missingSelectedModel) && <small title={selected?.model_id ?? value}>{selected?.model_id ?? value}</small>}
      </span>
      <span aria-hidden="true">⌄</span>
    </button>
    {popup}
  </div>;
}
