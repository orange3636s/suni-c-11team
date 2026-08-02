"use client";

import Link from "next/link";
import { createPortal } from "react-dom";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { getModels } from "@/lib/api";
import type { ModelSummary } from "@/types/data";

const LAST_MODEL_KEY = "semiconductor-ai:last-model-id";
const UNAVAILABLE_COMPATIBILITY_STATUSES = new Set([
  "dependency_missing",
  "model_file_missing",
  "invalid_metadata",
  "invalid_model",
  "load_error",
  "schema_incompatible",
  "incompatible",
]);

export type ModelSelectionReason = "user" | "reconcile";

type ModelSelectorProps = {
  value: string;
  onValueChange: (modelId: string, reason?: ModelSelectionReason) => void;
  onModelsChange?: (models: ModelSummary[], warnings: string[]) => void;
  unavailableModelIds?: string[];
  disabled?: boolean;
  ariaLabel?: string;
};

export function isModelUsable(model: ModelSummary | null | undefined): model is ModelSummary {
  return Boolean(
    model &&
    model.available !== false &&
    model.loadable !== false &&
    model.compatibility !== "incompatible" &&
    !UNAVAILABLE_COMPATIBILITY_STATUSES.has(model.compatibility_status),
  );
}

function statusLabel(model: ModelSummary): string {
  if (!isModelUsable(model)) return "사용 불가";
  if (model.compatibility === "legacy") return "이전 모델";
  if (model.compatibility === "unknown_schema") return "스키마 확인 필요";
  return model.model_type === "hybrid_multi_y" ? "Hybrid Multi-Y" : "호환 가능";
}

function statusClass(model: ModelSummary): string {
  return isModelUsable(model) ? model.compatibility : "unavailable";
}

function isSelectable(model: ModelSummary | null | undefined, unavailableIds: Set<string>): boolean {
  return isModelUsable(model) && !unavailableIds.has(model.model_id);
}

function metric(value: unknown): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(4) : "-";
}

function createdAtLabel(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString("ko-KR");
}

function removeStalePersistedSelections(models: ModelSummary[], unavailableIds: Set<string>): void {
  const usableIds = new Set(models.filter((model) => isSelectable(model, unavailableIds)).map((model) => model.model_id));
  const url = new URL(window.location.href);
  const queryModel = url.searchParams.get("model_id");
  if (queryModel && !usableIds.has(queryModel)) {
    url.searchParams.delete("model_id");
    window.history.replaceState({}, "", url);
  }
  for (const storage of [window.sessionStorage, window.localStorage]) {
    const storedModel = storage.getItem(LAST_MODEL_KEY);
    if (storedModel && !usableIds.has(storedModel)) storage.removeItem(LAST_MODEL_KEY);
  }
}

function persistedCandidate(models: ModelSummary[], unavailableIds: Set<string>): string {
  const candidates = [
    new URLSearchParams(window.location.search).get("model_id"),
    window.sessionStorage.getItem(LAST_MODEL_KEY),
    window.localStorage.getItem(LAST_MODEL_KEY),
  ];
  return candidates.find((candidate) =>
    Boolean(candidate && isSelectable(models.find((model) => model.model_id === candidate), unavailableIds))) ?? "";
}

function persistSelection(modelId: string): void {
  const url = new URL(window.location.href);
  if (modelId) {
    window.sessionStorage.setItem(LAST_MODEL_KEY, modelId);
    window.localStorage.setItem(LAST_MODEL_KEY, modelId);
    if (url.searchParams.get("model_id") !== modelId) {
      url.searchParams.set("model_id", modelId);
      window.history.replaceState({}, "", url);
    }
    return;
  }
  window.sessionStorage.removeItem(LAST_MODEL_KEY);
  window.localStorage.removeItem(LAST_MODEL_KEY);
  if (url.searchParams.has("model_id")) {
    url.searchParams.delete("model_id");
    window.history.replaceState({}, "", url);
  }
}

export default function ModelSelector({ value, onValueChange, onModelsChange, unavailableModelIds = [], disabled = false, ariaLabel = "저장 모델 선택" }: ModelSelectorProps) {
  const [models, setModels] = useState<ModelSummary[]>([]);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [loaded, setLoaded] = useState(false);
  const [error, setError] = useState("");
  const [open, setOpen] = useState(false);
  const [activeIndex, setActiveIndex] = useState(0);
  const [popupStyle, setPopupStyle] = useState<React.CSSProperties>({});
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const onValueChangeRef = useRef(onValueChange);
  const onModelsChangeRef = useRef(onModelsChange);
  const unavailableIds = useMemo(() => new Set(unavailableModelIds), [unavailableModelIds]);

  useEffect(() => { onValueChangeRef.current = onValueChange; }, [onValueChange]);
  useEffect(() => { onModelsChangeRef.current = onModelsChange; }, [onModelsChange]);

  const loadModels = useCallback(async () => {
    setLoading(true);
    setLoaded(false);
    setError("");
    try {
      const response = await getModels();
      const currentModels = [...response.models];
      setModels(currentModels);
      setLoaded(true);
      onModelsChangeRef.current?.(currentModels, response.warnings);
    } catch (requestError) {
      setModels([]);
      setError(requestError instanceof Error ? requestError.message : "모델 목록을 불러오지 못했습니다.");
      onModelsChangeRef.current?.([], []);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void loadModels(), 0);
    return () => window.clearTimeout(timer);
  }, [loadModels]);

  useEffect(() => {
    if (!loaded) return;
    const current = models.find((model) => model.model_id === value);
    const usableModels = models.filter((model) => isSelectable(model, unavailableIds));
    removeStalePersistedSelections(models, unavailableIds);
    const nextValue = current && isSelectable(current, unavailableIds)
      ? current.model_id
      : persistedCandidate(models, unavailableIds) || usableModels[0]?.model_id || "";
    persistSelection(nextValue);
    if (nextValue !== value) onValueChangeRef.current(nextValue, "reconcile");
  }, [loaded, models, unavailableIds, value]);

  const filtered = useMemo(() => {
    const keyword = search.trim().toLocaleLowerCase("ko");
    return models.filter((model) => !keyword || `${model.model_name} ${model.model_id} ${model.model_type ?? ""} ${model.created_at} ${statusLabel(model)} ${model.incompatibility_reason ?? ""}`.toLocaleLowerCase("ko").includes(keyword));
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
    if (!model || !isSelectable(model, unavailableIds)) return;
    persistSelection(modelId);
    onValueChange(modelId, "user");
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
            const unavailable = !isSelectable(model, unavailableIds);
            return <button type="button" role="option" aria-selected={model.model_id === value} aria-disabled={unavailable} className={`modelSelectorOption ${model.model_id === value || index === activeIndex ? "active" : ""}`} key={model.model_id} disabled={unavailable} title={model.incompatibility_reason ?? undefined} onMouseEnter={() => setActiveIndex(index)} onClick={() => selectModel(model.model_id)}>
              <span className="modelSelectorOptionTop">
                <strong title={model.model_name}>{model.model_name}</strong>
                <span className={`modelCompatibility ${unavailable ? "unavailable" : statusClass(model)}`}>{unavailable ? "사용 불가" : statusLabel(model)}</span>
              </span>
              <code title={model.model_id}>{model.model_id}</code>
              <small>{model.model_type ?? "Single Model"} · R² {metric(model.test_metrics.r2)} · RMSE {metric(model.test_metrics.rmse)} · {createdAtLabel(model.created_at)}</small>
              {unavailable && <small className="modelUnavailableReason">{model.incompatibility_reason ?? (unavailableIds.has(model.model_id) ? "현재 서버의 상세 검증에서 사용할 수 없는 모델로 확인되었습니다." : "현재 서버에서 이 모델을 사용할 수 없습니다.")}</small>}
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
