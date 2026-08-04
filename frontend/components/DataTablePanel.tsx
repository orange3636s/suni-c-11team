"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";

export type SortOption<T> = { value: string; label: string; compare: (a: T, b: T) => number };

function useDebouncedValue(value: string, delay: number): string {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), delay);
    return () => window.clearTimeout(timer);
  }, [value, delay]);
  return debounced;
}

// Shared search+sort state for the alarm/recommendation tables (spec §4) --
// both tables need identical client-side filtering, debounce, and tie-break
// behavior, so this is the one place that logic lives.
export function useTableSearchSort<T>(
  items: T[],
  searchText: (item: T) => string,
  sortOptions: SortOption<T>[],
  defaultSort: string,
  tieBreak: (a: T, b: T) => number,
) {
  const [search, setSearch] = useState("");
  const [sort, setSort] = useState(defaultSort);
  const debouncedSearch = useDebouncedValue(search, 200);

  const filtered = useMemo(() => {
    const q = debouncedSearch.trim().toLowerCase();
    if (!q) return items;
    return items.filter((item) => searchText(item).toLowerCase().includes(q));
  }, [items, debouncedSearch, searchText]);

  const sorted = useMemo(() => {
    const option = sortOptions.find((candidate) => candidate.value === sort) ?? sortOptions[0];
    const copy = [...filtered];
    copy.sort((a, b) => option.compare(a, b) || tieBreak(a, b));
    return copy;
  }, [filtered, sort, sortOptions, tieBreak]);

  return { search, setSearch, sort, setSort, filtered, sorted };
}

export function TableToolbar({
  search,
  onSearchChange,
  sort,
  onSortChange,
  sortOptions,
  placeholder,
  extra,
}: {
  search: string;
  onSearchChange: (value: string) => void;
  sort: string;
  onSortChange: (value: string) => void;
  sortOptions: Array<{ value: string; label: string }>;
  placeholder: string;
  extra?: ReactNode;
}) {
  return (
    <div className="tableToolbar">
      {extra}
      <div className="tableSearchBox">
        <svg className="tableSearchIcon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="11" cy="11" r="7" />
          <path d="m20 20-3.5-3.5" />
        </svg>
        <input
          type="text"
          value={search}
          onChange={(event) => onSearchChange(event.target.value)}
          placeholder={placeholder}
          aria-label="검색"
        />
        {search && (
          <button type="button" className="tableSearchClear" onClick={() => onSearchChange("")} aria-label="검색어 지우기">
            ×
          </button>
        )}
      </div>
      <select className="tableSortSelect" value={sort} onChange={(event) => onSortChange(event.target.value)} aria-label="정렬">
        {sortOptions.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </div>
  );
}

export function ScrollTableBody({ children }: { children: ReactNode }) {
  return (
    <div className="tableWrap scrollTableWrap">
      {children}
      <div className="scrollTableFade" />
    </div>
  );
}

export function TableCaption({ total, shown }: { total: number; shown: number }) {
  return (
    <p className="tableCaption">
      전체 {total}건 중 {shown}건 표시
    </p>
  );
}

export function NoSearchResults({ onClear }: { onClear: () => void }) {
  return (
    <div className="emptyMessage tableEmptySearch">
      <span>검색 결과가 없습니다</span>
      <button type="button" onClick={onClear}>검색어 지우기</button>
    </div>
  );
}
