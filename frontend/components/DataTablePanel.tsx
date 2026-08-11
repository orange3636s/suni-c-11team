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

// Shared search+sort state for the alarm/recommendation tables --
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

// .tableWrap (outer) only ever does border + radius + overflow:hidden --
// it never scrolls itself. .tableScroll (inner) is the one element that
// scrolls vertically. Splitting these into two elements, instead of one
// div wearing both classes, is deliberate: a single element can only have
// one computed `overflow`, so a later rule giving it `overflow-y:auto`
// for scrolling silently wins over an earlier `overflow:hidden` meant for
// corner-rounding -- the two goals were never compatible on one box.
export function ScrollTableBody({ children, rows = 10 }: { children: ReactNode; rows?: number }) {
  return (
    <div className="tableWrap">
      <div className="tableScroll" style={{ ["--scroll-rows" as string]: rows }}>
        {children}
        <div className="scrollTableFade" />
      </div>
    </div>
  );
}

// Horizontal-scroll variant of ScrollTableBody: same
// vertical-scroll/sticky-header shell, plus table-layout:auto (so no
// column ellipsis-truncates), a horizontally scrolling body once content
// exceeds `minWidth`, and a sticky first column so the row identity
// (Wafer) stays visible while scrolled. Shared by 알람 목록 and 개선 권장
// 목록 -- same column-truncation problem, same fix.
export function HScrollTableBody({
  children,
  rows = 10,
  minWidth = 900,
}: {
  children: ReactNode;
  rows?: number;
  minWidth?: number;
}) {
  return (
    <div className="tableWrap hScrollWrap">
      <div
        className="tableScroll hScroll"
        style={{ ["--scroll-rows" as string]: rows, ["--table-min-width" as string]: `${minWidth}px` }}
      >
        {children}
        <div className="scrollTableFade" />
      </div>
    </div>
  );
}

export function TableCaption({
  total,
  shown,
  totalUnit = "건",
  shownUnit = "건",
}: {
  total: number;
  shown: number;
  totalUnit?: string;
  shownUnit?: string;
}) {
  return (
    <p className="tableCaption">
      전체 {total}{totalUnit} 중 {shown}{shownUnit} 표시
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
