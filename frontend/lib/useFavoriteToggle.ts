"use client";

import { useEffect, useRef, useState } from "react";
import { createFavorite, deleteFavorite, getFavorites } from "@/lib/api";
import type { FavoriteSnapshot } from "@/types/data";

type FavoriteKeyInput = Pick<FavoriteSnapshot, "dataset" | "target" | "feature" | "viewType">;

/** 즐겨찾기 항목을 구분하는 유일한 키. viewType별로 별도 즐겨찾기이므로
 * 항상 포함한다("Scatter로 저장한 인자"와 "Box로 저장한 같은 인자"가
 * 같은 키로 잡혀 서로를 지우면 안 된다). */
function favoriteKeyOf(s: FavoriteKeyInput): string {
  return `${s.dataset}::${s.target}::${s.feature}::${s.viewType}`;
}

/** 즐겨찾기 ☆ 토글 공용 로직 -- 산점도·박스플롯이 쓴다. 생성/삭제·중복
 * 방지(pending 가드) 로직을 여기 하나로 모은다. */
export function useFavoriteToggle() {
  const [favoriteIdByKey, setFavoriteIdByKey] = useState<Record<string, string>>({});
  // 생성/삭제 요청이 아직 끝나지 않은 키는 다시 받지 않는다 -- 빠른
  // 더블클릭 시 두 호출 모두 같은(아직 갱신 전) favoriteIdByKey를 보고
  // 둘 다 "생성" 경로로 들어가 중복 레코드가 생기는 것을 막는다. ref인
  // 이유: 같은 렌더에서 연달아 호출되면 useState는 아직 반영 전(stale)이라
  // 막지 못한다.
  const pendingKeysRef = useRef<Set<string>>(new Set());
  const [pendingKeys, setPendingKeys] = useState<Set<string>>(new Set());

  useEffect(() => {
    let cancelled = false;
    getFavorites()
      .then((response) => {
        if (cancelled) return;
        const map: Record<string, string> = {};
        for (const item of response.items) {
          map[favoriteKeyOf(item.snapshot)] = item.favorite_id;
        }
        setFavoriteIdByKey(map);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  async function toggleFavorite(snapshot: FavoriteSnapshot) {
    const key = favoriteKeyOf(snapshot);
    if (pendingKeysRef.current.has(key)) return;
    pendingKeysRef.current.add(key);
    setPendingKeys(new Set(pendingKeysRef.current));
    try {
      const existingId = favoriteIdByKey[key];
      if (existingId) {
        setFavoriteIdByKey((previous) => {
          const next = { ...previous };
          delete next[key];
          return next;
        });
        try {
          await deleteFavorite(existingId);
        } catch {
          // Best-effort -- a failed unfavorite just leaves the star filled;
          // the user can retry.
          setFavoriteIdByKey((previous) => ({ ...previous, [key]: existingId }));
        }
        return;
      }
      try {
        const created = await createFavorite(snapshot);
        setFavoriteIdByKey((previous) => ({ ...previous, [key]: created.favorite_id }));
      } catch {
        // Best-effort -- 저장 실패 시 별은 그대로 빈 채로 남는다.
      }
    } finally {
      pendingKeysRef.current.delete(key);
      setPendingKeys(new Set(pendingKeysRef.current));
    }
  }

  function isFavorited(snapshot: FavoriteKeyInput): boolean {
    return Boolean(favoriteIdByKey[favoriteKeyOf(snapshot)]);
  }
  function isFavoritePending(snapshot: FavoriteKeyInput): boolean {
    return pendingKeys.has(favoriteKeyOf(snapshot));
  }

  return { isFavorited, isFavoritePending, toggleFavorite };
}
