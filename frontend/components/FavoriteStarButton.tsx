"use client";

/** 즐겨찾기 별 토글 -- 원인 분석(산점도·박스플롯·파레토·히트맵)과 Config별
 * 트리맵이 모두 같은 순서·크기·스타일로 쓴다(제목 옆, 이미지 저장 버튼
 * 앞). 저장 시점 상태 스냅샷만 넘긴다, 점 데이터는 절대 포함하지 않는다. */
export function FavoriteStarButton({
  favorited,
  disabled,
  onClick,
}: {
  favorited: boolean;
  // 생성/삭제 요청이 진행 중인 동안 버튼을 막는다 -- 빠른 더블클릭이
  // 중복 즐겨찾기(좀비 레코드)를 만드는 걸 막는 시각적 짝.
  disabled?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className={`favoriteStarButton ${favorited ? "active" : ""}`}
      onClick={onClick}
      disabled={disabled}
      aria-pressed={favorited}
      aria-label={favorited ? "즐겨찾기 해제" : "즐겨찾기 추가"}
      title={favorited ? "즐겨찾기 해제" : "즐겨찾기 추가"}
    >
      {favorited ? "★" : "☆"}
    </button>
  );
}

export default FavoriteStarButton;
