import type { ReactNode } from "react";
import { PageHeaderMeta } from "@/components/LastRunNote";

/** 여섯 화면(모니터링/Config별 트리맵/원인 분석/수율 예측/알림 기록/즐겨찾기)의
 * 제목 블록을 하나로 합친다. 이전에는 화면마다 `pageHeading`/`uploadIntro`
 * 클래스 조합이 갈려(원인 분석·알림 기록·즐겨찾기만 `uploadIntro`를 함께 써서
 * 24px, 나머지 셋은 26px) 제목 크기가 서로 달랐다 -- 이 컴포넌트는 그 레거시
 * 클래스에 기대지 않고 자체 CSS(`.pageHeader*`, globals.css 맨 끝)만 쓰므로
 * 어떤 부모(.rcPage 유무)에 놓여도 항상 같은 크기로 렌더된다. */
export function PageHeader({
  eyebrow,
  title,
  description,
  metaLabel,
  children,
}: {
  eyebrow?: string;
  title: string;
  description?: ReactNode;
  metaLabel?: string;
  children?: ReactNode;
}) {
  return (
    <div className="pageHeader">
      {eyebrow && <span className="pageHeaderEyebrow">{eyebrow}</span>}
      <h1 className="pageHeaderTitle">{title}</h1>
      {description && <p className="pageHeaderDescription">{description}</p>}
      <PageHeaderMeta label={metaLabel} />
      {children}
    </div>
  );
}

export default PageHeader;
