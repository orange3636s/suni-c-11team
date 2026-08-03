"use client";

import dynamic from "next/dynamic";
import { useEffect, useRef } from "react";
import type { Data, Layout } from "plotly.js";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Props = {
  spec: { data?: Data[]; layout?: Partial<Layout> } | Record<string, unknown>;
  height?: number;
};

export default function PlotlyChart({ spec, height = 360 }: Props) {
  const data = (spec as { data?: Data[] }).data ?? [];
  const layout = (spec as { layout?: Partial<Layout> }).layout ?? {};
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = containerRef.current;
    if (!node || typeof ResizeObserver === "undefined") return;
    let timer: number | undefined;
    const observer = new ResizeObserver(() => {
      window.clearTimeout(timer);
      // react-plotly.js's `useResizeHandler` below only listens for the
      // window's own "resize" event -- a sidebar/AI panel collapsing or
      // expanding changes this chart's container width via a CSS grid
      // transition without the window itself resizing, so it would
      // otherwise never re-lay-out. Dispatching a synthetic resize event
      // reuses that existing wiring; it only calls Plotly's relayout, so
      // traces/shapes are untouched and zoom/selection survive.
      timer = window.setTimeout(() => window.dispatchEvent(new Event("resize")), 120);
    });
    observer.observe(node);
    return () => {
      window.clearTimeout(timer);
      observer.disconnect();
    };
  }, []);

  return (
    <div ref={containerRef} style={{ width: "100%" }}>
      <Plot
        data={data}
        layout={{
          ...layout,
          autosize: true,
          height,
          margin: { l: 50, r: 50, t: 50, b: 40, ...layout.margin },
          paper_bgcolor: "transparent",
          plot_bgcolor: "transparent",
        }}
        config={{ displaylogo: false, responsive: true }}
        style={{ width: "100%" }}
        useResizeHandler
      />
    </div>
  );
}
