"use client";

import dynamic from "next/dynamic";
import type { Data, Layout } from "plotly.js";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

type Props = {
  spec: { data?: Data[]; layout?: Partial<Layout> } | Record<string, unknown>;
  height?: number;
};

export default function PlotlyChart({ spec, height = 360 }: Props) {
  const data = (spec as { data?: Data[] }).data ?? [];
  const layout = (spec as { layout?: Partial<Layout> }).layout ?? {};
  return (
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
  );
}
