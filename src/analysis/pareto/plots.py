"""Plot-spec builders for Pareto factors.

Returns Plotly figure specs as plain dicts (``fig.to_dict()``) so the API
layer can serialize them to JSON and the Next.js frontend renders them
client-side with react-plotly.js. No image rendering happens server-side.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from src.analysis.pareto.selector import ParetoFactor, TargetParetoResult

LOW_SAMPLE_THRESHOLD = 20


def _quantile_profile_with_ci(x: pd.Series, y: pd.Series, bins: int = 8) -> pd.DataFrame:
    try:
        q = pd.qcut(x, bins, duplicates="drop")
    except ValueError:
        return pd.DataFrame(columns=["x_mean", "y_mean", "y_lo", "y_hi", "n"])
    frame = pd.DataFrame({"x": x, "y": y, "q": q})
    rows = []
    for _, group in frame.groupby("q", observed=True):
        n = len(group)
        y_mean = group["y"].mean()
        y_sem = group["y"].std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
        ci = 1.96 * y_sem
        rows.append(
            {
                "x_mean": group["x"].mean(),
                "y_mean": y_mean,
                "y_lo": y_mean - ci,
                "y_hi": y_mean + ci,
                "n": n,
            }
        )
    return pd.DataFrame(rows).sort_values("x_mean").reset_index(drop=True)


def build_scatter_plot(df: pd.DataFrame, factor: ParetoFactor) -> dict:
    """Continuous factor (R/D): scatter + quantile-mean overlay + shape annotation."""
    frame = pd.DataFrame({"x": df[factor.feature], "y": df[factor.target]}).dropna()
    profile = _quantile_profile_with_ci(frame["x"], frame["y"])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame["x"],
            y=frame["y"],
            mode="markers",
            marker=dict(opacity=0.35, size=6),
            name="관측값",
        )
    )
    if not profile.empty:
        fig.add_trace(
            go.Scatter(
                x=profile["x_mean"],
                y=profile["y_mean"],
                mode="lines+markers",
                line=dict(width=3),
                name="분위구간 평균",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=pd.concat([profile["x_mean"], profile["x_mean"][::-1]]),
                y=pd.concat([profile["y_hi"], profile["y_lo"][::-1]]),
                fill="toself",
                fillcolor="rgba(99,110,250,0.15)",
                line=dict(width=0),
                name="95% CI",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    if factor.relation_shape == "u_shape" and factor.optimal_center is not None:
        fig.add_vline(
            x=factor.optimal_center,
            line=dict(dash="dash", color="crimson"),
            annotation_text=f"최적 중심 {factor.optimal_center:.2f}",
        )
    elif factor.relation_shape in ("monotonic_increasing", "monotonic_decreasing"):
        slope, intercept, r_value, _, _ = stats.linregress(frame["x"], frame["y"])
        x_line = np.linspace(frame["x"].min(), frame["x"].max(), 50)
        fig.add_trace(
            go.Scatter(
                x=x_line,
                y=slope * x_line + intercept,
                mode="lines",
                line=dict(dash="dash", color="crimson"),
                name=f"회귀선 (R²={r_value ** 2:.3f})",
            )
        )

    fig.update_layout(
        title=(
            f"{factor.target} vs {factor.feature} "
            f"(n={factor.n_observed}, ε²={factor.eps2:.3f}, q={factor.q_value:.3g})"
        ),
        xaxis_title=factor.feature,
        yaxis_title=factor.target,
        template="plotly_white",
    )
    return fig.to_dict()


def build_config_boxplot(df: pd.DataFrame, factor: ParetoFactor) -> dict:
    """Categorical factor (Config): boxplot, sorted by category mean descending."""
    frame = pd.DataFrame({"x": df[factor.feature], "y": df[factor.target]}).dropna()
    counts = frame["x"].value_counts()
    means = frame.groupby("x", observed=True)["y"].mean().sort_values(ascending=False)

    fig = go.Figure()
    for category in means.index:
        group = frame.loc[frame["x"] == category, "y"]
        n = counts[category]
        low_sample = n < LOW_SAMPLE_THRESHOLD
        fig.add_trace(
            go.Box(
                y=group,
                name=f"{category} (n={n})" + (" [표본 부족]" if low_sample else ""),
                marker=dict(color="lightgray" if low_sample else None),
                boxpoints="outliers",
            )
        )

    fig.update_layout(
        title=(
            f"{factor.target} vs {factor.feature} "
            f"(n={factor.n_observed}, ε²={factor.eps2:.3f}, q={factor.q_value:.3g})"
        ),
        yaxis_title=factor.target,
        template="plotly_white",
        showlegend=False,
    )
    return fig.to_dict()


def build_factor_plot(df: pd.DataFrame, factor: ParetoFactor) -> dict:
    if factor.kind == "Config":
        return build_config_boxplot(df, factor)
    return build_scatter_plot(df, factor)


def build_pareto_chart(result: TargetParetoResult) -> dict:
    """Bar (individual eps2 contribution) + line (cumulative %) dual-axis chart."""
    factors = result.factors
    fig = go.Figure()
    if factors:
        fig.add_trace(
            go.Bar(
                x=[f.feature for f in factors],
                y=[f.contribution_pct for f in factors],
                name="개별 기여율(%)",
                yaxis="y1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[f.feature for f in factors],
                y=[f.cumulative_pct for f in factors],
                mode="lines+markers",
                name="누적 기여율(%)",
                yaxis="y2",
            )
        )
        fig.add_hline(y=80, line=dict(dash="dash", color="gray"), yref="y2")

    fig.update_layout(
        title=f"{result.target} Pareto 차트",
        xaxis_title="인자",
        yaxis=dict(title="개별 기여율(%)"),
        yaxis2=dict(title="누적 기여율(%)", overlaying="y", side="right", range=[0, 105]),
        template="plotly_white",
    )
    return fig.to_dict()
