# -*- coding: utf-8 -*-
"""
experiment_app.py — 环境微特征电力现货预测增强试验舱（Streamlit Web 界面）

启动方式
--------
    pip install streamlit plotly
    streamlit run experiment_app.py

说明
----
- 默认 Demo 模拟数据，无需 CSV 即可完整体验。
- 实验核心逻辑复用 ``run_experiment.py``，本文件仅负责交互与可视化。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# 保证可导入同目录下的实验脚本
_APP_ROOT = Path(__file__).resolve().parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

from run_experiment import (  # noqa: E402
    BENCHMARK_FEATURE_COLUMNS,
    COL_TIMESTAMP,
    ENHANCED_ENV_COLUMNS,
    ExperimentConfig,
    ExperimentReport,
    _pct_alpha,
    _pct_improvement,
    add_model_ready_features,
    backtest_storage_revenue_yuan,
    build_enhanced_wide_table,
    build_synthetic_raw_tables,
    evaluate_regression,
    extract_env_feature_importance,
    load_raw_tables,
    temporal_train_test_split,
    train_quantile_lgbm,
    _import_optimizer,
    _sanitize_xy,
)
from run_experiment import COL_TARGET  # noqa: E402
from data_pipeline import EnvironmentalDataPipeline, PipelineConfig  # noqa: E402

# 工业级默认回溯窗口：30 天 × 24h，保证 LightGBM 有足够样本学习环境长尾微特征
PIPELINE_LOOKBACK_HOURS: int = 720

# ---------------------------------------------------------------------------
# 页面基础配置
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="环境微特征 · 电价预测增强试验舱",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 自定义样式：科技感 Banner、蓝色主按钮、指标高光卡片
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .main-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 45%, #0ea5e9 100%);
        padding: 1.6rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 32px rgba(14, 165, 233, 0.25);
    }
    .main-banner h1 {
        color: #f8fafc;
        font-size: 1.55rem;
        font-weight: 700;
        margin: 0 0 0.35rem 0;
        letter-spacing: 0.02em;
    }
    .main-banner p {
        color: #bae6fd;
        margin: 0;
        font-size: 0.95rem;
    }
    div.stButton > button[kind="primary"] {
        background: linear-gradient(90deg, #2563eb, #0ea5e9) !important;
        color: white !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.65rem 1.5rem !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.45) !important;
    }
    div.stButton > button[kind="primary"]:hover {
        filter: brightness(1.08);
    }
    .alpha-highlight {
        font-size: 1.35rem;
        font-weight: 800;
        color: #16a34a;
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .label { color: #64748b; font-size: 0.85rem; }
    .metric-card .value { color: #0f172a; font-size: 1.75rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_banner() -> None:
    """顶部专业 Banner。"""
    st.markdown(
        """
        <div class="main-banner">
            <h1>量化前沿：环境微特征（辐射/污染/水温）电力电价预测增强试验舱</h1>
            <p>Quant Frontier Lab · ERA5 容量加权辐射 × 污染负荷交叉 × 储能套利经济价值实证</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_config_from_sidebar() -> tuple[ExperimentConfig, bool]:
    """
    从侧边栏读取用户配置。

    Returns
    -------
    (config, run_clicked) : 实验配置与是否点击了运行按钮
    """
    st.sidebar.header("试验控制台")
    data_mode = st.sidebar.radio(
        "数据模式",
        options=["Demo 模拟数据", "智能数据中台 (SQLite)", "真实数据目录"],
        index=0,
        help="Demo：内置 Mock；数据中台：读取 data_lake 历史库；真实目录：本地 CSV。",
    )
    pipeline_refresh = st.sidebar.checkbox(
        f"运行前先执行日更（回溯 {PIPELINE_LOOKBACK_HOURS}h / 30天）",
        value=True,
        disabled=(data_mode != "智能数据中台 (SQLite)"),
    )
    if data_mode == "智能数据中台 (SQLite)":
        st.sidebar.info(
            "💡 提示：系统当前已自动触发 720 小时（30天）工业级时序回溯，"
            "确保环境长尾微特征拥有充足的训练样本。"
        )
    train_ratio = st.sidebar.slider(
        "时序划分比例（训练集占比）",
        min_value=0.60,
        max_value=0.90,
        value=0.80,
        step=0.05,
        help="严格按时间顺序切分，防止未来信息泄露。",
    )
    data_dir_str = st.sidebar.text_input(
        "真实数据目录路径",
        value=str(_APP_ROOT / "data"),
        disabled=(data_mode != "真实数据目录"),
    )
    st.sidebar.markdown("---")
    st.sidebar.caption("储能回测参数（与 run_experiment 默认一致）")
    power_mw = st.sidebar.number_input("额定功率 (MW)", value=50.0, min_value=1.0)
    capacity_mwh = st.sidebar.number_input("容量 (MWh)", value=200.0, min_value=1.0)

    run_clicked = st.sidebar.button(
        "⚡ 启动双模型打擂台（Run Experiment）",
        type="primary",
        use_container_width=True,
    )

    demo = data_mode == "Demo 模拟数据"
    use_pipeline = data_mode == "智能数据中台 (SQLite)"
    data_dir = Path(data_dir_str) if data_mode == "真实数据目录" else None
    config = ExperimentConfig(
        demo=demo,
        data_dir=data_dir,
        train_ratio=train_ratio,
        storage_power_mw=power_mw,
        storage_capacity_mwh=capacity_mwh,
    )
    return config, run_clicked, use_pipeline, pipeline_refresh


def load_tables_from_pipeline(
    run_update: bool,
    lookback_hours: int = PIPELINE_LOOKBACK_HOURS,
) -> dict:
    """
    从智能数据中台加载原始六表；可选先执行日更。

    Web 试验舱默认 ``lookback_hours=720``（30 天），与 CLI
    ``python data_pipeline.py --lookback-hours 720`` 保持一致。
    """
    pipeline = EnvironmentalDataPipeline(
        PipelineConfig(lookback_hours=lookback_hours)
    )
    if run_update:
        pipeline.run_daily_update()
    else:
        raw = pipeline.load_raw_tables_from_db()
        if any(df.empty for df in raw.values()):
            st.warning(
                f"本地库为空或不全，自动触发 {lookback_hours} 小时回溯日更..."
            )
            pipeline.run_daily_update()
    return pipeline.load_raw_tables_from_db()


def run_experiment_with_status(
    config: ExperimentConfig,
    pipeline_tables: Optional[dict] = None,
) -> ExperimentReport:
    """
    分步骤执行实验 pipeline，配合 ``st.status`` 展示进度动效。

    逻辑与 ``run_experiment.run_experiment`` 一致，但拆步便于前端反馈。
    """
    benchmark_cols = list(BENCHMARK_FEATURE_COLUMNS)
    enhanced_cols = benchmark_cols + list(ENHANCED_ENV_COLUMNS)
    required_cols = [COL_TARGET] + enhanced_cols

    with st.status("实验引擎运行中...", expanded=True) as status:
        # --- 步骤 1：特征中台 ---
        st.write("正在调用特征中台对齐空间辐射数据...")
        if pipeline_tables is not None:
            tables = pipeline_tables
        elif config.demo:
            tables = build_synthetic_raw_tables()
        else:
            if config.data_dir is None or not config.data_dir.is_dir():
                raise FileNotFoundError(
                    f"真实数据目录不存在: {config.data_dir}。"
                    "请创建 data/ 并放入 market.csv 等文件，或切换 Demo / 数据中台模式。"
                )
            tables = load_raw_tables(config.data_dir)

        df_wide = build_enhanced_wide_table(tables, schema=config.schema)
        df_wide = add_model_ready_features(df_wide)

        missing = [c for c in required_cols if c not in df_wide.columns]
        if missing:
            raise ValueError(f"宽表缺少列: {missing}")

        feature_frame = df_wide[required_cols + [COL_TIMESTAMP]].copy()
        feature_frame = feature_frame.dropna(subset=required_cols).reset_index(drop=True)
        train_df, test_df = temporal_train_test_split(feature_frame, config.train_ratio)

        # --- 步骤 2：基准模型 ---
        st.write("正在训练基准 LightGBM...")
        X_train_b, y_train = _sanitize_xy(train_df[benchmark_cols], train_df[COL_TARGET])
        X_train_e, _ = _sanitize_xy(train_df[enhanced_cols], train_df[COL_TARGET])
        X_test_b, y_test = _sanitize_xy(test_df[benchmark_cols], test_df[COL_TARGET])
        X_test_e, _ = _sanitize_xy(test_df[enhanced_cols], test_df[COL_TARGET])

        valid_size = max(1, int(len(X_train_e) * 0.1))
        X_tr_b = X_train_b.iloc[:-valid_size]
        y_tr_b = y_train.iloc[:-valid_size]
        X_va_b = X_train_b.iloc[-valid_size:]
        y_va_b = y_train.iloc[-valid_size:]

        model_bench = train_quantile_lgbm(X_tr_b, y_tr_b, X_va_b, y_va_b)
        pred_bench = model_bench.predict(X_test_b)

        # --- 步骤 3：增强模型 ---
        st.write("正在引入环境微特征训练增强模型...")
        X_tr_e = X_train_e.iloc[:-valid_size]
        y_tr_e = y_train.iloc[:-valid_size]
        X_va_e = X_train_e.iloc[-valid_size:]
        y_va_e = y_train.iloc[-valid_size:]

        model_enh = train_quantile_lgbm(X_tr_e, y_tr_e, X_va_e, y_va_e)
        pred_enh = model_enh.predict(X_test_e)

        metrics_bench = evaluate_regression(y_test.values, pred_bench)
        metrics_enh = evaluate_regression(y_test.values, pred_enh)

        # --- 步骤 4：储能套利 ---
        st.write("正在通过 SciPy 引擎进行储能套利回测...")
        backend, optimize_fn, hourly_pnl_fn = _import_optimizer()
        settle = y_test.values.astype(float)

        rev_bench = backtest_storage_revenue_yuan(
            pred_bench,
            settle,
            backend,
            optimize_fn,
            hourly_pnl_fn,
            power_mw=config.storage_power_mw,
            capacity_mwh=config.storage_capacity_mwh,
        )
        rev_enh = backtest_storage_revenue_yuan(
            pred_enh,
            settle,
            backend,
            optimize_fn,
            hourly_pnl_fn,
            power_mw=config.storage_power_mw,
            capacity_mwh=config.storage_capacity_mwh,
        )

        env_imp = extract_env_feature_importance(
            model_enh, enhanced_cols, ENHANCED_ENV_COLUMNS
        )
        test_ts = test_df.loc[X_test_e.index, COL_TIMESTAMP].reset_index(drop=True)
        test_forecast = pd.DataFrame(
            {
                COL_TIMESTAMP: test_ts,
                "actual": y_test.values,
                "benchmark_pred": pred_bench,
                "enhanced_pred": pred_enh,
            }
        )
        full_imp = pd.DataFrame(
            {"feature": enhanced_cols, "importance": model_enh.feature_importances_}
        ).sort_values("importance", ascending=False).reset_index(drop=True)

        status.update(label="实验完成", state="complete", expanded=False)

    return ExperimentReport(
        n_train=len(train_df),
        n_test=len(test_df),
        benchmark_metrics=metrics_bench,
        enhanced_metrics=metrics_enh,
        benchmark_revenue_yuan=rev_bench,
        enhanced_revenue_yuan=rev_enh,
        env_importance=env_imp,
        test_forecast=test_forecast,
        enhanced_full_importance=full_imp,
    )


def render_metric_cards(report: ExperimentReport) -> None:
    """大字高光指标卡片：精度 + 经济价值。"""
    bm, em = report.benchmark_metrics, report.enhanced_metrics
    rev_alpha = _pct_alpha(report.benchmark_revenue_yuan, report.enhanced_revenue_yuan)
    alpha_abs = report.enhanced_revenue_yuan - report.benchmark_revenue_yuan

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("RMSE（基准）", f"{bm.rmse:.2f}", help="元/MWh，越低越好")
    c2.metric("RMSE（增强）", f"{em.rmse:.2f}", delta=f"{em.rmse - bm.rmse:+.2f}")
    c3.metric("MAE（增强）", f"{em.mae:.2f}")
    c4.metric("R²（增强）", f"{em.r2:.4f}")
    c5.metric(
        "储能套利（增强）",
        f"¥{report.enhanced_revenue_yuan:,.0f}",
        delta=f"+¥{alpha_abs:,.0f}",
    )

    st.markdown(
        f'<p class="alpha-highlight">储能套利收益提升了 {rev_alpha:+.2f}%'
        f"（增量 Alpha，较基准 ¥{report.benchmark_revenue_yuan:,.2f}）</p>",
        unsafe_allow_html=True,
    )


def render_comparison_table(report: ExperimentReport) -> None:
    """HTML 对比表：统计精度 + 经济收益二维矩阵。"""
    bm, em = report.benchmark_metrics, report.enhanced_metrics
    rmse_chg = _pct_improvement(bm.rmse, em.rmse)
    mae_chg = _pct_improvement(bm.mae, em.mae)
    rev_alpha = _pct_alpha(report.benchmark_revenue_yuan, report.enhanced_revenue_yuan)

    table_html = f"""
    <table style="width:100%; border-collapse:collapse; font-size:0.95rem;">
      <thead>
        <tr style="background:#1e3a5f; color:#fff;">
          <th style="padding:10px; text-align:left;">维度</th>
          <th style="padding:10px;">指标</th>
          <th style="padding:10px;">基准模型</th>
          <th style="padding:10px;">增强模型</th>
          <th style="padding:10px;">变化</th>
        </tr>
      </thead>
      <tbody>
        <tr style="background:#f1f5f9;">
          <td rowspan="3" style="padding:10px; font-weight:600;">统计精度</td>
          <td style="padding:10px;">RMSE</td>
          <td style="padding:10px; text-align:center;">{bm.rmse:.4f}</td>
          <td style="padding:10px; text-align:center;">{em.rmse:.4f}</td>
          <td style="padding:10px; text-align:center;">{rmse_chg:+.2f}%</td>
        </tr>
        <tr style="background:#fff;">
          <td style="padding:10px;">MAE</td>
          <td style="padding:10px; text-align:center;">{bm.mae:.4f}</td>
          <td style="padding:10px; text-align:center;">{em.mae:.4f}</td>
          <td style="padding:10px; text-align:center;">{mae_chg:+.2f}%</td>
        </tr>
        <tr style="background:#f1f5f9;">
          <td style="padding:10px;">R²</td>
          <td style="padding:10px; text-align:center;">{bm.r2:.4f}</td>
          <td style="padding:10px; text-align:center;">{em.r2:.4f}</td>
          <td style="padding:10px; text-align:center;">—</td>
        </tr>
        <tr style="background:#ecfdf5;">
          <td style="padding:10px; font-weight:600;">经济价值</td>
          <td style="padding:10px;">储能套利总利润（元）</td>
          <td style="padding:10px; text-align:center;">{report.benchmark_revenue_yuan:,.2f}</td>
          <td style="padding:10px; text-align:center; font-weight:700;">
            {report.enhanced_revenue_yuan:,.2f}
          </td>
          <td style="padding:10px; text-align:center; color:#16a34a; font-weight:800;">
            {rev_alpha:+.2f}%
          </td>
        </tr>
      </tbody>
    </table>
    <p style="color:#64748b; font-size:0.85rem; margin-top:8px;">
      训练样本 {report.n_train} 条 · 测试样本 {report.n_test} 条 · 时序切分无 shuffle
    </p>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def plot_price_comparison(
    test_forecast: pd.DataFrame,
    window_hours: int = 72,
) -> go.Figure:
    """
    图表一：测试集电价时序对比（真实 vs 基准预测 vs 增强预测）。

    默认展示最后 ``window_hours`` 小时，便于观察尖峰贴合度。
    """
    df = test_forecast.sort_values(COL_TIMESTAMP).tail(window_hours).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df[COL_TIMESTAMP],
            y=df["actual"],
            name="真实电价",
            line=dict(color="#0f172a", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df[COL_TIMESTAMP],
            y=df["benchmark_pred"],
            name="基准模型预测",
            line=dict(color="#94a3b8", width=1.8, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df[COL_TIMESTAMP],
            y=df["enhanced_pred"],
            name="增强模型预测",
            line=dict(color="#2563eb", width=2),
        )
    )
    fig.update_layout(
        title=f"测试集电价预测对比（最近 {len(df)} 小时）",
        xaxis_title="时间",
        yaxis_title="现货电价（元/MWh）",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
    )
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    """
    图表二：增强模型特征重要性横向条形图，前三名高亮。
    """
    df = importance_df.head(top_n).sort_values("importance", ascending=True).copy()
    colors = []
    top3_names = set(df.tail(3)["feature"].tolist())
    for name in df["feature"]:
        if name in top3_names:
            colors.append("#2563eb")
        else:
            colors.append("#cbd5e1")

    fig = go.Figure(
        go.Bar(
            x=df["importance"],
            y=df["feature"],
            orientation="h",
            marker_color=colors,
            text=df["importance"].round(1),
            textposition="outside",
        )
    )
    fig.update_layout(
        title="增强模型 LightGBM 特征重要性（Top 3 蓝色高亮）",
        xaxis_title="Importance (split)",
        yaxis_title="",
        template="plotly_white",
        height=max(400, 28 * len(df)),
    )
    return fig


def render_charts(report: ExperimentReport) -> None:
    """渲染 Plotly 图表区。"""
    if report.test_forecast is None or report.enhanced_full_importance is None:
        st.warning("缺少可视化数据，请重新运行实验。")
        return

    window = st.slider(
        "时序图展示窗口（小时）",
        min_value=24,
        max_value=min(168, len(report.test_forecast)),
        value=min(72, len(report.test_forecast)),
        step=12,
    )

    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        st.plotly_chart(
            plot_price_comparison(report.test_forecast, window_hours=window),
            use_container_width=True,
        )
    with col_r:
        st.plotly_chart(
            plot_feature_importance(report.enhanced_full_importance),
            use_container_width=True,
        )

    st.subheader("环境因子专项排名")
    st.dataframe(
        report.env_importance.style.format({"importance": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    """应用入口。"""
    render_banner()
    config, run_clicked, use_pipeline, pipeline_refresh = build_config_from_sidebar()

    # 主区也放置醒目运行按钮（与侧边栏联动）
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        main_run = st.button(
            "⚡ 启动双模型打擂台（Run Experiment）",
            type="primary",
            use_container_width=True,
        )
    with col_info:
        if config.demo:
            mode_label = "Demo 模拟"
        elif use_pipeline:
            mode_label = "智能数据中台 · data_lake/mel_env_history.db"
        else:
            mode_label = f"真实数据 · {config.data_dir}"
        st.info(
            f"当前模式：**{mode_label}** · 训练占比 **{config.train_ratio:.0%}** · "
            "点击按钮开始实验（约 1–2 分钟）"
        )

    if use_pipeline:
        st.markdown(
            """
            <div style="padding:0.75rem 1rem; background:#ecfdf5; border-left:4px solid #16a34a; border-radius:6px; margin-bottom:1rem;">
            <strong>💡 提示：</strong>系统当前已自动触发 <strong>720 小时（30天）</strong> 工业级时序回溯，
            确保环境长尾微特征拥有充足的训练样本。
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not (run_clicked or main_run):
        st.markdown(
            """
            #### 使用指南
            1. 在左侧选择 **Demo 模拟数据**（默认，无需准备 CSV）或指定真实数据目录。  
            2. 调整 **时序划分比例**（默认 80% 训练 / 20% 测试）。  
            3. 点击 **启动双模型打擂台**，查看实证报告与 Plotly 图表。  
            """
        )
        return

    try:
        pipeline_tables = None
        if use_pipeline:
            with st.spinner(
                f"正在执行 {PIPELINE_LOOKBACK_HOURS}h（30天）数据中台日更并加载样本..."
            ):
                # 打擂台前显式 720h 回溯，与命令行 --lookback-hours 720 对齐
                pipeline_tables = load_tables_from_pipeline(
                    run_update=pipeline_refresh,
                    lookback_hours=PIPELINE_LOOKBACK_HOURS,
                )
        report = run_experiment_with_status(config, pipeline_tables=pipeline_tables)
    except Exception as exc:
        st.error(f"实验失败：{type(exc).__name__}: {exc}")
        st.exception(exc)
        return

    st.success("双模型擂台实验已完成")
    st.markdown("### 实证报告看板")
    render_metric_cards(report)
    render_comparison_table(report)

    st.markdown("---")
    st.markdown("### 专业可视化")
    render_charts(report)


if __name__ == "__main__":
    main()
