# -*- coding: utf-8 -*-
"""
experiment_app.py — 环境微特征三模型竞技场（基准 / LightGBM / PPO）Streamlit Web 控制台

启动方式
--------
    pip install streamlit plotly sqlalchemy psycopg2-binary
    streamlit run experiment_app.py

说明
----
- **三级容灾抽水**：PostgreSQL 生产库 → 云端内存 SQLite 沙盒 → Mock 自愈灌库。
- 任意层级均自动激活 ``v_features_pipeline_ready`` 窗口函数视图（LAG / AVG OVER）。
- 适配 Streamlit Cloud（无本地 PG、无 git 数据文件）零配置演示。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.pool import StaticPool

# 保证可导入同目录下的实验脚本
_APP_ROOT = Path(__file__).resolve().parent
if str(_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_APP_ROOT))

from db_feature_view import SQL_CREATE_VIEW  # noqa: E402
from db_injector import DEFAULT_DATABASE_URL, create_db_engine  # noqa: E402
from run_experiment import (  # noqa: E402
    BENCHMARK_FEATURE_COLUMNS,
    COL_TARGET,
    COL_TIMESTAMP,
    ENHANCED_ENV_COLUMNS,
    STORAGE_CYCLE_COST,
    STORAGE_ETA,
    STORAGE_MIN_SOC_RATIO,
    EXPERIMENT_LOOKBACK_HOURS_DEFAULT,
    EXPERIMENT_LOOKBACK_HOURS_MIN,
    ExperimentConfig,
    ExperimentReport,
    PRODUCTION_DATABASE_URL,
    build_full_feature_wide_table,
    effective_lookback_hours,
    ensure_sufficient_wide_table,
    _pct_alpha,
    _pct_improvement,
    _ensure_feature_view,
    add_model_ready_features,
    adapt_sql_pipeline_to_model_frame,
    backtest_storage_revenue_yuan,
    build_enhanced_wide_table,
    build_synthetic_raw_tables,
    evaluate_regression,
    extract_env_feature_importance,
    load_raw_tables,
    temporal_train_test_split,
    train_quantile_lgbm,
    _import_optimizer,
    _robust_fill_enhanced_env_features,
    _sanitize_xy,
)
from frontier_physics_constants import (  # noqa: E402
    FRONTIER_PHYSICS_BLOCKS,
    audit_frontier_physics_features,
)
from data_pipeline import EnvironmentalDataPipeline, PipelineConfig  # noqa: E402
from environmental_feature_engineer import (  # noqa: E402
    FeatureSchema,
    enrich_market_physics_inputs,
    merge_frontier_physics_features,
)

# 工业级默认回溯窗口：30 天 × 24h，保证 LightGBM 有足够样本学习环境长尾微特征
PIPELINE_LOOKBACK_HOURS: int = 720

# 建模拉取：库内窗口函数实时产出特征宽表（全节点、时间升序）
SQL_LOAD_PIPELINE_WIDE = """
SELECT *
FROM v_features_pipeline_ready
ORDER BY timestamp ASC
"""

SQL_PREVIEW_SAMPLE = """
SELECT *
FROM v_features_pipeline_ready
ORDER BY RANDOM()
LIMIT 5
"""

# SQLite 内存沙盒 DDL（与 PostgreSQL 语义对齐，供 Streamlit Cloud 容灾）
DDL_SQLITE_MARKET_NODES = """
CREATE TABLE IF NOT EXISTS market_nodes (
    node_id    TEXT PRIMARY KEY,
    rto_name   TEXT NOT NULL,
    zone_name  TEXT NOT NULL,
    node_type  TEXT NOT NULL
);
"""

DDL_SQLITE_RTO_METRICS = """
CREATE TABLE IF NOT EXISTS rto_hourly_metrics (
    timestamp                  TEXT NOT NULL,
    node_id                    TEXT NOT NULL,
    price_da                   REAL,
    price_rt                   REAL,
    system_load                REAL,
    heat_index                 REAL,
    wind_shear_alpha           REAL,
    bifacial_gain_index        REAL,
    panel_efficiency_discount  REAL,
    PRIMARY KEY (timestamp, node_id),
    FOREIGN KEY (node_id) REFERENCES market_nodes (node_id)
);
"""

# 窗口视图（SQLite 用 DROP + CREATE；PostgreSQL 用 OR REPLACE，见激活函数）
SQL_CREATE_VIEW_SQLITE = """
CREATE VIEW v_features_pipeline_ready AS
SELECT
    m.timestamp,
    m.node_id,
    n.rto_name,
    n.zone_name,
    n.node_type,
    m.price_da,
    m.price_rt,
    m.system_load,
    m.heat_index,
    m.wind_shear_alpha,
    m.bifacial_gain_index,
    m.panel_efficiency_discount,
    (m.price_da - m.price_rt) AS basis_spread,
    LAG(m.price_rt, 1) OVER (
        PARTITION BY m.node_id
        ORDER BY m.timestamp
    ) AS price_rt_lag_1h,
    AVG(m.price_rt) OVER (
        PARTITION BY m.node_id
        ORDER BY m.timestamp
        ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING
    ) AS price_rt_ma_24h
FROM rto_hourly_metrics AS m
LEFT JOIN market_nodes AS n
    ON n.node_id = m.node_id;
"""

CSV_FALLBACK_CANDIDATES: tuple[Path, ...] = (
    _APP_ROOT / "data" / "features_ready.csv",
    _APP_ROOT / "data" / "feature_ready.csv",
)

MOCK_SEED_HOURS = EXPERIMENT_LOOKBACK_HOURS_DEFAULT
DEFAULT_MOCK_NODE = "PJM_HUB"
SESSION_ENGINE_KEY = "mel_feature_engine"
SESSION_TIER_KEY = "mel_data_tier"
SESSION_EXPERIMENT_REPORT = "mel_experiment_report"
SESSION_EXPERIMENT_RUN = "experiment_executed"
SESSION_PPO_REPORT = "mel_ppo_report"
SESSION_ARENA_PNL = "mel_arena_pnl_df"
SESSION_PPO_EXECUTED = "ppo_executed"

PPO_WEB_EPOCHS = 5
PPO_WEB_ROLLOUT = 256
PPO_POLICY_PATH = _APP_ROOT / "artifacts" / "vpp_ppo_policy.pt"

# ---------------------------------------------------------------------------
# I18n 双语矩阵 (Chinese / English)
# ---------------------------------------------------------------------------
LOCALES: dict[str, dict[str, str]] = {
    "zh": {
        "page_title": "环境微特征 · 电价预测增强试验舱",
        "banner_title": "量化前沿：环境微特征（辐射/污染/水温）电力电价预测增强试验舱",
        "banner_subtitle": "Quant Frontier Lab · PostgreSQL 窗口算力 × 储能套利经济价值实证",
        "tier1_alert": "⚡ 生产级 PostgreSQL 数据库连接成功！已激活 TimescaleDB/PG 混合算力引擎",
        "tier2_alert": "☁️ PostgreSQL 不可用（云端环境）：已降级至内存 SQLite 高仿真沙盒，SQL 窗口视图已激活。",
        "tier3_alert": "🛡️ Streamlit Cloud 容灾：已启用内存 SQLite 沙盒，并动态 Mock {hours}h 仿真时序（含四大物理因子 + 窗口函数视图）。",
        "tier1_label": "Tier-1 · PostgreSQL 生产库",
        "tier2_label": "Tier-2 · 云端内存 SQLite 沙盒",
        "tier3_label": "Tier-3 · 内存沙盒 + Mock 自愈灌库",
        "pipeline_caption": "数据管线：**{tier}** · Engine=`{engine}`",
        "sql_expander_title": "🔍 查看后端核心 SQL 窗口函数架构 (Advanced SQL Code)",
        "sql_expander_caption": "视图在库内完成特征工程；`ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING` 严格排除当前行，防止未来信息泄露（No Look-ahead）。",
        "sql_query_caption": "建模拉取语句（Web 试验舱实时查询）",
        "sql_preview_caption": "随机抽样预览（面试官可见窗口函数已生效）",
        "sql_preview_empty": "预览数据为空，请点击「启动双模型打擂台」触发自愈灌库。",
        "sidebar_console": "试验控制台",
        "data_mode": "数据模式",
        "data_mode_postgres": "PostgreSQL 生产库 (推荐)",
        "data_mode_demo": "Demo 模拟数据",
        "data_mode_sqlite": "智能数据中台 (SQLite)",
        "data_mode_csv": "真实数据目录",
        "data_mode_help": "推荐：PostgreSQL 窗口特征视图；失败自动降级 CSV。",
        "pg_url_label": "PostgreSQL URL",
        "pg_url_help": "可通过环境变量 MEL_DATABASE_URL 覆盖。",
        "pipeline_refresh": "运行前先执行日更（回溯 {hours}h / 30天）",
        "pipeline_hint": "💡 提示：系统当前已自动触发 720 小时（30天）工业级时序回溯，确保环境长尾微特征拥有充足的训练样本。",
        "lookback_hours": "历史回溯小时数 (lookback_hours)",
        "lookback_help": "不低于 {min_h}h（30天）；默认 {default_h}h 保障训练样本 ≥1000。",
        "train_ratio": "时序划分比例（训练集占比）",
        "train_ratio_help": "严格按时间顺序切分，防止未来信息泄露。",
        "data_dir": "真实数据目录路径",
        "storage_params": "储能回测参数（与 run_experiment 默认一致）",
        "power_mw": "额定功率 (MW)",
        "capacity_mwh": "容量 (MWh)",
        "run_button": "⚡ 启动三模型竞技场（Run 3-Model Arena）",
        "ppo_btn": "🤖 运行 PPO 智能体深度强化学习打擂台 (Run PPO Agent Control)",
        "ppo_spinner": "PyTorch PPO 智能体训练中（Web 轻量 {epochs} Epochs）...",
        "ppo_done": "PPO 强化学习智能体回测完成",
        "ppo_fail": "PPO 训练失败：{exc}",
        "ppo_need_lgbm": "请先运行三模型竞技场主实验（LightGBM 擂台），再启动 PPO 智能体。",
        "ppo_missing_torch": "未安装 PyTorch / Gymnasium。请执行: pip install torch gymnasium",
        "ppo_reset": "🔄 重置 PPO 实验 / Reset PPO",
        "arena_title": "### 🏆 终极三模型竞技场 (3-Model Arena Showdown)",
        "arena_caption": "固定基准策略 · LightGBM 环境增强 · PyTorch PPO 深度强化学习 — 同一测试集储能套利正面对决",
        "storage_bench": "储能套利（基准）",
        "storage_lgbm": "储能套利（LightGBM）",
        "storage_ppo": "储能套利（PPO DRL）",
        "profitable_hours": "盈利小时占比",
        "beat_idle_hours": "优于空仓小时占比",
        "alpha_vs_idle": "超额收益率 (Alpha vs 空仓)",
        "table_ppo": "PPO 智能体",
        "table_lgbm": "LightGBM 增强",
        "chart_cum_pnl_title": "终极三资产累计收益率曲线对比 (Cumulative PnL Showdown)",
        "chart_cum_pnl_y": "累计套利收益（元）",
        "curve_bench": "固定基准策略 (Benchmark Optimizer)",
        "curve_lgbm": "LightGBM 预测策略 (ML Forecast)",
        "curve_ppo": "PPO 智能体动作策略 (PyTorch DRL)",
        "mdp_expander_title": "🤖 查看马尔可夫决策过程 (MDP) 状态空间与奖励设计",
        "mdp_state_title": "**状态空间 State**",
        "mdp_state_body": (
            "s_t = [ x_t^{{SQL}} ; SOC_t / E_{{max}} ]，其中 x_t^{{SQL}} 来自 "
            "``v_features_pipeline_ready`` 窗口视图映射后的建模宽表：\n\n"
            "- **日前/实时电价代理**: ``spot_price_lag_1h``, ``spot_price_lag_24h``, ``price_rt_ma_24h``\n"
            "- **系统负荷**: ``total_load``\n"
            "- **环境微特征**: 辐射/污染/水温滞后 + 四大前沿物理因子\n"
            "- **储能荷电状态**: SOC_t ∈ [SOC_{{min}}, E_{{max}}]\n\n"
            "共 {obs_dim} 维连续观测（Gymnasium ``Box``）。"
        ),
        "mdp_reward_title": "**奖励函数 Reward**",
        "mdp_reward_body": (
            "r_t = PnL(p_t^{{RT}}, a_t) - λ_{{deg}} · max(-a_t, 0)\n\n"
            "其中 a_t ∈ [-P_{{max}}, P_{{max}}] MW（正=放电），p_t^{{RT}} 为测试集真实结算电价 "
            "（``spot_price`` / ``price_rt``），λ_{{deg}} = {deg_cost} 元/MWh（电池非线性退化惩罚）。"
        ),
        "experiment_done": "三模型竞技场主实验已完成",
        "report_board": "### 🏆 三模型实证报告看板",
        "alpha_three_way": "PPO 相对 LightGBM 储能 Alpha: {pct:+.2f}%（¥{ppo:,.0f} vs ¥{lgbm:,.0f}）",
        "mode_demo": "Demo 模拟",
        "mode_postgres": "PostgreSQL 生产库 · {url}",
        "mode_sqlite": "智能数据中台 · data_lake/mel_env_history.db",
        "mode_csv": "真实数据 · {path}",
        "mode_info": "当前模式：**{mode}** · 训练占比 **{ratio:.0%}** · 点击按钮开始实验（约 1–2 分钟）",
        "pipeline_banner": "**💡 提示：**系统当前已自动触发 **720 小时（30天）** 工业级时序回溯，确保环境长尾微特征拥有充足的训练样本。",
        "guide_title": "#### 使用指南",
        "guide_body": (
            "1. 默认 **三级容灾抽水**：PostgreSQL → 内存 SQLite → Mock 自愈（适配 Streamlit Cloud）。  \n"
            "2. 点击 **启动三模型竞技场** 运行基准 vs LightGBM 擂台。  \n"
            "3. 再点击 **PPO 智能体打擂台**（轻量 5 Epochs）解锁累计 PnL 三曲线对比与 MDP 文档。"
        ),
        "tier_spinner": "三级容灾抽水：PG → 内存 SQLite → Mock 自愈...",
        "pipeline_spinner": "正在执行 {hours}h（30天）数据中台日更并加载样本...",
        "data_init_error": "数据管线初始化失败：{exc}",
        "status_running": "实验引擎运行中...",
        "status_done": "实验完成",
        "step_sql_wide": "正在消费 SQL 窗口特征宽表（v_features_pipeline_ready）...",
        "step_align": "正在调用特征中台对齐空间辐射数据...",
        "step_physics": "正在计算四大前沿物理微特征（酷热/风切变/反照率/积尘）...",
        "step_demo": "正在生成 Demo 大样本宽表（{hours}h）...",
        "sample_stats": "建模样本：全量 {total} 行 · 训练 {train} · 测试 {test}",
        "step_bench": "正在训练基准 LightGBM...",
        "step_enhanced": "正在引入环境微特征训练增强模型...",
        "step_storage": "正在通过 SciPy 引擎进行储能套利回测...",
        "dir_missing": "真实数据目录不存在: {path}。请创建 data/ 或切换 PostgreSQL / Demo 模式。",
        "missing_cols": "宽表缺少列: {cols}",
        "train_too_small": "训练集仅 {n} 条（目标 ≥{min}）。请增大侧边栏 lookback_hours（当前 {hours}）。",
        "pipeline_empty_warn": "本地库为空或不全，自动触发 {hours} 小时回溯日更...",
        "experiment_fail": "实验失败：{exc}",
        "viz_section": "### 专业可视化 · 三模型竞技",
        "rmse_bench": "RMSE（基准）",
        "rmse_enh": "RMSE（增强）",
        "rmse_help": "元/MWh，越低越好",
        "mae_enh": "MAE（增强）",
        "r2_enh": "R²（增强）",
        "storage_enh": "储能套利（增强）",
        "alpha_highlight": "储能套利收益提升了 {pct:+.2f}%（增量 Alpha，较基准 ¥{bench:,.0f}）",
        "table_dim": "维度",
        "table_metric": "指标",
        "table_bench": "基准模型",
        "table_enh": "增强模型",
        "table_change": "变化",
        "table_stat_acc": "统计精度",
        "table_econ": "经济价值",
        "table_storage_profit": "储能套利总利润（元）",
        "table_footer": "训练样本 {train} 条 · 测试样本 {test} 条 · 时序切分无 shuffle",
        "chart_actual": "真实电价",
        "chart_bench_pred": "基准模型预测",
        "chart_enh_pred": "增强模型预测",
        "chart_price_title": "测试集电价预测对比（最近 {n} 小时）",
        "chart_time": "时间",
        "chart_price_y": "现货电价（元/MWh）",
        "chart_fi_title": "增强模型 LightGBM 特征重要性（Top 3 蓝色高亮）",
        "chart_fi_x": "Importance (split)",
        "shap_title": "前沿物理因子 · SHAP 实时解释",
        "shap_caption": "基于增强模型在测试集上的 TreeSHAP，聚合四大微观物理板块的 |SHAP| 强度，量化其在打擂台中的超额边际贡献（非因果效应，仅供业务解读）。",
        "shap_ok": "已并入",
        "shap_miss": "缺失",
        "shap_run_first": "请先完成一次擂台实验以生成 SHAP 解释。",
        "shap_spinner": "正在计算 TreeSHAP（前沿物理板块）...",
        "shap_missing_lib": "未安装 ``shap`` 库。请执行: pip install shap",
        "shap_chart_title": "四大前沿物理因子 · 平均 |SHAP| 超额贡献",
        "shap_chart_x": "Mean |SHAP| (测试集)",
        "env_rank": "环境因子专项排名",
        "viz_missing": "缺少可视化数据，请重新运行实验。",
        "chart_window": "时序图展示窗口（小时）",
    },
    "en": {
        "page_title": "Env Micro-Features · Price Forecast Lab",
        "banner_title": "Quant Frontier: Environmental Micro-Features Power Price Prediction Enhancement Sandbox",
        "banner_subtitle": "Quant Frontier Lab · PostgreSQL Window Compute × Storage Arbitrage PnL",
        "tier1_alert": "⚡ Production PostgreSQL connected! TimescaleDB/PG hybrid compute engine active.",
        "tier2_alert": "☁️ PostgreSQL unavailable (cloud): fell back to in-memory SQLite sandbox; SQL window view active.",
        "tier3_alert": "🛡️ Streamlit Cloud Resilience: Memory SQLite sandbox activated; dynamic {hours}h mock series injected (4 physics factors + window view).",
        "tier1_label": "Tier-1 · PostgreSQL Production",
        "tier2_label": "Tier-2 · Cloud In-Memory SQLite Sandbox",
        "tier3_label": "Tier-3 · Memory Sandbox + Mock Self-Heal",
        "pipeline_caption": "Pipeline: **{tier}** · Engine=`{engine}`",
        "sql_expander_title": "🔍 Backend SQL Window Architecture (Advanced SQL Code)",
        "sql_expander_caption": "Features computed in-database; `ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING` excludes the current row to prevent look-ahead leakage.",
        "sql_query_caption": "Model pull query (live Web lab)",
        "sql_preview_caption": "Random sample preview (proves window functions are live)",
        "sql_preview_empty": "Preview empty — click Run Experiment to trigger self-healing ingest.",
        "sidebar_console": "Experiment Console",
        "data_mode": "Data mode",
        "data_mode_postgres": "PostgreSQL Production (Recommended)",
        "data_mode_demo": "Demo Synthetic Data",
        "data_mode_sqlite": "Smart Data Lake (SQLite)",
        "data_mode_csv": "Local CSV Directory",
        "data_mode_help": "Recommended: PostgreSQL window feature view; auto CSV fallback on failure.",
        "pg_url_label": "PostgreSQL URL",
        "pg_url_help": "Override via MEL_DATABASE_URL env var.",
        "pipeline_refresh": "Run daily update first (lookback {hours}h / 30d)",
        "pipeline_hint": "💡 Tip: 720-hour (30-day) industrial lookback ensures enough samples for long-tail micro-features.",
        "lookback_hours": "Lookback hours (lookback_hours)",
        "lookback_help": "Min {min_h}h (30d); default {default_h}h targets ≥1000 training rows.",
        "train_ratio": "Chronological train ratio",
        "train_ratio_help": "Time-ordered split only — no future leakage.",
        "data_dir": "Local data directory path",
        "storage_params": "Storage backtest params (run_experiment defaults)",
        "power_mw": "Rated power (MW)",
        "capacity_mwh": "Capacity (MWh)",
        "run_button": "⚡ Run 3-Model Arena (Run Experiment)",
        "ppo_btn": "🤖 Execute PPO Reinforcement Learning Strategy",
        "ppo_spinner": "Training PyTorch PPO agent (lightweight {epochs} epochs)...",
        "ppo_done": "PPO agent backtest complete",
        "ppo_fail": "PPO training failed: {exc}",
        "ppo_need_lgbm": "Run the main 3-Model Arena experiment (LightGBM) before launching PPO.",
        "ppo_missing_torch": "PyTorch / Gymnasium not installed. Run: pip install torch gymnasium",
        "ppo_reset": "🔄 Reset PPO experiment / Reset",
        "arena_title": "### 🏆 Ultimate 3-Model Arena Showdown",
        "arena_caption": "Benchmark optimizer · LightGBM enhanced forecast · PyTorch PPO DRL — same test-set storage arbitrage",
        "storage_bench": "Storage arbitrage (Benchmark)",
        "storage_lgbm": "Storage arbitrage (LightGBM)",
        "storage_ppo": "Storage arbitrage (PPO DRL)",
        "profitable_hours": "Profitable hours ratio",
        "beat_idle_hours": "Beat-idle hours ratio",
        "alpha_vs_idle": "Alpha vs idle benchmark",
        "table_ppo": "PPO agent",
        "table_lgbm": "LightGBM enhanced",
        "chart_cum_pnl_title": "Cumulative PnL Showdown — Three Asset Curves",
        "chart_cum_pnl_y": "Cumulative arbitrage profit (CNY)",
        "curve_bench": "Fixed benchmark strategy (optimizer)",
        "curve_lgbm": "LightGBM forecast strategy (ML)",
        "curve_ppo": "PPO agent action strategy (PyTorch DRL)",
        "mdp_expander_title": "🤖 MDP state space & reward design",
        "mdp_state_title": "**State space**",
        "mdp_state_body": (
            "s_t = [ x_t^{{SQL}} ; SOC_t / E_{{max}} ], where x_t^{{SQL}} is mapped from "
            "``v_features_pipeline_ready`` window view:\n\n"
            "- **Day-ahead / real-time price proxies**: ``spot_price_lag_1h``, ``spot_price_lag_24h``, ``price_rt_ma_24h``\n"
            "- **System load**: ``total_load``\n"
            "- **Environmental micro-features**: radiation / pollution / water lags + frontier physics blocks\n"
            "- **Battery SOC**: SOC_t ∈ [SOC_{{min}}, E_{{max}}]\n\n"
            "{obs_dim}-dim continuous observation (Gymnasium ``Box``)."
        ),
        "mdp_reward_title": "**Reward**",
        "mdp_reward_body": (
            "r_t = PnL(p_t^{{RT}}, a_t) - λ_{{deg}} · max(-a_t, 0)\n\n"
            "a_t ∈ [-P_{{max}}, P_{{max}}] MW (+ = discharge), p_t^{{RT}} is realized settlement price, "
            "λ_{{deg}} = {deg_cost} CNY/MWh (non-linear degradation penalty)."
        ),
        "experiment_done": "3-Model Arena main experiment complete",
        "report_board": "### 🏆 3-Model empirical dashboard",
        "alpha_three_way": "PPO vs LightGBM storage alpha: {pct:+.2f}% (¥{ppo:,.0f} vs ¥{lgbm:,.0f})",
        "mode_demo": "Demo synthetic",
        "mode_postgres": "PostgreSQL production · {url}",
        "mode_sqlite": "Data lake · data_lake/mel_env_history.db",
        "mode_csv": "Local CSV · {path}",
        "mode_info": "Mode: **{mode}** · Train ratio **{ratio:.0%}** · Click to run (~1–2 min)",
        "pipeline_banner": "**💡 Tip:** 720-hour (30-day) lookback is enabled for industrial-grade environmental time series.",
        "guide_title": "#### Quick Start",
        "guide_body": (
            "1. Default **3-tier pipeline**: PostgreSQL → in-memory SQLite → mock self-heal (Streamlit Cloud ready).  \n"
            "2. Click **Run 3-Model Arena** for benchmark vs LightGBM.  \n"
            "3. Then click **Execute PPO** (5 lightweight epochs) for cumulative PnL curves & MDP docs."
        ),
        "tier_spinner": "3-tier sourcing: PG → memory SQLite → mock self-heal...",
        "pipeline_spinner": "Running {hours}h (30d) data-lake daily update & load...",
        "data_init_error": "Data pipeline init failed: {exc}",
        "status_running": "Experiment engine running...",
        "status_done": "Experiment complete",
        "step_sql_wide": "Loading SQL window feature wide table (v_features_pipeline_ready)...",
        "step_align": "Aligning spatial radiation via feature lake...",
        "step_physics": "Computing frontier physics micro-features (heat/shear/albedo/soiling)...",
        "step_demo": "Building demo wide table ({hours}h)...",
        "sample_stats": "Samples: total {total} · train {train} · test {test}",
        "step_bench": "Training benchmark LightGBM...",
        "step_enhanced": "Training enhanced model with environmental features...",
        "step_storage": "Running storage arbitrage backtest (SciPy optimizer)...",
        "dir_missing": "Data directory not found: {path}. Add CSV files or switch PostgreSQL / Demo mode.",
        "missing_cols": "Wide table missing columns: {cols}",
        "train_too_small": "Only {n} training rows (need ≥{min}). Increase lookback_hours (current {hours}).",
        "pipeline_empty_warn": "Local DB empty/incomplete — auto {hours}h daily refresh...",
        "experiment_fail": "Experiment failed: {exc}",
        "viz_section": "### Professional visualizations · 3-model arena",
        "rmse_bench": "RMSE (Benchmark)",
        "rmse_enh": "RMSE (Enhanced)",
        "rmse_help": "CNY/MWh, lower is better",
        "mae_enh": "MAE (Enhanced)",
        "r2_enh": "R² (Enhanced)",
        "storage_enh": "Storage arbitrage (Enhanced)",
        "alpha_highlight": "Storage revenue up {pct:+.2f}% (incremental alpha vs benchmark ¥{bench:,.0f})",
        "table_dim": "Dimension",
        "table_metric": "Metric",
        "table_bench": "Benchmark",
        "table_enh": "Enhanced",
        "table_change": "Change",
        "table_stat_acc": "Statistical accuracy",
        "table_econ": "Economic value",
        "table_storage_profit": "Storage arbitrage profit (CNY)",
        "table_footer": "Train {train} · Test {test} · chronological split, no shuffle",
        "chart_actual": "Actual price",
        "chart_bench_pred": "Benchmark forecast",
        "chart_enh_pred": "Enhanced forecast",
        "chart_price_title": "Test-set price forecast (last {n} hours)",
        "chart_time": "Time",
        "chart_price_y": "Spot price (CNY/MWh)",
        "chart_fi_title": "Enhanced LightGBM feature importance (Top 3 highlighted)",
        "chart_fi_x": "Importance (split)",
        "shap_title": "Frontier physics · live SHAP explainability",
        "shap_caption": "TreeSHAP on the test set, aggregated |SHAP| by physics block (interpretability, not causal inference).",
        "shap_ok": "OK",
        "shap_miss": "Missing",
        "shap_run_first": "Run an arena experiment first to generate SHAP outputs.",
        "shap_spinner": "Computing TreeSHAP (physics blocks)...",
        "shap_missing_lib": "Package ``shap`` not installed. Run: pip install shap",
        "shap_chart_title": "Frontier physics blocks · mean |SHAP| contribution",
        "shap_chart_x": "Mean |SHAP| (test set)",
        "env_rank": "Environmental factor ranking",
        "viz_missing": "Missing visualization data — re-run the experiment.",
        "chart_window": "Price chart window (hours)",
    },
}

_DATA_MODE_KEYS: tuple[str, ...] = ("postgres", "demo", "sqlite", "csv")


def t(lang: str, key: str, **kwargs: object) -> str:
    """I18n lookup with optional ``str.format`` placeholders."""
    text = LOCALES[lang][key]
    return text.format(**kwargs) if kwargs else text


def resolve_lang() -> str:
    """侧边栏语言切换：English → en，简体中文 → zh。"""
    choice = st.sidebar.selectbox(
        "🌐 Language / 语言切换",
        ["English", "简体中文"],
        key="language_selector",
    )
    return "en" if choice == "English" else "zh"


def tier_label_for(lang: str, tier: int) -> str:
    return t(lang, f"tier{tier}_label")


def init_session_state() -> None:
    """Streamlit Session State 初始化（防止按钮 rerun 后结果丢失）。"""
    if SESSION_PPO_EXECUTED not in st.session_state:
        st.session_state[SESSION_PPO_EXECUTED] = False
    if SESSION_EXPERIMENT_RUN not in st.session_state:
        st.session_state[SESSION_EXPERIMENT_RUN] = False


@dataclass
class DataSourcingContext:
    """三级容灾抽水结果上下文。"""

    tier: Literal[1, 2, 3]
    postgres_connected: bool
    engine: Optional[Engine]
    sql_preview: pd.DataFrame
    database_url: str

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


def render_banner(lang: str) -> None:
    """顶部专业 Banner。"""
    st.markdown(
        f"""
        <div class="main-banner">
            <h1>{t(lang, "banner_title")}</h1>
            <p>{t(lang, "banner_subtitle")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_tier_banner(ctx: DataSourcingContext, lang: str) -> None:
    """主界面：按容灾层级展示状态 + SQL 架构 + 抽样数据。"""
    if ctx.tier == 1:
        st.success(t(lang, "tier1_alert"))
    elif ctx.tier == 2:
        st.info(t(lang, "tier2_alert"))
    else:
        st.warning(t(lang, "tier3_alert", hours=MOCK_SEED_HOURS))

    engine_name = ctx.engine.dialect.name if ctx.engine else "n/a"
    st.caption(
        t(
            lang,
            "pipeline_caption",
            tier=tier_label_for(lang, ctx.tier),
            engine=engine_name,
        )
    )

    with st.expander(t(lang, "sql_expander_title")):
        st.caption(t(lang, "sql_expander_caption"))
        st.code(SQL_CREATE_VIEW.strip(), language="sql")
        st.caption(t(lang, "sql_query_caption"))
        st.code(SQL_LOAD_PIPELINE_WIDE.strip(), language="sql")
        st.caption(t(lang, "sql_preview_caption"))
        if ctx.sql_preview is not None and not ctx.sql_preview.empty:
            st.dataframe(ctx.sql_preview, use_container_width=True, hide_index=True)
        else:
            st.info(t(lang, "sql_preview_empty"))


def create_memory_sqlite_engine() -> Engine:
    """第二级：云端内存 SQLite 沙盒（StaticPool 保持 :memory: 库不销毁）。"""
    return create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


def activate_feature_view_on_engine(engine: Engine) -> None:
    """在 PostgreSQL 或 SQLite 上幂等激活窗口特征视图。"""
    dialect = engine.dialect.name
    with engine.begin() as conn:
        if dialect == "postgresql":
            conn.execute(text(SQL_CREATE_VIEW))
        else:
            conn.execute(text("DROP VIEW IF EXISTS v_features_pipeline_ready"))
            conn.execute(text(DDL_SQLITE_MARKET_NODES))
            conn.execute(text(DDL_SQLITE_RTO_METRICS))
            conn.execute(text(SQL_CREATE_VIEW_SQLITE))


def _count_metrics_rows(engine: Engine) -> int:
    with engine.connect() as conn:
        try:
            val = conn.execute(text("SELECT COUNT(*) FROM rto_hourly_metrics")).scalar()
            return int(val or 0)
        except OperationalError:
            return 0


def generate_mock_seed_tables(n_hours: int = MOCK_SEED_HOURS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """第三级：动态 Mock 大样本仿真时序 + 四大物理因子（默认 ≥1536h）。"""
    hours = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    t = np.arange(n_hours, dtype=float)

    price_da = 300.0 + 50.0 * np.sin(t / 24.0) + rng.normal(0, 5.0, n_hours)
    price_rt = price_da + rng.normal(0, 1.5, n_hours)
    system_load = 11000.0 + 500.0 * np.sin(t / 24.0) + rng.normal(0, 80.0, n_hours)

    nodes = pd.DataFrame(
        [
            {
                "node_id": DEFAULT_MOCK_NODE,
                "rto_name": "PJM",
                "zone_name": "HUB_ZONE",
                "node_type": "HUB",
            }
        ]
    )
    facts = pd.DataFrame(
        {
            "timestamp": hours.strftime("%Y-%m-%d %H:%M:%S"),
            "node_id": DEFAULT_MOCK_NODE,
            "price_da": price_da,
            "price_rt": price_rt,
            "system_load": system_load,
            "heat_index": 25.0 + rng.normal(0, 1.0, n_hours),
            "wind_shear_alpha": np.clip(0.15 + rng.normal(0, 0.02, n_hours), 0.05, 0.35),
            "bifacial_gain_index": 1.0 + rng.normal(0, 0.05, n_hours),
            "panel_efficiency_discount": np.clip(
                0.02 + np.cumsum(rng.normal(0, 0.001, n_hours)), 0.0, 0.15
            ),
        }
    )
    return nodes, facts


def _hydrate_sqlite_from_csv(engine: Engine, data_dir: Optional[Path]) -> bool:
    """尝试将本地 CSV 注入内存库（Streamlit Cloud 通常无文件，静默跳过）。"""
    candidates = list(CSV_FALLBACK_CANDIDATES)
    if data_dir is not None:
        candidates.insert(0, data_dir / "features_ready.csv")

    for path in candidates:
        if not path.is_file():
            continue
        try:
            raw = pd.read_csv(path)
            if raw.empty or "timestamp" not in raw.columns:
                continue
            ts = pd.to_datetime(raw["timestamp"], utc=True, errors="coerce")
            price_col = raw["price_da"] if "price_da" in raw.columns else raw.get("spot_price")
            rt_col = raw["price_rt"] if "price_rt" in raw.columns else price_col
            node_series = (
                raw["node_id"].astype(str)
                if "node_id" in raw.columns
                else pd.Series([DEFAULT_MOCK_NODE] * len(raw))
            )
            facts = pd.DataFrame(
                {
                    "timestamp": ts.dt.strftime("%Y-%m-%d %H:%M:%S"),
                    "node_id": node_series,
                    "price_da": pd.to_numeric(price_col, errors="coerce"),
                    "price_rt": pd.to_numeric(rt_col, errors="coerce"),
                    "system_load": pd.to_numeric(
                        raw.get("system_load", raw.get("total_load", 0)), errors="coerce"
                    ),
                    "heat_index": pd.to_numeric(raw.get("heat_index", 0), errors="coerce"),
                    "wind_shear_alpha": pd.to_numeric(
                        raw.get("wind_shear_alpha", raw.get("wind_shear_coefficient", 0)),
                        errors="coerce",
                    ),
                    "bifacial_gain_index": pd.to_numeric(
                        raw.get("bifacial_gain_index", 0), errors="coerce"
                    ),
                    "panel_efficiency_discount": pd.to_numeric(
                        raw.get("panel_efficiency_discount", 0), errors="coerce"
                    ),
                }
            ).dropna(subset=["timestamp"])
            if facts.empty:
                continue
            node_id = str(facts["node_id"].iloc[0])[:20]
            rto_name = str(raw["rto_name"].iloc[0]) if "rto_name" in raw.columns else "PJM"
            nodes = pd.DataFrame(
                [
                    {
                        "node_id": node_id,
                        "rto_name": rto_name,
                        "zone_name": "HUB_ZONE",
                        "node_type": "HUB",
                    }
                ]
            )
            with engine.begin() as conn:
                conn.execute(text(DDL_SQLITE_MARKET_NODES))
                conn.execute(text(DDL_SQLITE_RTO_METRICS))
            nodes.to_sql("market_nodes", engine, if_exists="append", index=False)
            facts.to_sql("rto_hourly_metrics", engine, if_exists="append", index=False, method="multi")
            return True
        except Exception:  # noqa: BLE001
            continue
    return False


def bootstrap_memory_sqlite_database(
    engine: Engine,
    data_dir: Optional[Path] = None,
) -> Literal[2, 3]:
    """
    初始化内存库：建表 → CSV 注水（若有）→ Mock 自愈（若无数据）。
    返回实际启用的子层级 2（CSV）或 3（Mock）。
    """
    with engine.begin() as conn:
        conn.execute(text(DDL_SQLITE_MARKET_NODES))
        conn.execute(text(DDL_SQLITE_RTO_METRICS))

    if _count_metrics_rows(engine) == 0:
        if _hydrate_sqlite_from_csv(engine, data_dir):
            tier_sub: Literal[2, 3] = 2
        else:
            nodes, facts = generate_mock_seed_tables()
            nodes.to_sql("market_nodes", engine, if_exists="append", index=False)
            facts.to_sql(
                "rto_hourly_metrics",
                engine,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=5000,
            )
            tier_sub = 3
    else:
        tier_sub = 2

    activate_feature_view_on_engine(engine)
    return tier_sub


def fetch_sql_pipeline_wide(engine: Engine) -> pd.DataFrame:
    """从已激活视图的引擎拉取特征宽表。"""
    df = pd.read_sql_query(SQL_LOAD_PIPELINE_WIDE, engine)
    if df.empty:
        raise ValueError("v_features_pipeline_ready 查询结果为空。")
    return df


def fetch_sql_preview_sample(engine: Engine) -> pd.DataFrame:
    try:
        return pd.read_sql_query(SQL_PREVIEW_SAMPLE, engine)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


def try_tier1_postgresql(database_url: str) -> Optional[DataSourcingContext]:
    """第一级：真实 PostgreSQL 生产环境。"""
    try:
        engine = create_db_engine(database_url)
        _ensure_feature_view(engine)
        df_sql = fetch_sql_pipeline_wide(engine)
        preview = fetch_sql_preview_sample(engine)
        return DataSourcingContext(
            tier=1,
            postgres_connected=True,
            engine=engine,
            sql_preview=preview,
            database_url=database_url,
        )
    except OperationalError:
        return None
    except (ConnectionError, SQLAlchemyError, OSError, ValueError):
        return None


def build_tier2_or_tier3_context(
    database_url: str,
    data_dir: Optional[Path] = None,
    engine: Optional[Engine] = None,
) -> DataSourcingContext:
    """第二级内存 SQLite；必要时第三级 Mock 自愈。"""
    mem_engine = engine or create_memory_sqlite_engine()
    sub_tier = bootstrap_memory_sqlite_database(mem_engine, data_dir=data_dir)
    preview = fetch_sql_preview_sample(mem_engine)

    if sub_tier == 3:
        return DataSourcingContext(
            tier=3,
            postgres_connected=False,
            engine=mem_engine,
            sql_preview=preview,
            database_url=database_url,
        )
    return DataSourcingContext(
        tier=2,
        postgres_connected=False,
        engine=mem_engine,
        sql_preview=preview,
        database_url=database_url,
    )


def resolve_three_tier_data_sourcing(
    config: ExperimentConfig,
    *,
    reuse_session: bool = True,
) -> DataSourcingContext:
    """
    智能三级降级抽水：PG → 内存 SQLite → Mock 自愈。
    """
    database_url = os.getenv("MEL_DATABASE_URL", config.database_url or PRODUCTION_DATABASE_URL)

    if reuse_session and SESSION_TIER_KEY in st.session_state:
        cached: DataSourcingContext = st.session_state[SESSION_TIER_KEY]
        if cached.engine is not None:
            try:
                cached.sql_preview = fetch_sql_preview_sample(cached.engine)
            except Exception:  # noqa: BLE001
                pass
            return cached

    ctx = try_tier1_postgresql(database_url)
    if ctx is None:
        mem_engine = st.session_state.get(SESSION_ENGINE_KEY)
        ctx = build_tier2_or_tier3_context(
            database_url,
            data_dir=config.data_dir,
            engine=mem_engine if isinstance(mem_engine, Engine) else None,
        )

    st.session_state[SESSION_TIER_KEY] = ctx
    st.session_state[SESSION_ENGINE_KEY] = ctx.engine
    return ctx


def load_feature_wide_table_with_fallback(
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, DataSourcingContext]:
    """
    三级容灾抽水后，将 SQL 宽表映射为 LightGBM 建模矩阵。
    """
    ctx = resolve_three_tier_data_sourcing(config, reuse_session=True)
    assert ctx.engine is not None

    df_sql = fetch_sql_pipeline_wide(ctx.engine)
    df_wide = adapt_sql_pipeline_to_model_frame(df_sql)
    df_wide = _robust_fill_enhanced_env_features(df_wide)
    required = [COL_TARGET] + list(BENCHMARK_FEATURE_COLUMNS) + list(ENHANCED_ENV_COLUMNS)
    df_wide = ensure_sufficient_wide_table(config, df_wide, required)
    return df_wide, ctx


def build_config_from_sidebar(
    lang: str,
) -> tuple[ExperimentConfig, bool, bool, bool, bool]:
    """
    从侧边栏读取用户配置。

    Returns
    -------
    (config, run_clicked, use_pipeline, pipeline_refresh, use_postgres)
    """
    st.sidebar.header(t(lang, "sidebar_console"))
    mode_labels = [t(lang, f"data_mode_{key}") for key in _DATA_MODE_KEYS]
    data_mode_label = st.sidebar.radio(
        t(lang, "data_mode"),
        options=mode_labels,
        index=0,
        help=t(lang, "data_mode_help"),
    )
    data_mode_key = _DATA_MODE_KEYS[mode_labels.index(data_mode_label)]

    st.sidebar.text_input(
        t(lang, "pg_url_label"),
        value=os.getenv("MEL_DATABASE_URL", DEFAULT_DATABASE_URL),
        disabled=(data_mode_key != "postgres"),
        help=t(lang, "pg_url_help"),
        key="pg_database_url",
    )
    pipeline_refresh = st.sidebar.checkbox(
        t(lang, "pipeline_refresh", hours=PIPELINE_LOOKBACK_HOURS),
        value=True,
        disabled=(data_mode_key != "sqlite"),
    )
    if data_mode_key == "sqlite":
        st.sidebar.info(t(lang, "pipeline_hint"))
    lookback_hours = st.sidebar.number_input(
        t(lang, "lookback_hours"),
        min_value=EXPERIMENT_LOOKBACK_HOURS_MIN,
        max_value=8760,
        value=EXPERIMENT_LOOKBACK_HOURS_DEFAULT,
        step=24,
        help=t(
            lang,
            "lookback_help",
            min_h=EXPERIMENT_LOOKBACK_HOURS_MIN,
            default_h=EXPERIMENT_LOOKBACK_HOURS_DEFAULT,
        ),
    )
    train_ratio = st.sidebar.slider(
        t(lang, "train_ratio"),
        min_value=0.60,
        max_value=0.90,
        value=0.80,
        step=0.05,
        help=t(lang, "train_ratio_help"),
    )
    data_dir_str = st.sidebar.text_input(
        t(lang, "data_dir"),
        value=str(_APP_ROOT / "data"),
        disabled=(data_mode_key != "csv"),
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(t(lang, "storage_params"))
    power_mw = st.sidebar.number_input(t(lang, "power_mw"), value=50.0, min_value=1.0)
    capacity_mwh = st.sidebar.number_input(
        t(lang, "capacity_mwh"), value=200.0, min_value=1.0
    )

    run_clicked = st.sidebar.button(
        t(lang, "run_button"),
        type="primary",
        use_container_width=True,
    )

    demo = data_mode_key == "demo"
    use_pipeline = data_mode_key == "sqlite"
    use_postgres = data_mode_key == "postgres"
    data_dir = Path(data_dir_str) if data_mode_key in ("csv", "postgres") else None
    pg_url = st.session_state.get("pg_database_url", PRODUCTION_DATABASE_URL)
    config = ExperimentConfig(
        demo=demo,
        production_db=use_postgres,
        database_url=pg_url,
        data_dir=data_dir,
        lookback_hours=int(lookback_hours),
        train_ratio=train_ratio,
        storage_power_mw=power_mw,
        storage_capacity_mwh=capacity_mwh,
    )
    return config, run_clicked, use_pipeline, pipeline_refresh, use_postgres


def load_tables_from_pipeline(
    run_update: bool,
    lookback_hours: int = PIPELINE_LOOKBACK_HOURS,
    *,
    lang: str = "zh",
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
            st.warning(t(lang, "pipeline_empty_warn", hours=lookback_hours))
            pipeline.run_daily_update()
    return pipeline.load_raw_tables_from_db()


def run_experiment_with_status(
    config: ExperimentConfig,
    pipeline_tables: Optional[dict] = None,
    df_wide_preloaded: Optional[pd.DataFrame] = None,
    *,
    lang: str = "zh",
) -> ExperimentReport:
    """
    分步骤执行实验 pipeline，配合 ``st.status`` 展示进度动效。

    逻辑与 ``run_experiment.run_experiment`` 一致，但拆步便于前端反馈。
    """
    benchmark_cols = list(BENCHMARK_FEATURE_COLUMNS)
    enhanced_cols = benchmark_cols + list(ENHANCED_ENV_COLUMNS)
    required_cols = [COL_TARGET] + enhanced_cols

    with st.status(t(lang, "status_running"), expanded=True) as status:
        # --- 步骤 1：特征宽表（PostgreSQL SQL / 容灾 CSV / Demo / 数据中台）---
        if df_wide_preloaded is not None:
            st.write(t(lang, "step_sql_wide"))
            df_wide = ensure_sufficient_wide_table(config, df_wide_preloaded, required_cols)
        elif pipeline_tables is not None:
            st.write(t(lang, "step_align"))
            tables = enrich_market_physics_inputs(pipeline_tables, schema=config.schema)
            df_wide = build_enhanced_wide_table(tables, schema=config.schema)
            st.write(t(lang, "step_physics"))
            df_wide = merge_frontier_physics_features(df_wide, schema=config.schema)
            df_wide = add_model_ready_features(df_wide)
            df_wide = _robust_fill_enhanced_env_features(df_wide)
        elif config.demo:
            st.write(
                t(lang, "step_demo", hours=effective_lookback_hours(config))
            )
            df_wide = build_full_feature_wide_table(config)
        else:
            if config.data_dir is None or not config.data_dir.is_dir():
                raise FileNotFoundError(
                    t(lang, "dir_missing", path=config.data_dir)
                )
            tables = load_raw_tables(config.data_dir)
            tables = enrich_market_physics_inputs(tables, schema=config.schema)
            df_wide = build_enhanced_wide_table(tables, schema=config.schema)
            df_wide = merge_frontier_physics_features(df_wide, schema=config.schema)
            df_wide = add_model_ready_features(df_wide)
            df_wide = _robust_fill_enhanced_env_features(df_wide)

        st.session_state["frontier_physics_audit"] = audit_frontier_physics_features(df_wide)

        missing = [c for c in required_cols if c not in df_wide.columns]
        if missing:
            raise ValueError(t(lang, "missing_cols", cols=missing))

        feature_frame = df_wide[required_cols + [COL_TIMESTAMP]].copy()
        feature_frame = feature_frame.dropna(subset=required_cols).reset_index(drop=True)
        train_df, test_df = temporal_train_test_split(feature_frame, config.train_ratio)
        st.write(
            t(
                lang,
                "sample_stats",
                total=len(feature_frame),
                train=len(train_df),
                test=len(test_df),
            )
        )
        if len(train_df) < config.min_train_samples:
            raise ValueError(
                t(
                    lang,
                    "train_too_small",
                    n=len(train_df),
                    min=config.min_train_samples,
                    hours=config.lookback_hours,
                )
            )

        # --- 步骤 2：基准模型 ---
        st.write(t(lang, "step_bench"))
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
        st.write(t(lang, "step_enhanced"))
        X_tr_e = X_train_e.iloc[:-valid_size]
        y_tr_e = y_train.iloc[:-valid_size]
        X_va_e = X_train_e.iloc[-valid_size:]
        y_va_e = y_train.iloc[-valid_size:]

        model_enh = train_quantile_lgbm(X_tr_e, y_tr_e, X_va_e, y_va_e)
        pred_enh = model_enh.predict(X_test_e)

        metrics_bench = evaluate_regression(y_test.values, pred_bench)
        metrics_enh = evaluate_regression(y_test.values, pred_enh)

        # --- 步骤 4：储能套利 ---
        st.write(t(lang, "step_storage"))
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

        status.update(label=t(lang, "status_done"), state="complete", expanded=False)

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
        model_enhanced=model_enh,
        X_test_enhanced=X_test_e.copy(),
    )


def render_metric_cards(report: ExperimentReport, lang: str) -> None:
    """大字高光指标卡片：精度 + 经济价值。"""
    bm, em = report.benchmark_metrics, report.enhanced_metrics
    rev_alpha = _pct_alpha(report.benchmark_revenue_yuan, report.enhanced_revenue_yuan)
    alpha_abs = report.enhanced_revenue_yuan - report.benchmark_revenue_yuan

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(t(lang, "rmse_bench"), f"{bm.rmse:.2f}", help=t(lang, "rmse_help"))
    c2.metric(t(lang, "rmse_enh"), f"{em.rmse:.2f}", delta=f"{em.rmse - bm.rmse:+.2f}")
    c3.metric(t(lang, "mae_enh"), f"{em.mae:.2f}")
    c4.metric(t(lang, "r2_enh"), f"{em.r2:.4f}")
    c5.metric(
        t(lang, "storage_enh"),
        f"¥{report.enhanced_revenue_yuan:,.0f}",
        delta=f"+¥{alpha_abs:,.0f}",
    )

    st.markdown(
        f'<p class="alpha-highlight">{t(lang, "alpha_highlight", pct=rev_alpha, bench=report.benchmark_revenue_yuan)}</p>',
        unsafe_allow_html=True,
    )


def render_comparison_table(report: ExperimentReport, lang: str) -> None:
    """HTML 对比表：统计精度 + 经济收益二维矩阵。"""
    bm, em = report.benchmark_metrics, report.enhanced_metrics
    rmse_chg = _pct_improvement(bm.rmse, em.rmse)
    mae_chg = _pct_improvement(bm.mae, em.mae)
    rev_alpha = _pct_alpha(report.benchmark_revenue_yuan, report.enhanced_revenue_yuan)

    table_html = f"""
    <table style="width:100%; border-collapse:collapse; font-size:0.95rem;">
      <thead>
        <tr style="background:#1e3a5f; color:#fff;">
          <th style="padding:10px; text-align:left;">{t(lang, "table_dim")}</th>
          <th style="padding:10px;">{t(lang, "table_metric")}</th>
          <th style="padding:10px;">{t(lang, "table_bench")}</th>
          <th style="padding:10px;">{t(lang, "table_enh")}</th>
          <th style="padding:10px;">{t(lang, "table_change")}</th>
        </tr>
      </thead>
      <tbody>
        <tr style="background:#f1f5f9;">
          <td rowspan="3" style="padding:10px; font-weight:600;">{t(lang, "table_stat_acc")}</td>
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
          <td style="padding:10px; font-weight:600;">{t(lang, "table_econ")}</td>
          <td style="padding:10px;">{t(lang, "table_storage_profit")}</td>
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
      {t(lang, "table_footer", train=report.n_train, test=report.n_test)}
    </p>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def backtest_hourly_pnl_series(
    predicted_prices: np.ndarray,
    settlement_prices: np.ndarray,
    config: ExperimentConfig,
    horizon_hours: int = 24,
) -> np.ndarray:
    """逐小时储能套利 PnL（与 ``backtest_storage_revenue_yuan`` 同逻辑）。"""
    backend, optimize_fn, hourly_pnl_fn = _import_optimizer()
    pred = np.asarray(predicted_prices, dtype=float)
    settle = np.asarray(settlement_prices, dtype=float)
    power_mw = config.storage_power_mw
    capacity_mwh = config.storage_capacity_mwh
    soc_min = capacity_mwh * STORAGE_MIN_SOC_RATIO
    soc_init: Optional[float] = (soc_min + capacity_mwh) / 2.0
    pnls: list[float] = []

    for start in range(0, len(pred), horizon_hours):
        end = min(start + horizon_hours, len(pred))
        pred_h = pred[start:end]
        settle_h = settle[start:end]
        if len(pred_h) < 2:
            continue

        if backend == "power_quant":
            opt = optimize_fn(
                pred_h,
                pred_h,
                pred_h,
                power_mw,
                capacity_mwh,
                risk_mode="balanced",
                eta=STORAGE_ETA,
                cycle_degradation_cost=STORAGE_CYCLE_COST,
                min_soc_ratio=STORAGE_MIN_SOC_RATIO,
                soc_init=soc_init,
            )
        else:
            opt = optimize_fn(
                pred_h,
                power_mw,
                capacity_mwh,
                eta=STORAGE_ETA,
                cycle_degradation_cost=STORAGE_CYCLE_COST,
                min_soc_ratio=STORAGE_MIN_SOC_RATIO,
            )

        power = np.asarray(opt["power"], dtype=float)
        for t in range(len(settle_h)):
            pnls.append(hourly_pnl_fn(power[t], settle_h[t], STORAGE_CYCLE_COST))

        if "soc" in opt and len(opt["soc"]) > 0:
            soc_init = float(np.clip(opt["soc"][-1], soc_min, capacity_mwh))

    return np.asarray(pnls, dtype=np.float64)


def build_arena_pnl_dataframe(
    report: ExperimentReport,
    config: ExperimentConfig,
    ppo_hourly: np.ndarray,
) -> pd.DataFrame:
    """合成三策略累计 PnL 对比宽表。"""
    if report.test_forecast is None:
        raise ValueError("test_forecast missing")

    tf = report.test_forecast.sort_values(COL_TIMESTAMP).reset_index(drop=True)
    settle = tf["actual"].to_numpy(dtype=float)
    bench_pred = tf["benchmark_pred"].to_numpy(dtype=float)
    enh_pred = tf["enhanced_pred"].to_numpy(dtype=float)

    pnl_bench = backtest_hourly_pnl_series(bench_pred, settle, config)
    pnl_lgbm = backtest_hourly_pnl_series(enh_pred, settle, config)
    n = min(len(pnl_bench), len(pnl_lgbm), len(ppo_hourly), len(tf))

    return pd.DataFrame(
        {
            COL_TIMESTAMP: tf[COL_TIMESTAMP].iloc[:n],
            "pnl_bench": pnl_bench[:n],
            "pnl_lgbm": pnl_lgbm[:n],
            "pnl_ppo": ppo_hourly[:n],
            "cum_bench": np.cumsum(pnl_bench[:n]),
            "cum_lgbm": np.cumsum(pnl_lgbm[:n]),
            "cum_ppo": np.cumsum(ppo_hourly[:n]),
        }
    )


def run_ppo_for_web(
    config: ExperimentConfig,
    *,
    epochs: int = PPO_WEB_EPOCHS,
    use_postgres: bool = True,
) -> tuple[Any, Any, np.ndarray]:
    """
    Web 端轻量 PPO：优先加载预训练权重，否则快速训练若干 Epoch。
    """
    # macOS：缓解 LightGBM(libomp) × PyTorch(MKL) 同进程导致的 SIGSEGV (139)
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        from ppo_agent import (
            BacktestReport,
            PPOAgent,
            PPOHyperParams,
            build_vpp_data_bundle,
            run_policy_on_env_with_trace,
            train_ppo,
        )
        from vpp_environment import VPPEnvironment, VPP_OBS_COLUMNS
    except ImportError as exc:
        raise ImportError("ppo_agent / vpp_environment") from exc

    ppo_config = ExperimentConfig(
        demo=config.demo,
        production_db=use_postgres and not config.demo,
        database_url=config.database_url,
        lookback_hours=config.lookback_hours,
        train_ratio=config.train_ratio,
        storage_power_mw=config.storage_power_mw,
        storage_capacity_mwh=config.storage_capacity_mwh,
        node_id=config.node_id,
    )

    bundle = build_vpp_data_bundle(ppo_config)
    test_env = VPPEnvironment(
        config=ppo_config,
        split="test",
        power_mw=ppo_config.storage_power_mw,
        capacity_mwh=ppo_config.storage_capacity_mwh,
        data_bundle=bundle,
    )
    obs_dim = len(VPP_OBS_COLUMNS)
    hp = PPOHyperParams(
        total_epochs=epochs,
        rollout_steps=PPO_WEB_ROLLOUT,
        eval_interval=max(epochs + 1, 999),
    )
    if PPO_POLICY_PATH.is_file():
        agent = PPOAgent(obs_dim, device="cpu", hparams=hp)
        agent.load(PPO_POLICY_PATH)
    else:
        agent, _ = train_ppo(
            ppo_config,
            hparams=hp,
            device="cpu",
            save_path=PPO_POLICY_PATH,
            verbose=False,
        )

    ppo_report, hourly = run_policy_on_env_with_trace(agent, test_env, deterministic=True)
    return agent, ppo_report, hourly


def plot_cumulative_pnl_showdown(arena_df: pd.DataFrame, *, lang: str) -> go.Figure:
    """三资产累计收益率曲线对比图。"""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=arena_df[COL_TIMESTAMP],
            y=arena_df["cum_bench"],
            name=t(lang, "curve_bench"),
            line=dict(color="#94a3b8", width=2.2, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=arena_df[COL_TIMESTAMP],
            y=arena_df["cum_lgbm"],
            name=t(lang, "curve_lgbm"),
            line=dict(color="#2563eb", width=2.6),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=arena_df[COL_TIMESTAMP],
            y=arena_df["cum_ppo"],
            name=t(lang, "curve_ppo"),
            line=dict(color="#7c3aed", width=3.0),
        )
    )
    fig.update_layout(
        title=t(lang, "chart_cum_pnl_title"),
        xaxis_title=t(lang, "chart_time"),
        yaxis_title=t(lang, "chart_cum_pnl_y"),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=520,
    )
    return fig


def render_mdp_expander(lang: str, config: ExperimentConfig) -> None:
    """MDP 状态空间与奖励设计（中英双语）。"""
    try:
        from vpp_environment import VPP_OBS_COLUMNS
    except ImportError:
        VPP_OBS_COLUMNS = ("features", "soc_ratio")  # type: ignore[assignment]

    with st.expander(t(lang, "mdp_expander_title")):
        st.markdown(t(lang, "mdp_state_title"))
        st.markdown(
            t(lang, "mdp_state_body", obs_dim=len(VPP_OBS_COLUMNS)),
        )
        st.markdown(t(lang, "mdp_reward_title"))
        st.markdown(
            t(
                lang,
                "mdp_reward_body",
                deg_cost=STORAGE_CYCLE_COST,
            ),
        )
        st.caption(
            f"P_max={config.storage_power_mw:.0f} MW · "
            f"E_max={config.storage_capacity_mwh:.0f} MWh · "
            f"η={STORAGE_ETA:.2f} · SOC_min={STORAGE_MIN_SOC_RATIO:.0%}"
        )


def render_three_model_metrics(
    report: ExperimentReport,
    lang: str,
    ppo_report: Optional[Any] = None,
) -> None:
    """三模型竞技场指标高光卡片。"""
    bm, em = report.benchmark_metrics, report.enhanced_metrics
    rev_alpha = _pct_alpha(report.benchmark_revenue_yuan, report.enhanced_revenue_yuan)
    alpha_abs = report.enhanced_revenue_yuan - report.benchmark_revenue_yuan

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(t(lang, "rmse_bench"), f"{bm.rmse:.2f}", help=t(lang, "rmse_help"))
    c2.metric(t(lang, "rmse_enh"), f"{em.rmse:.2f}", delta=f"{em.rmse - bm.rmse:+.2f}")
    c3.metric(t(lang, "mae_enh"), f"{em.mae:.2f}")
    c4.metric(t(lang, "r2_enh"), f"{em.r2:.4f}")

    s1, s2, s3 = st.columns(3)
    s1.metric(
        t(lang, "storage_bench"),
        f"¥{report.benchmark_revenue_yuan:,.0f}",
    )
    s2.metric(
        t(lang, "storage_lgbm"),
        f"¥{report.enhanced_revenue_yuan:,.0f}",
        delta=f"+¥{alpha_abs:,.0f}",
    )
    if ppo_report is not None:
        ppo_rev = float(ppo_report.total_revenue_yuan)
        s3.metric(
            t(lang, "storage_ppo"),
            f"¥{ppo_rev:,.0f}",
            delta=f"{ppo_rev - report.enhanced_revenue_yuan:+,.0f}",
        )
        ppo_alpha = _pct_alpha(report.enhanced_revenue_yuan, ppo_rev)
        st.markdown(
            f'<p class="alpha-highlight">{t(lang, "alpha_highlight", pct=rev_alpha, bench=report.benchmark_revenue_yuan)}</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p class="alpha-highlight" style="color:#7c3aed;">{t(lang, "alpha_three_way", pct=ppo_alpha, ppo=ppo_rev, lgbm=report.enhanced_revenue_yuan)}</p>',
            unsafe_allow_html=True,
        )
        p1, p2, p3 = st.columns(3)
        p1.metric(
            t(lang, "profitable_hours"),
            f"{ppo_report.profitable_hour_ratio:.1%}",
        )
        p2.metric(
            t(lang, "beat_idle_hours"),
            f"{ppo_report.beat_idle_hour_ratio:.1%}",
        )
        p3.metric(
            t(lang, "alpha_vs_idle"),
            f"{ppo_report.alpha_vs_idle_pct:+.2f}%",
        )
    else:
        s3.metric(t(lang, "storage_ppo"), "—")
        st.markdown(
            f'<p class="alpha-highlight">{t(lang, "alpha_highlight", pct=rev_alpha, bench=report.benchmark_revenue_yuan)}</p>',
            unsafe_allow_html=True,
        )


def render_three_model_table(
    report: ExperimentReport,
    lang: str,
    ppo_report: Optional[Any] = None,
) -> None:
    """三模型 HTML 对比表。"""
    bm, em = report.benchmark_metrics, report.enhanced_metrics
    rmse_chg = _pct_improvement(bm.rmse, em.rmse)
    mae_chg = _pct_improvement(bm.mae, em.mae)
    rev_alpha_ml = _pct_alpha(report.benchmark_revenue_yuan, report.enhanced_revenue_yuan)

    ppo_rev_cell = "—"
    ppo_chg_cell = "—"
    if ppo_report is not None:
        ppo_rev = float(ppo_report.total_revenue_yuan)
        ppo_rev_cell = f"{ppo_rev:,.2f}"
        ppo_chg_cell = f"{_pct_alpha(report.enhanced_revenue_yuan, ppo_rev):+.2f}%"

    table_html = f"""
    <table style="width:100%; border-collapse:collapse; font-size:0.95rem;">
      <thead>
        <tr style="background:#1e3a5f; color:#fff;">
          <th style="padding:10px; text-align:left;">{t(lang, "table_dim")}</th>
          <th style="padding:10px;">{t(lang, "table_metric")}</th>
          <th style="padding:10px;">{t(lang, "table_bench")}</th>
          <th style="padding:10px;">{t(lang, "table_lgbm")}</th>
          <th style="padding:10px;">{t(lang, "table_ppo")}</th>
          <th style="padding:10px;">{t(lang, "table_change")}</th>
        </tr>
      </thead>
      <tbody>
        <tr style="background:#f1f5f9;">
          <td rowspan="3" style="padding:10px; font-weight:600;">{t(lang, "table_stat_acc")}</td>
          <td style="padding:10px;">RMSE</td>
          <td style="padding:10px; text-align:center;">{bm.rmse:.4f}</td>
          <td style="padding:10px; text-align:center;">{em.rmse:.4f}</td>
          <td style="padding:10px; text-align:center;">—</td>
          <td style="padding:10px; text-align:center;">{rmse_chg:+.2f}%</td>
        </tr>
        <tr style="background:#fff;">
          <td style="padding:10px;">MAE</td>
          <td style="padding:10px; text-align:center;">{bm.mae:.4f}</td>
          <td style="padding:10px; text-align:center;">{em.mae:.4f}</td>
          <td style="padding:10px; text-align:center;">—</td>
          <td style="padding:10px; text-align:center;">{mae_chg:+.2f}%</td>
        </tr>
        <tr style="background:#f1f5f9;">
          <td style="padding:10px;">R²</td>
          <td style="padding:10px; text-align:center;">{bm.r2:.4f}</td>
          <td style="padding:10px; text-align:center;">{em.r2:.4f}</td>
          <td style="padding:10px; text-align:center;">—</td>
          <td style="padding:10px; text-align:center;">—</td>
        </tr>
        <tr style="background:#ecfdf5;">
          <td style="padding:10px; font-weight:600;">{t(lang, "table_econ")}</td>
          <td style="padding:10px;">{t(lang, "table_storage_profit")}</td>
          <td style="padding:10px; text-align:center;">{report.benchmark_revenue_yuan:,.2f}</td>
          <td style="padding:10px; text-align:center; font-weight:700;">
            {report.enhanced_revenue_yuan:,.2f}
          </td>
          <td style="padding:10px; text-align:center; font-weight:700; color:#7c3aed;">
            {ppo_rev_cell}
          </td>
          <td style="padding:10px; text-align:center; color:#16a34a; font-weight:800;">
            ML {rev_alpha_ml:+.2f}% · PPO {ppo_chg_cell}
          </td>
        </tr>
      </tbody>
    </table>
    <p style="color:#64748b; font-size:0.85rem; margin-top:8px;">
      {t(lang, "table_footer", train=report.n_train, test=report.n_test)}
    </p>
    """
    st.markdown(table_html, unsafe_allow_html=True)


def render_three_model_arena(
    report: ExperimentReport,
    config: ExperimentConfig,
    lang: str,
    *,
    use_postgres: bool,
) -> None:
    """三模型竞技场：状态锁驱动的 PPO 触发 + 累计 PnL + MDP 文档。"""
    st.markdown(t(lang, "arena_title"))
    st.caption(t(lang, "arena_caption"))

    render_mdp_expander(lang, config)

    # --- 阶段 1：仅锁状态，立即 rerun（避免在同一次 run 内训练后丢失上下文）---
    if st.button(
        t(lang, "ppo_btn"),
        type="secondary",
        use_container_width=True,
        key="ppo_arena_button",
    ):
        st.session_state[SESSION_PPO_EXECUTED] = True
        st.rerun()

    # --- 阶段 2：ppo_executed=True 时持久渲染训练结果 ---
    if st.session_state.get(SESSION_PPO_EXECUTED):
        if st.session_state.get(SESSION_PPO_REPORT) is None:
            try:
                import torch  # noqa: F401
                import gymnasium  # noqa: F401
            except ImportError:
                st.error(t(lang, "ppo_missing_torch"))
                st.session_state[SESSION_PPO_EXECUTED] = False
                return

            with st.spinner(t(lang, "ppo_spinner", epochs=PPO_WEB_EPOCHS)):
                try:
                    _, ppo_report, hourly = run_ppo_for_web(
                        config,
                        epochs=PPO_WEB_EPOCHS,
                        use_postgres=use_postgres,
                    )
                    arena_df = build_arena_pnl_dataframe(report, config, hourly)
                    st.session_state[SESSION_PPO_REPORT] = ppo_report
                    st.session_state[SESSION_ARENA_PNL] = arena_df
                    st.success(t(lang, "ppo_done"))
                except Exception as exc:  # noqa: BLE001
                    st.error(t(lang, "ppo_fail", exc=f"{type(exc).__name__}: {exc}"))
                    st.exception(exc)
                    st.session_state[SESSION_PPO_EXECUTED] = False
                    return

        ppo_report = st.session_state.get(SESSION_PPO_REPORT)
        arena_df = st.session_state.get(SESSION_ARENA_PNL)

        if arena_df is not None and ppo_report is not None:
            st.plotly_chart(
                plot_cumulative_pnl_showdown(arena_df, lang=lang),
                use_container_width=True,
            )

            m1, m2, m3 = st.columns(3)
            m1.metric(
                t(lang, "storage_ppo"),
                f"¥{float(ppo_report.total_revenue_yuan):,.0f}",
            )
            m2.metric(
                t(lang, "profitable_hours"),
                f"{ppo_report.profitable_hour_ratio:.1%}",
            )
            m3.metric(
                t(lang, "alpha_vs_idle"),
                f"{ppo_report.alpha_vs_idle_pct:+.2f}%",
            )

        if st.button(t(lang, "ppo_reset"), key="ppo_reset_button"):
            st.session_state[SESSION_PPO_EXECUTED] = False
            st.session_state.pop(SESSION_PPO_REPORT, None)
            st.session_state.pop(SESSION_ARENA_PNL, None)
            st.rerun()


def render_experiment_dashboard(
    report: ExperimentReport,
    config: ExperimentConfig,
    lang: str,
    *,
    use_postgres: bool,
) -> None:
    """主实验 + 三模型竞技场 + 可视化（Session 持久化渲染）。"""
    st.success(t(lang, "experiment_done"))
    st.markdown(t(lang, "report_board"))

    ppo_report = st.session_state.get(SESSION_PPO_REPORT)
    render_three_model_metrics(report, lang, ppo_report=ppo_report)
    render_three_model_table(report, lang, ppo_report=ppo_report)

    render_three_model_arena(
        report,
        config,
        lang,
        use_postgres=use_postgres,
    )

    st.markdown("---")
    st.markdown(t(lang, "viz_section"))
    render_charts(report, lang)


def plot_price_comparison(
    test_forecast: pd.DataFrame,
    window_hours: int = 72,
    *,
    lang: str = "zh",
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
            name=t(lang, "chart_actual"),
            line=dict(color="#0f172a", width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df[COL_TIMESTAMP],
            y=df["benchmark_pred"],
            name=t(lang, "chart_bench_pred"),
            line=dict(color="#94a3b8", width=1.8, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df[COL_TIMESTAMP],
            y=df["enhanced_pred"],
            name=t(lang, "chart_enh_pred"),
            line=dict(color="#2563eb", width=2),
        )
    )
    fig.update_layout(
        title=t(lang, "chart_price_title", n=len(df)),
        xaxis_title=t(lang, "chart_time"),
        yaxis_title=t(lang, "chart_price_y"),
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
    )
    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 12,
    *,
    lang: str = "zh",
) -> go.Figure:
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
        title=t(lang, "chart_fi_title"),
        xaxis_title=t(lang, "chart_fi_x"),
        yaxis_title="",
        template="plotly_white",
        height=max(400, 28 * len(df)),
    )
    return fig


def compute_frontier_physics_shap(
    model: Any,
    X_test: pd.DataFrame,
    max_samples: int = 280,
) -> Optional[pd.DataFrame]:
    """
    计算四大前沿物理因子板块的 mean |SHAP| 贡献（测试集样本）。

    用于解释增强模型相对基准的微观物理超额贡献。
    """
    try:
        import shap
    except ImportError:
        return None

    if model is None or X_test is None or X_test.empty:
        return None

    sample = X_test
    if len(sample) > max_samples:
        sample = sample.sample(max_samples, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    if isinstance(shap_values, list):
        shap_arr = np.asarray(shap_values[0])
    else:
        shap_arr = np.asarray(shap_values)

    if shap_arr.ndim == 3:
        shap_arr = shap_arr[:, :, 0]

    mean_abs = np.mean(np.abs(shap_arr), axis=0)
    feat_shap = pd.Series(mean_abs, index=sample.columns)

    rows: list[dict[str, object]] = []
    for block, cols in FRONTIER_PHYSICS_BLOCKS.items():
        present = [c for c in cols if c in feat_shap.index]
        if not present:
            rows.append(
                {
                    "physics_block": block,
                    "mean_abs_shap": 0.0,
                    "features": "",
                }
            )
            continue
        rows.append(
            {
                "physics_block": block,
                "mean_abs_shap": float(feat_shap[present].sum()),
                "features": ", ".join(present),
            }
        )

    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False)


def plot_frontier_shap_bar(shap_summary: pd.DataFrame, *, lang: str = "zh") -> go.Figure:
    """前沿物理板块 SHAP 贡献条形图。"""
    df = shap_summary.sort_values("mean_abs_shap", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=df["mean_abs_shap"],
            y=df["physics_block"],
            orientation="h",
            marker_color="#7c3aed",
            text=df["mean_abs_shap"].round(4),
            textposition="outside",
        )
    )
    fig.update_layout(
        title=t(lang, "shap_chart_title"),
        xaxis_title=t(lang, "shap_chart_x"),
        yaxis_title="",
        template="plotly_white",
        height=360,
    )
    return fig


def render_frontier_shap_panel(report: ExperimentReport, lang: str) -> None:
    """在特征重要性旁展示 SHAP 物理解释模块。"""
    st.subheader(t(lang, "shap_title"))
    st.caption(t(lang, "shap_caption"))

    audit = st.session_state.get("frontier_physics_audit")
    if audit:
        cols = st.columns(4)
        for col, (block, ok) in zip(cols, audit.items()):
            col.metric(
                block.split()[0],
                t(lang, "shap_ok") if ok else t(lang, "shap_miss"),
                delta=None,
            )

    if report.model_enhanced is None or report.X_test_enhanced is None:
        st.info(t(lang, "shap_run_first"))
        return

    with st.spinner(t(lang, "shap_spinner")):
        shap_df = compute_frontier_physics_shap(
            report.model_enhanced,
            report.X_test_enhanced,
        )

    if shap_df is None:
        st.warning(t(lang, "shap_missing_lib"))
        return

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.plotly_chart(
            plot_frontier_shap_bar(shap_df, lang=lang),
            use_container_width=True,
        )
    with c2:
        st.dataframe(
            shap_df.style.format({"mean_abs_shap": "{:.5f}"}),
            use_container_width=True,
            hide_index=True,
        )


def render_charts(report: ExperimentReport, lang: str) -> None:
    """渲染 Plotly 图表区。"""
    if report.test_forecast is None or report.enhanced_full_importance is None:
        st.warning(t(lang, "viz_missing"))
        return

    window = st.slider(
        t(lang, "chart_window"),
        min_value=24,
        max_value=min(168, len(report.test_forecast)),
        value=min(72, len(report.test_forecast)),
        step=12,
    )

    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        st.plotly_chart(
            plot_price_comparison(
                report.test_forecast, window_hours=window, lang=lang
            ),
            use_container_width=True,
        )
    with col_r:
        st.plotly_chart(
            plot_feature_importance(
                report.enhanced_full_importance, lang=lang
            ),
            use_container_width=True,
        )

    render_frontier_shap_panel(report, lang)

    st.subheader(t(lang, "env_rank"))
    st.dataframe(
        report.env_importance.style.format({"importance": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    """应用入口。"""
    init_session_state()
    lang = resolve_lang()
    render_banner(lang)
    config, run_clicked, use_pipeline, pipeline_refresh, use_postgres = (
        build_config_from_sidebar(lang)
    )

    # 三级容灾：页面加载即解析（不阻塞），供 Banner / SQL 架构展示
    sourcing_ctx: Optional[DataSourcingContext] = None
    if use_postgres and not config.demo:
        try:
            sourcing_ctx = resolve_three_tier_data_sourcing(config, reuse_session=True)
            render_data_tier_banner(sourcing_ctx, lang)
        except Exception as exc:  # noqa: BLE001
            st.error(t(lang, "data_init_error", exc=f"{type(exc).__name__}: {exc}"))

    # 主区也放置醒目运行按钮（与侧边栏联动）
    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        main_run = st.button(
            t(lang, "run_button"),
            type="primary",
            use_container_width=True,
        )
    with col_info:
        if config.demo:
            mode_label = t(lang, "mode_demo")
        elif use_postgres:
            mode_label = t(lang, "mode_postgres", url=config.database_url)
        elif use_pipeline:
            mode_label = t(lang, "mode_sqlite")
        else:
            mode_label = t(lang, "mode_csv", path=config.data_dir)
        st.info(
            t(lang, "mode_info", mode=mode_label, ratio=config.train_ratio)
        )

    if use_pipeline:
        st.markdown(
            f'<div style="padding:0.75rem 1rem; background:#ecfdf5; border-left:4px solid #16a34a; border-radius:6px; margin-bottom:1rem;">{t(lang, "pipeline_banner")}</div>',
            unsafe_allow_html=True,
        )

    if run_clicked or main_run:
        st.session_state[SESSION_PPO_EXECUTED] = False
        st.session_state.pop(SESSION_PPO_REPORT, None)
        st.session_state.pop(SESSION_ARENA_PNL, None)

        try:
            pipeline_tables = None
            df_wide_preloaded: Optional[pd.DataFrame] = None

            if use_postgres and not config.demo:
                with st.spinner(t(lang, "tier_spinner")):
                    df_wide_preloaded, sourcing_ctx = load_feature_wide_table_with_fallback(
                        config
                    )
                    st.session_state["data_source"] = tier_label_for(
                        lang, sourcing_ctx.tier
                    )
                    render_data_tier_banner(sourcing_ctx, lang)

            if use_pipeline:
                with st.spinner(
                    t(lang, "pipeline_spinner", hours=PIPELINE_LOOKBACK_HOURS)
                ):
                    pipeline_tables = load_tables_from_pipeline(
                        run_update=pipeline_refresh,
                        lookback_hours=PIPELINE_LOOKBACK_HOURS,
                        lang=lang,
                    )
            report = run_experiment_with_status(
                config,
                pipeline_tables=pipeline_tables,
                df_wide_preloaded=df_wide_preloaded,
                lang=lang,
            )
            st.session_state[SESSION_EXPERIMENT_REPORT] = report
            st.session_state[SESSION_EXPERIMENT_RUN] = True
        except Exception as exc:
            st.error(t(lang, "experiment_fail", exc=f"{type(exc).__name__}: {exc}"))
            st.exception(exc)
            return

    if not st.session_state.get(SESSION_EXPERIMENT_RUN):
        st.markdown(f"{t(lang, 'guide_title')}\n{t(lang, 'guide_body')}")
        return

    report = st.session_state[SESSION_EXPERIMENT_REPORT]
    render_experiment_dashboard(
        report,
        config,
        lang,
        use_postgres=use_postgres,
    )


if __name__ == "__main__":
    main()
