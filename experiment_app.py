# -*- coding: utf-8 -*-
"""
experiment_app.py — 环境微特征电力现货预测增强试验舱（Streamlit Web 界面）

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
    COL_TIMESTAMP,
    ENHANCED_ENV_COLUMNS,
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
from run_experiment import COL_TARGET  # noqa: E402
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


@dataclass
class DataSourcingContext:
    """三级容灾抽水结果上下文。"""

    tier: Literal[1, 2, 3]
    tier_label: str
    banner_message: str
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


def render_banner() -> None:
    """顶部专业 Banner。"""
    st.markdown(
        """
        <div class="main-banner">
            <h1>量化前沿：环境微特征（辐射/污染/水温）电力电价预测增强试验舱</h1>
            <p>Quant Frontier Lab · PostgreSQL 窗口算力 × 储能套利经济价值实证</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_tier_banner(ctx: DataSourcingContext) -> None:
    """主界面：按容灾层级展示状态 + SQL 架构 + 抽样数据。"""
    if ctx.tier == 1:
        st.success(ctx.banner_message)
    elif ctx.tier == 2:
        st.info(ctx.banner_message)
    else:
        st.warning(ctx.banner_message)

    st.caption(f"数据管线：**{ctx.tier_label}** · Engine=`{ctx.engine.dialect.name if ctx.engine else 'n/a'}`")

    with st.expander("🔍 查看后端核心 SQL 窗口函数架构 (Advanced SQL Code)"):
        st.caption(
            "视图在库内完成特征工程；`ROWS BETWEEN 24 PRECEDING AND 1 PRECEDING` "
            "严格排除当前行，防止未来信息泄露（No Look-ahead）。"
        )
        st.code(SQL_CREATE_VIEW.strip(), language="sql")
        st.caption("建模拉取语句（Web 试验舱实时查询）")
        st.code(SQL_LOAD_PIPELINE_WIDE.strip(), language="sql")
        st.caption("随机抽样预览（面试官可见窗口函数已生效）")
        if ctx.sql_preview is not None and not ctx.sql_preview.empty:
            st.dataframe(ctx.sql_preview, use_container_width=True, hide_index=True)
        else:
            st.info("预览数据为空，请点击「启动双模型打擂台」触发自愈灌库。")


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
            tier_label="Tier-1 · PostgreSQL 生产库",
            banner_message=(
                "⚡ 生产级 PostgreSQL 数据库连接成功！已激活 TimescaleDB/PG 混合算力引擎"
            ),
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
            tier_label="Tier-3 · 内存沙盒 + Mock 自愈灌库",
            banner_message=(
                f"🛡️ Streamlit Cloud 容灾：已启用内存 SQLite 沙盒，并动态 Mock {MOCK_SEED_HOURS}h 仿真时序 "
                "（含四大物理因子 + 窗口函数视图）。"
            ),
            postgres_connected=False,
            engine=mem_engine,
            sql_preview=preview,
            database_url=database_url,
        )
    return DataSourcingContext(
        tier=2,
        tier_label="Tier-2 · 云端内存 SQLite 沙盒",
        banner_message=(
            "☁️ PostgreSQL 不可用（云端环境）：已降级至内存 SQLite 高仿真沙盒，"
            "SQL 窗口视图已激活。"
        ),
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


def build_config_from_sidebar() -> tuple[ExperimentConfig, bool, bool, bool]:
    """
    从侧边栏读取用户配置。

    Returns
    -------
    (config, run_clicked) : 实验配置与是否点击了运行按钮
    """
    st.sidebar.header("试验控制台")
    data_mode = st.sidebar.radio(
        "数据模式",
        options=[
            "PostgreSQL 生产库 (推荐)",
            "Demo 模拟数据",
            "智能数据中台 (SQLite)",
            "真实数据目录",
        ],
        index=0,
        help="推荐：PostgreSQL 窗口特征视图；失败自动降级 CSV。",
    )
    st.sidebar.text_input(
        "PostgreSQL URL",
        value=os.getenv("MEL_DATABASE_URL", DEFAULT_DATABASE_URL),
        disabled=(data_mode != "PostgreSQL 生产库 (推荐)"),
        help="可通过环境变量 MEL_DATABASE_URL 覆盖。",
        key="pg_database_url",
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
    lookback_hours = st.sidebar.number_input(
        "历史回溯小时数 (lookback_hours)",
        min_value=EXPERIMENT_LOOKBACK_HOURS_MIN,
        max_value=8760,
        value=EXPERIMENT_LOOKBACK_HOURS_DEFAULT,
        step=24,
        help=f"不低于 {EXPERIMENT_LOOKBACK_HOURS_MIN}h（30天）；默认 {EXPERIMENT_LOOKBACK_HOURS_DEFAULT}h 保障训练样本 ≥1000。",
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
    use_postgres = data_mode == "PostgreSQL 生产库 (推荐)"
    data_dir = Path(data_dir_str) if data_mode in (
        "真实数据目录",
        "PostgreSQL 生产库 (推荐)",
    ) else None
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
    df_wide_preloaded: Optional[pd.DataFrame] = None,
) -> ExperimentReport:
    """
    分步骤执行实验 pipeline，配合 ``st.status`` 展示进度动效。

    逻辑与 ``run_experiment.run_experiment`` 一致，但拆步便于前端反馈。
    """
    benchmark_cols = list(BENCHMARK_FEATURE_COLUMNS)
    enhanced_cols = benchmark_cols + list(ENHANCED_ENV_COLUMNS)
    required_cols = [COL_TARGET] + enhanced_cols

    with st.status("实验引擎运行中...", expanded=True) as status:
        # --- 步骤 1：特征宽表（PostgreSQL SQL / 容灾 CSV / Demo / 数据中台）---
        if df_wide_preloaded is not None:
            st.write("正在消费 SQL 窗口特征宽表（v_features_pipeline_ready）...")
            df_wide = ensure_sufficient_wide_table(config, df_wide_preloaded, required_cols)
        elif pipeline_tables is not None:
            st.write("正在调用特征中台对齐空间辐射数据...")
            tables = enrich_market_physics_inputs(pipeline_tables, schema=config.schema)
            df_wide = build_enhanced_wide_table(tables, schema=config.schema)
            st.write("正在计算四大前沿物理微特征（酷热/风切变/反照率/积尘）...")
            df_wide = merge_frontier_physics_features(df_wide, schema=config.schema)
            df_wide = add_model_ready_features(df_wide)
            df_wide = _robust_fill_enhanced_env_features(df_wide)
        elif config.demo:
            st.write(
                f"正在生成 Demo 大样本宽表（{effective_lookback_hours(config)}h）..."
            )
            df_wide = build_full_feature_wide_table(config)
        else:
            if config.data_dir is None or not config.data_dir.is_dir():
                raise FileNotFoundError(
                    f"真实数据目录不存在: {config.data_dir}。"
                    "请创建 data/ 并放入 market.csv 等文件，或切换 PostgreSQL / Demo 模式。"
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
            raise ValueError(f"宽表缺少列: {missing}")

        feature_frame = df_wide[required_cols + [COL_TIMESTAMP]].copy()
        feature_frame = feature_frame.dropna(subset=required_cols).reset_index(drop=True)
        train_df, test_df = temporal_train_test_split(feature_frame, config.train_ratio)
        st.write(
            f"建模样本：全量 {len(feature_frame)} 行 · 训练 {len(train_df)} · 测试 {len(test_df)}"
        )
        if len(train_df) < config.min_train_samples:
            raise ValueError(
                f"训练集仅 {len(train_df)} 条（目标 ≥{config.min_train_samples}）。"
                f"请增大侧边栏 lookback_hours（当前 {config.lookback_hours}）。"
            )

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
        model_enhanced=model_enh,
        X_test_enhanced=X_test_e.copy(),
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


def plot_frontier_shap_bar(shap_summary: pd.DataFrame) -> go.Figure:
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
        title="四大前沿物理因子 · 平均 |SHAP| 超额贡献",
        xaxis_title="Mean |SHAP| (测试集)",
        yaxis_title="",
        template="plotly_white",
        height=360,
    )
    return fig


def render_frontier_shap_panel(report: ExperimentReport) -> None:
    """在特征重要性旁展示 SHAP 物理解释模块。"""
    st.subheader("前沿物理因子 · SHAP 实时解释")
    st.caption(
        "基于增强模型在测试集上的 TreeSHAP，聚合四大微观物理板块的 |SHAP| 强度，"
        "量化其在打擂台中的超额边际贡献（非因果效应，仅供业务解读）。"
    )

    audit = st.session_state.get("frontier_physics_audit")
    if audit:
        cols = st.columns(4)
        for col, (block, ok) in zip(cols, audit.items(), strict=False):
            col.metric(block.split()[0], "已并入" if ok else "缺失", delta=None)

    if report.model_enhanced is None or report.X_test_enhanced is None:
        st.info("请先完成一次擂台实验以生成 SHAP 解释。")
        return

    with st.spinner("正在计算 TreeSHAP（前沿物理板块）..."):
        shap_df = compute_frontier_physics_shap(
            report.model_enhanced,
            report.X_test_enhanced,
        )

    if shap_df is None:
        st.warning("未安装 ``shap`` 库。请执行: pip install shap")
        return

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.plotly_chart(plot_frontier_shap_bar(shap_df), use_container_width=True)
    with c2:
        st.dataframe(
            shap_df.style.format({"mean_abs_shap": "{:.5f}"}),
            use_container_width=True,
            hide_index=True,
        )


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

    render_frontier_shap_panel(report)

    st.subheader("环境因子专项排名")
    st.dataframe(
        report.env_importance.style.format({"importance": "{:.2f}"}),
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    """应用入口。"""
    render_banner()
    config, run_clicked, use_pipeline, pipeline_refresh, use_postgres = (
        build_config_from_sidebar()
    )

    # 三级容灾：页面加载即解析（不阻塞），供 Banner / SQL 架构展示
    sourcing_ctx: Optional[DataSourcingContext] = None
    if use_postgres and not config.demo:
        try:
            sourcing_ctx = resolve_three_tier_data_sourcing(config, reuse_session=True)
            render_data_tier_banner(sourcing_ctx)
        except Exception as exc:  # noqa: BLE001
            st.error(f"数据管线初始化失败：{type(exc).__name__}: {exc}")

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
        elif use_postgres:
            mode_label = f"PostgreSQL 生产库 · {config.database_url}"
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
            1. 默认 **三级容灾抽水**：PostgreSQL → 内存 SQLite → Mock 自愈（适配 Streamlit Cloud）。  
            2. 展开 **SQL 窗口函数架构** 可查看完整 ``LAG`` / ``AVG OVER`` 代码与抽样数据。  
            3. 调整 **时序划分比例** 后点击 **启动双模型打擂台**。  
            """
        )
        return

    try:
        pipeline_tables = None
        df_wide_preloaded: Optional[pd.DataFrame] = None

        if use_postgres and not config.demo:
            with st.spinner("三级容灾抽水：PG → 内存 SQLite → Mock 自愈..."):
                df_wide_preloaded, sourcing_ctx = load_feature_wide_table_with_fallback(config)
                st.session_state["data_source"] = sourcing_ctx.tier_label
                render_data_tier_banner(sourcing_ctx)

        if use_pipeline:
            with st.spinner(
                f"正在执行 {PIPELINE_LOOKBACK_HOURS}h（30天）数据中台日更并加载样本..."
            ):
                pipeline_tables = load_tables_from_pipeline(
                    run_update=pipeline_refresh,
                    lookback_hours=PIPELINE_LOOKBACK_HOURS,
                )
        report = run_experiment_with_status(
            config,
            pipeline_tables=pipeline_tables,
            df_wide_preloaded=df_wide_preloaded,
        )
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
