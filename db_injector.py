# -*- coding: utf-8 -*-
"""
db_injector.py — PostgreSQL 海量时序数据清洗与高速灌入器

使用 SQLAlchemy 连接本地 PostgreSQL，自动建表后通过 Pandas ``to_sql``
（chunksize=5000）批量写入维度表与时序事实表。

用法
----
    python db_injector.py
    python db_injector.py --csv data/features_ready.csv
    python db_injector.py --database-url postgresql://localhost:5432/postgres
    python db_injector.py --fresh   # 清空后全量重灌
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from frontier_physics_constants import FRONTIER_PHYSICS_BLOCKS  # noqa: E402

# ---------------------------------------------------------------------------
# 连接与 Schema 契约
# ---------------------------------------------------------------------------
DEFAULT_DATABASE_URL = os.getenv(
    "MEL_DATABASE_URL",
    "postgresql://localhost:5432/postgres",
)

DEFAULT_CSV_CANDIDATES: tuple[Path, ...] = (
    _PROJECT_ROOT / "data" / "features_ready.csv",
    _PROJECT_ROOT / "data" / "feature_ready.csv",
)

PHYSICS_FACTOR_COLUMNS: tuple[str, ...] = (
    "heat_index",
    "wind_shear_alpha",
    "bifacial_gain_index",
    "panel_efficiency_discount",
)

PHYSICS_ALIASES: dict[str, tuple[str, ...]] = {
    "wind_shear_alpha": ("wind_shear_coefficient", "wind_shear_alpha"),
}

CHUNK_SIZE = 5000
MIN_INGEST_HOURS = 1536
DEFAULT_MOCK_NODE = "PJM_HUB"

DDL_MARKET_NODES = """
CREATE TABLE IF NOT EXISTS market_nodes (
    node_id    VARCHAR(20)  PRIMARY KEY,
    rto_name   VARCHAR(64)  NOT NULL,
    zone_name  VARCHAR(64)  NOT NULL,
    node_type  VARCHAR(32)  NOT NULL
);
"""

DDL_RTO_HOURLY_METRICS = """
CREATE TABLE IF NOT EXISTS rto_hourly_metrics (
    timestamp                  TIMESTAMPTZ    NOT NULL,
    node_id                    VARCHAR(20)    NOT NULL,
    price_da                   NUMERIC(12, 4),
    price_rt                   NUMERIC(12, 4),
    system_load                NUMERIC(14, 4),
    heat_index                 NUMERIC(12, 6),
    wind_shear_alpha           NUMERIC(12, 6),
    bifacial_gain_index        NUMERIC(12, 6),
    panel_efficiency_discount  NUMERIC(12, 6),
    PRIMARY KEY (timestamp, node_id),
    CONSTRAINT fk_rto_metrics_node
        FOREIGN KEY (node_id) REFERENCES market_nodes (node_id)
);
"""

DDL_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_rto_metrics_timestamp "
    "ON rto_hourly_metrics (timestamp);",
    "CREATE INDEX IF NOT EXISTS idx_rto_metrics_node "
    "ON rto_hourly_metrics (node_id);",
    "CREATE INDEX IF NOT EXISTS idx_rto_metrics_node_ts "
    "ON rto_hourly_metrics (node_id, timestamp);",
)


# ---------------------------------------------------------------------------
# 终端日志
# ---------------------------------------------------------------------------
def _log(level: str, msg: str) -> None:
    icons = {"INFO": "ℹ️", "OK": "✅", "WARN": "⚠️", "ERR": "❌", "BOLT": "⚡"}
    icon = icons.get(level, "·")
    print(f"{icon} [{level}] {msg}", flush=True)


def _banner(title: str) -> None:
    line = "═" * 64
    print(f"\n{line}\n  {title}\n{line}", flush=True)


# ---------------------------------------------------------------------------
# 引擎与 CSV
# ---------------------------------------------------------------------------
def create_db_engine(database_url: str = DEFAULT_DATABASE_URL) -> Engine:
    """创建 SQLAlchemy 引擎并探测连接。"""
    try:
        engine = create_engine(database_url, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except SQLAlchemyError as exc:
        raise ConnectionError(
            f"无法连接 PostgreSQL: {database_url}\n"
            f"请确认服务已启动: brew services start postgresql@14\n"
            f"原始错误: {exc}"
        ) from exc


def _resolve_csv_path(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(f"指定的 CSV 不存在: {path}")
        return path

    for candidate in DEFAULT_CSV_CANDIDATES:
        if candidate.is_file():
            return candidate

    lake_db = _PROJECT_ROOT / "data_lake" / "mel_env_history.db"
    if lake_db.is_file():
        try:
            import sqlite3

            _log("INFO", f"未找到 data/*.csv，从 SQLite 数据中台导出: {lake_db}")
            with sqlite3.connect(lake_db) as conn:
                cur = conn.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='features_ready'"
                )
                if cur.fetchone():
                    df = pd.read_sql("SELECT * FROM features_ready", conn)
                    out = _PROJECT_ROOT / "data" / "features_ready.csv"
                    out.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(out, index=False)
                    _log("OK", f"已导出 {len(df)} 行至 {out}")
                    return out
        except Exception as exc:  # noqa: BLE001
            _log("WARN", f"SQLite 导出失败，继续查找 CSV: {exc}")

    raise FileNotFoundError(
        "未找到特征 CSV。请先运行:\n"
        "  python data_pipeline.py --lookback-hours 720 --export-csv\n"
        "或将 features_ready.csv 放入 data/ 目录。"
    )


def _pick_column(df: pd.DataFrame, primary: str, aliases: tuple[str, ...] = ()) -> Optional[str]:
    if primary in df.columns:
        return primary
    for alt in aliases:
        if alt in df.columns:
            return alt
    return None


def _normalize_timestamptz(series: pd.Series) -> pd.Series:
    ts = pd.to_datetime(series, utc=True, errors="coerce")
    return ts.dt.tz_convert("UTC")


def _coerce_numeric(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


# ---------------------------------------------------------------------------
# 数据清洗
# ---------------------------------------------------------------------------
def load_and_clean_features(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取 CSV，映射市场/物理列，产出节点维表与事实表行集。"""
    try:
        raw = pd.read_csv(csv_path)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"读取 CSV 失败: {csv_path}") from exc

    if raw.empty:
        raise ValueError(f"CSV 为空: {csv_path}")

    df = raw.copy()
    ts_col = _pick_column(df, "timestamp")
    if ts_col is None:
        raise ValueError("CSV 必须包含 timestamp 列")

    df["timestamp"] = _normalize_timestamptz(df[ts_col])
    df = df.dropna(subset=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    price_da_col = _pick_column(df, "price_da", ("spot_price", "da_price", "lmp_da"))
    price_rt_col = _pick_column(df, "price_rt", ("rt_price", "lmp_rt"))
    load_col = _pick_column(df, "system_load", ("total_load", "load", "system_demand"))

    if price_da_col is None and price_rt_col is None:
        raise ValueError("CSV 需包含 spot_price / price_da 至少一种电价列")

    if price_da_col is None:
        df["price_da"] = pd.to_numeric(df[price_rt_col], errors="coerce")
    else:
        df["price_da"] = pd.to_numeric(df[price_da_col], errors="coerce")

    if price_rt_col is None:
        df["price_rt"] = df["price_da"]
    else:
        df["price_rt"] = pd.to_numeric(df[price_rt_col], errors="coerce")

    if load_col is None:
        df["system_load"] = 0.0
        _log("WARN", "未找到 total_load / system_load，system_load 填 0")
    else:
        df["system_load"] = pd.to_numeric(df[load_col], errors="coerce")

    for col in PHYSICS_FACTOR_COLUMNS:
        aliases = PHYSICS_ALIASES.get(col, ())
        src = _pick_column(df, col, aliases)
        if src is None:
            df[col] = 0.0
            block_hint = next(
                (name for name, cols in FRONTIER_PHYSICS_BLOCKS.items() if col in cols),
                col,
            )
            _log("WARN", f"缺少物理列 {col}（{block_hint}），已填 0")
        elif src != col:
            df[col] = pd.to_numeric(df[src], errors="coerce")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _coerce_numeric(
        df,
        ["price_da", "price_rt", "system_load", *PHYSICS_FACTOR_COLUMNS],
    )

    if "node_id" in df.columns:
        df["node_id"] = df["node_id"].astype(str).str.slice(0, 20)
    else:
        df["node_id"] = "PJM_HUB"

    if "rto_name" not in df.columns:
        df["rto_name"] = "PJM"
    if "zone_name" not in df.columns:
        df["zone_name"] = "HUB_ZONE"
    if "node_type" not in df.columns:
        df["node_type"] = "HUB"

    nodes = (
        df[["node_id", "rto_name", "zone_name", "node_type"]]
        .dropna(subset=["node_id"])
        .drop_duplicates(subset=["node_id"])
    )

    fact_cols = [
        "timestamp",
        "node_id",
        "price_da",
        "price_rt",
        "system_load",
        *PHYSICS_FACTOR_COLUMNS,
    ]
    facts = df[fact_cols].copy()
    facts["node_id"] = facts["node_id"].fillna("PJM_HUB").astype(str).str.slice(0, 20)
    facts = facts.drop_duplicates(subset=["timestamp", "node_id"], keep="last")
    facts = facts.sort_values(["timestamp", "node_id"]).reset_index(drop=True)

    return nodes, facts


def synthesize_mock_facts_nodes(n_hours: int = MIN_INGEST_HOURS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成高仿真时序事实表（用于 CSV 稀疏时自动扩容灌库）。"""
    hours = pd.date_range("2024-01-01", periods=n_hours, freq="h", tz="UTC")
    rng = np.random.default_rng(42)
    t = np.arange(n_hours, dtype=float)
    price_da = 300.0 + 50.0 * np.sin(t / 24.0) + rng.normal(0, 5.0, n_hours)
    price_rt = price_da + rng.normal(0, 1.5, n_hours)
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
            "system_load": 11000.0 + 500.0 * np.sin(t / 24.0) + rng.normal(0, 80.0, n_hours),
            "heat_index": 25.0 + rng.normal(0, 1.0, n_hours),
            "wind_shear_alpha": np.clip(0.15 + rng.normal(0, 0.02, n_hours), 0.05, 0.35),
            "bifacial_gain_index": 1.0 + rng.normal(0, 0.05, n_hours),
            "panel_efficiency_discount": np.clip(
                0.02 + np.cumsum(rng.normal(0, 0.001, n_hours)), 0.0, 0.15
            ),
        }
    )
    return nodes, facts


def ensure_minimum_ingest_rows(
    nodes: pd.DataFrame,
    facts: pd.DataFrame,
    min_hours: int = MIN_INGEST_HOURS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """事实表行数不足时，切换为大样本仿真时序。"""
    if len(facts) >= min_hours:
        return nodes, facts
    _log(
        "WARN",
        f"输入仅 {len(facts)} 行，低于灌库下限 {min_hours}h，自动扩容仿真时序。",
    )
    return synthesize_mock_facts_nodes(n_hours=min_hours)


# ---------------------------------------------------------------------------
# PostgreSQL 建表与批量灌入
# ---------------------------------------------------------------------------
def init_database_schema(engine: Engine) -> None:
    """自动执行 DDL 建表与索引。"""
    try:
        with engine.begin() as conn:
            conn.execute(text(DDL_MARKET_NODES))
            conn.execute(text(DDL_RTO_HOURLY_METRICS))
            for stmt in DDL_INDEXES:
                conn.execute(text(stmt))
        _log("OK", "Schema 就绪: market_nodes · rto_hourly_metrics · 索引")
    except SQLAlchemyError as exc:
        raise RuntimeError(f"建表失败: {exc}") from exc


def truncate_production_tables(engine: Engine) -> None:
    """清空事实表与维度表（--fresh 模式）。"""
    try:
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE rto_hourly_metrics RESTART IDENTITY CASCADE;"))
            conn.execute(text("TRUNCATE TABLE market_nodes RESTART IDENTITY CASCADE;"))
        _log("WARN", "已清空 market_nodes / rto_hourly_metrics（--fresh）")
    except SQLAlchemyError as exc:
        raise RuntimeError(f"清表失败: {exc}") from exc


def _bulk_to_sql(
    df: pd.DataFrame,
    table_name: str,
    engine: Engine,
    chunk_size: int = CHUNK_SIZE,
) -> int:
    """Pandas to_sql 分块高速写入。"""
    if df.empty:
        return 0
    try:
        df.to_sql(
            table_name,
            con=engine,
            if_exists="append",
            index=False,
            method="multi",
            chunksize=chunk_size,
        )
        return len(df)
    except SQLAlchemyError as exc:
        raise RuntimeError(f"批量写入表 {table_name} 失败: {exc}") from exc


def inject_nodes_upsert(engine: Engine, nodes: pd.DataFrame) -> int:
    """维度表灌入（主键冲突则忽略）。"""
    if nodes.empty:
        return 0
    try:
        with engine.begin() as conn:
            for row in nodes.itertuples(index=False):
                conn.execute(
                    text(
                        """
                        INSERT INTO market_nodes (node_id, rto_name, zone_name, node_type)
                        VALUES (:node_id, :rto_name, :zone_name, :node_type)
                        ON CONFLICT (node_id) DO NOTHING
                        """
                    ),
                    {
                        "node_id": str(row.node_id),
                        "rto_name": str(row.rto_name),
                        "zone_name": str(row.zone_name),
                        "node_type": str(row.node_type),
                    },
                )
        return len(nodes)
    except SQLAlchemyError as exc:
        raise RuntimeError(f"维度表灌入失败: {exc}") from exc


def inject_metrics_bulk(
    engine: Engine,
    facts: pd.DataFrame,
    chunk_size: int = CHUNK_SIZE,
) -> int:
    """
    事实表分块 to_sql 灌入；主键冲突行由 PostgreSQL 忽略（ON CONFLICT DO NOTHING）。
    """
    if facts.empty:
        return 0

    staging = "_staging_rto_hourly_metrics"
    total = 0
    try:
        with engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {staging};"))

        for start in range(0, len(facts), chunk_size):
            chunk = facts.iloc[start : start + chunk_size]
            chunk.to_sql(
                staging,
                con=engine,
                if_exists="replace" if start == 0 else "append",
                index=False,
                method="multi",
                chunksize=chunk_size,
            )
            total += len(chunk)

        with engine.begin() as conn:
            result = conn.execute(
                text(
                    f"""
                    INSERT INTO rto_hourly_metrics (
                        timestamp, node_id, price_da, price_rt, system_load,
                        heat_index, wind_shear_alpha, bifacial_gain_index,
                        panel_efficiency_discount
                    )
                    SELECT
                        timestamp, node_id, price_da, price_rt, system_load,
                        heat_index, wind_shear_alpha, bifacial_gain_index,
                        panel_efficiency_discount
                    FROM {staging}
                    ON CONFLICT (timestamp, node_id) DO NOTHING
                    """
                )
            )
            inserted = result.rowcount if result.rowcount is not None and result.rowcount >= 0 else total
            conn.execute(text(f"DROP TABLE IF EXISTS {staging};"))
        return int(inserted)
    except SQLAlchemyError as exc:
        raise RuntimeError(f"事实表批量灌入失败: {exc}") from exc


def count_table_rows(engine: Engine, table: str) -> int:
    with engine.connect() as conn:
        val = conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
    return int(val or 0)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run_injection(
    csv_path: Optional[Path] = None,
    database_url: str = DEFAULT_DATABASE_URL,
    fresh: bool = False,
    chunk_size: int = CHUNK_SIZE,
) -> None:
    _banner("MEL-F · PostgreSQL 时序数据高速灌入器")
    t0 = time.perf_counter()

    try:
        resolved_csv = _resolve_csv_path(csv_path)
        _log("INFO", f"读取特征宽表: {resolved_csv}")

        nodes, facts = load_and_clean_features(resolved_csv)
        nodes, facts = ensure_minimum_ingest_rows(nodes, facts, min_hours=MIN_INGEST_HOURS)
        _log(
            "INFO",
            f"清洗完成 · 节点={len(nodes)} · 事实行={len(facts)} · chunk={chunk_size}",
        )

        engine = create_db_engine(database_url)
        _log("OK", f"已连接 PostgreSQL: {database_url}")

        init_database_schema(engine)
        if fresh:
            truncate_production_tables(engine)

        inject_nodes_upsert(engine, nodes)
        inserted = inject_metrics_bulk(engine, facts, chunk_size=chunk_size)

        n_nodes = count_table_rows(engine, "market_nodes")
        n_facts = count_table_rows(engine, "rto_hourly_metrics")
        elapsed = time.perf_counter() - t0

        _log(
            "BOLT",
            "成功批量灌入海量时序数据至 PostgreSQL，"
            "联合主键索引已激活，可供窗口函数视图消费。",
        )
        _log("OK", f"market_nodes = {n_nodes} 行")
        _log("OK", f"rto_hourly_metrics = {n_facts} 行（本次尝试写入 {inserted} 行）")
        _log("OK", f"耗时 {elapsed:.2f}s")
        print(
            "\n👉 下一步请在终端执行:\n"
            "   python db_feature_view.py\n"
            "   python run_experiment.py --production-db\n",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        _log("ERR", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise SystemExit(1) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MEL-F PostgreSQL 时序库批量灌入器 (db_injector)",
    )
    parser.add_argument("--csv", type=Path, default=None, help="特征宽表 CSV 路径")
    parser.add_argument(
        "--database-url",
        type=str,
        default=DEFAULT_DATABASE_URL,
        help="PostgreSQL 连接 URL",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="灌入前清空 market_nodes / rto_hourly_metrics",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help=f"to_sql 分块大小（默认 {CHUNK_SIZE}）",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_injection(
        csv_path=args.csv,
        database_url=args.database_url,
        fresh=args.fresh,
        chunk_size=args.chunk_size,
    )
