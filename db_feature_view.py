# -*- coding: utf-8 -*-
"""
db_feature_view.py — PostgreSQL SQL 高级窗口函数特征生成器

在库内创建 ``v_features_pipeline_ready`` 视图，用标准 SQL Window Functions
完成 24h 移动平均、1h 滞后与日前-实时价差基差（严格防未来信息泄露）。

用法
----
    python db_feature_view.py
    python db_feature_view.py --database-url postgresql://localhost:5432/postgres
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from typing import Any, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from db_injector import DEFAULT_DATABASE_URL, create_db_engine  # noqa: E402

# 供 run_experiment 等模块复用（PostgreSQL CREATE OR REPLACE VIEW）
SQL_CREATE_VIEW = """
CREATE OR REPLACE VIEW v_features_pipeline_ready AS
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

SQL_PREVIEW_RANDOM = """
SELECT *
FROM v_features_pipeline_ready
ORDER BY RANDOM()
LIMIT 5;
"""


def _log(level: str, msg: str) -> None:
    icons = {"INFO": "ℹ️", "OK": "✅", "WARN": "⚠️", "ERR": "❌"}
    icon = icons.get(level, "·")
    print(f"{icon} [{level}] {msg}", flush=True)


def _print_dataframe_table(df: pd.DataFrame) -> None:
    if df.empty:
        _log("WARN", "(空结果集)")
        return
    headers = list(df.columns)
    col_widths = [len(str(h)) for h in headers]
    rows_str: list[list[str]] = []
    for _, row in df.iterrows():
        cells = ["" if pd.isna(row[h]) else str(row[h]) for h in headers]
        rows_str.append(cells)
        for i, cell in enumerate(cells):
            col_widths[i] = max(col_widths[i], len(cell))

    def fmt_line(cells: list[str]) -> str:
        return " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))

    sep = "-+-".join("-" * w for w in col_widths)
    print(fmt_line([str(h) for h in headers]), flush=True)
    print(sep, flush=True)
    for cells in rows_str:
        print(fmt_line(cells), flush=True)


def _table_exists(engine: Engine, table_name: str) -> bool:
    sql = text(
        """
        SELECT EXISTS (
            SELECT 1
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_name = :name
        )
        """
    )
    with engine.connect() as conn:
        return bool(conn.execute(sql, {"name": table_name}).scalar())


def activate_feature_view(database_url: str = DEFAULT_DATABASE_URL) -> None:
    """在 PostgreSQL 内激活特征视图并随机抽样预览。"""
    try:
        engine = create_db_engine(database_url)
        _log("INFO", f"已连接 PostgreSQL: {database_url}")

        if not _table_exists(engine, "rto_hourly_metrics"):
            raise RuntimeError(
                "表 rto_hourly_metrics 不存在。请先执行: python db_injector.py"
            )

        with engine.begin() as conn:
            conn.execute(text(SQL_CREATE_VIEW))
        _log("OK", "视图 v_features_pipeline_ready 已激活 (CREATE OR REPLACE VIEW)")

        with engine.connect() as conn:
            row_count = conn.execute(
                text("SELECT COUNT(*) FROM rto_hourly_metrics")
            ).scalar()
        _log("INFO", f"事实表 rto_hourly_metrics 行数: {int(row_count or 0)}")

        preview = pd.read_sql_query(SQL_PREVIEW_RANDOM, engine)
        _log("INFO", "执行抽样查询: SELECT * FROM v_features_pipeline_ready ORDER BY RANDOM() LIMIT 5;")
        print("", flush=True)
        _print_dataframe_table(preview)

        if not preview.empty:
            non_null_ma = int(preview["price_rt_ma_24h"].notna().sum())
            non_null_lag = int(preview["price_rt_lag_1h"].notna().sum())
            _log(
                "OK",
                f"窗口函数校验 · price_rt_ma_24h 非空={non_null_ma}/{len(preview)} · "
                f"price_rt_lag_1h 非空={non_null_lag}/{len(preview)} · basis_spread 已物化",
            )

        print(
            "\n👉 下一步请在终端执行:\n"
            "   python run_experiment.py --production-db\n",
            flush=True,
        )
    except SQLAlchemyError as exc:
        _log("ERR", f"数据库错误: {exc}")
        traceback.print_exc()
        raise SystemExit(1) from exc
    except Exception as exc:  # noqa: BLE001
        _log("ERR", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise SystemExit(1) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MEL-F PostgreSQL 窗口特征视图 (v_features_pipeline_ready)",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=os.getenv("MEL_DATABASE_URL", DEFAULT_DATABASE_URL),
        help="PostgreSQL 连接 URL",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    activate_feature_view(database_url=args.database_url)
