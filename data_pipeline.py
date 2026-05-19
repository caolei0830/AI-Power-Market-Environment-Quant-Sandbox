# -*- coding: utf-8 -*-
"""
data_pipeline.py — 环境微特征智能数据中台（Data Pipeline）

职责
----
1. **fetch_all_data**：尝试公共 API 拉取；失败则自动降级为高仿真 Mock 数据。
2. **store_to_db**：SQLite + Parquet 双写，按时间戳去重后增量追加。
3. **clean_and_transform**：插值清洗 + 调用 ``environmental_feature_engineer`` 特征工程。
4. **run_daily_update**：一键日更入口，终端进度日志。

终端用法
--------
    python data_pipeline.py
    python data_pipeline.py --lookback-hours 168
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from environmental_feature_engineer import (  # noqa: E402
    FeatureSchema,
    align_and_merge_all,
    process_pollution_interaction,
    process_radiation_features,
)

# ---------------------------------------------------------------------------
# 日志着色（Windows 终端友好）
# ---------------------------------------------------------------------------
class PipelineLogger:
    """轻量终端日志器，输出 [INFO] / [SUCCESS] / [WARN] / [ERROR] 进度。"""

    @staticmethod
    def info(msg: str) -> None:
        print(f"[INFO] {msg}")

    @staticmethod
    def success(msg: str) -> None:
        print(f"[SUCCESS] {msg}")

    @staticmethod
    def warn(msg: str) -> None:
        print(f"[WARN] {msg}")

    @staticmethod
    def error(msg: str) -> None:
        print(f"[ERROR] {msg}", file=sys.stderr)


log = PipelineLogger()


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------
@dataclass
class PipelineConfig:
    """
    数据中台运行配置。

    Attributes
    ----------
    root_dir :
        项目数据根目录，默认 ``<项目>/data_lake``。
    db_path :
        SQLite 数据库文件路径。
    parquet_dir :
        Parquet 冷存储目录（按表分文件）。
    lookback_hours :
        每次日更拉取/生成的时间窗口长度（小时）。
    schema :
        列名契约，与特征工程模块保持一致。
    request_timeout_sec :
        HTTP 请求超时（秒），超时即触发 Mock 降级。
    """

    root_dir: Path = field(default_factory=lambda: _PROJECT_ROOT / "data_lake")
    db_path: Optional[Path] = None
    parquet_dir: Optional[Path] = None
    lookback_hours: int = 72
    schema: FeatureSchema = field(default_factory=FeatureSchema)
    request_timeout_sec: float = 8.0
    default_lat: float = 30.25
    default_lon: float = 120.15
    default_cities: tuple[str, ...] = ("hangzhou", "ningbo", "jiaxing")

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        if self.db_path is None:
            self.db_path = self.root_dir / "mel_env_history.db"
        else:
            self.db_path = Path(self.db_path)
        if self.parquet_dir is None:
            self.parquet_dir = self.root_dir / "parquet"
        else:
            self.parquet_dir = Path(self.parquet_dir)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)


# 各表去重主键（与业务粒度一致）
_DEDUP_KEYS: dict[str, list[str]] = {
    "market": ["timestamp"],
    "era5": ["timestamp", "lat", "lon"],
    "pv_stations": ["station_id"],
    "aqi": ["timestamp", "city"],
    "load_base": ["timestamp", "city"],
    "water": ["timestamp"],
    "features_ready": ["timestamp"],
}


# ---------------------------------------------------------------------------
# 核心类：环境微特征智能数据中台
# ---------------------------------------------------------------------------
class EnvironmentalDataPipeline:
    """
    环境微特征一体化数据中台。

    将「获取 → 存储 → 清洗对齐」封装为可复用类，供 CLI、Streamlit 试验舱调用。
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._last_fetch_source: str = "unknown"
        self._ensure_database_schema()

    # ------------------------------------------------------------------
    # 1. 自动化获取
    # ------------------------------------------------------------------
    def fetch_all_data(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        拉取（或降级生成）全量原始表。

        网络容错策略
        ------------
        - 依次尝试和风天气、公开环境类接口（需环境变量 Token）。
        - 任一环节超时 / 无密钥 / 额度受限 → 记录 WARN，不抛异常。
        - 最终保证返回完整六张表，来源标记于 ``self.last_fetch_source``。
        """
        end = end or datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        # 内部统一用 naive UTC，避免与 SQLite 历史数据混用 tz-aware / str
        if end.tzinfo is not None:
            end = end.replace(tzinfo=None)
        start = start or (end - timedelta(hours=self.config.lookback_hours))
        if start.tzinfo is not None:
            start = start.replace(tzinfo=None)

        log.info(
            f"开始获取环境与市场数据窗口: {start.isoformat()} ~ {end.isoformat()}"
        )

        partial: dict[str, pd.DataFrame] = {}
        api_ok = False

        if requests is not None:
            try:
                partial = self._try_fetch_from_apis(start, end)
                if self._validate_partial_tables(partial):
                    api_ok = True
                    self._last_fetch_source = "live_api"
                    log.success("公共 API 数据拉取成功，进入存储流程。")
            except Exception as exc:  # noqa: BLE001 — 必须兜底，保证流水线不中断
                log.warn(f"API 拉取失败，将切换 Mock 降级: {type(exc).__name__}: {exc}")
        else:
            log.warn("未安装 requests，跳过 API，直接使用 Mock 降级。")

        if not api_ok:
            partial = self._generate_mock_live_data(start, end)
            self._last_fetch_source = "mock_live"
            log.warn("已启用 _generate_mock_live_data() 高仿真备份数据。")

        return self._finalize_raw_tables(partial, start, end)

    def _try_fetch_from_apis(
        self,
        start: datetime,
        end: datetime,
    ) -> dict[str, pd.DataFrame]:
        """
        尝试调用外部 API 组装原始表。

        环境变量
        --------
        - ``QWEATHER_API_KEY``：和风天气 Web API Key
        - ``AMAP_API_KEY``：高德 Web 服务 Key（可选，用于城市元数据扩展）
        """
        qweather_key = os.environ.get("QWEATHER_API_KEY", "").strip()
        if not qweather_key:
            raise RuntimeError("缺少 QWEATHER_API_KEY，无法调用和风天气接口。")

        hours = pd.date_range(start, end, freq="h", inclusive="left")
        if len(hours) == 0:
            raise RuntimeError("时间窗口为空。")

        # --- 和风：逐小时天气（温度、风速）---
        weather_rows = []
        for ts in hours:
            row = self._fetch_qweather_hour(
                qweather_key,
                self.config.default_lat,
                self.config.default_lon,
                ts,
            )
            if row:
                weather_rows.append(row)

        if len(weather_rows) < max(3, len(hours) // 4):
            raise RuntimeError("和风天气有效样本不足，触发降级。")

        weather_df = pd.DataFrame(weather_rows)
        market = self._build_market_from_weather(weather_df, hours)
        water = pd.DataFrame(
            {
                "timestamp": hours,
                "water_temp": 12.0 + 8.0 * np.sin(np.arange(len(hours)) / 24.0),
            }
        )

        # 环境监测公开接口往往需专用 Token；此处用物理约束 Mock 补全 AQI
        aqi, load_base = self._build_city_aqi_load_mock(hours)

        era5, pv_stations = self._build_era5_and_pv_static(hours)

        return {
            "market": market,
            "era5": era5,
            "pv_stations": pv_stations,
            "aqi": aqi,
            "load_base": load_base,
            "water": water,
        }

    def _fetch_qweather_hour(
        self,
        api_key: str,
        lat: float,
        lon: float,
        ts: pd.Timestamp,
    ) -> Optional[dict[str, Any]]:
        """调用和风天气「实时/逐小时」类接口的单小时映射（示例实现）。"""
        if requests is None:
            return None
        # 和风天气 API：这里使用 weather 实时接口做映射（无 Key 时由上层降级）
        url = "https://devapi.qweather.com/v7/weather/now"
        params = {"location": f"{lon:.2f},{lat:.2f}", "key": api_key}
        resp = requests.get(
            url,
            params=params,
            timeout=self.config.request_timeout_sec,
        )
        if resp.status_code == 429:
            raise RuntimeError("和风 API 额度受限 (HTTP 429)")
        if resp.status_code != 200:
            return None
        payload = resp.json()
        if str(payload.get("code")) != "200":
            return None
        now = payload.get("now", {})
        temp = float(now.get("temp", 15.0))
        wind = float(now.get("windSpeed", 3.0))
        # 用实时值代表该小时（真实生产应使用逐小时预报/再分析）
        price = 300.0 + 0.8 * temp + 5.0 * wind
        return {
            "timestamp": ts,
            "temperature": temp,
            "wind_speed": wind,
            "spot_price": price,
            "total_load": 11000.0,
            "wind_forecast": 700.0 + 20.0 * wind,
            "solar_forecast": max(0.0, 500.0 + 200.0 * np.sin(ts.hour / 24.0 * 2 * np.pi)),
        }

    def _build_market_from_weather(
        self,
        weather_df: pd.DataFrame,
        hours: pd.DatetimeIndex,
    ) -> pd.DataFrame:
        """将气象字段映射为市场主表字段。"""
        base = weather_df.set_index("timestamp").reindex(hours).interpolate().ffill()
        return base.reset_index().rename(columns={"index": "timestamp"})[
            [
                "timestamp",
                "spot_price",
                "total_load",
                "wind_forecast",
                "solar_forecast",
                "temperature",
                "wind_speed",
            ]
        ]

    def _build_city_aqi_load_mock(
        self,
        hours: pd.DatetimeIndex,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """城市 AQI / 负荷基准（公开 API 受限时的可解释 Mock）。"""
        aqi_rows, load_rows = [], []
        for i, ts in enumerate(hours):
            for j, city in enumerate(self.config.default_cities):
                aqi_rows.append(
                    {
                        "timestamp": ts,
                        "city": city,
                        "pm25": 20.0 + 15.0 * np.sin(i / 12.0) + j * 4,
                        "no2": 15.0 + 8.0 * np.cos(i / 18.0) + j * 2,
                        "aod": 0.2 + 0.04 * j + 0.02 * np.sin(i / 24.0),
                    }
                )
                load_rows.append(
                    {
                        "timestamp": ts,
                        "city": city,
                        "industrial_load_base": 450.0 + j * 90 + 40 * np.sin(i / 24.0),
                    }
                )
        return pd.DataFrame(aqi_rows), pd.DataFrame(load_rows)

    def _build_era5_and_pv_static(
        self,
        hours: pd.DatetimeIndex,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """ERA5 风格网格辐射 + 固定光伏电站清单。"""
        grid_lats = [30.0, 30.25, 30.5]
        grid_lons = [120.0, 120.25, 120.5]
        era5_rows = []
        for i, ts in enumerate(hours):
            for lat in grid_lats:
                for lon in grid_lons:
                    era5_rows.append(
                        {
                            "timestamp": ts,
                            "lat": lat,
                            "lon": lon,
                            "ssrd": max(
                                0.0,
                                120.0 + 100.0 * np.sin(i / 24.0 * 2 * np.pi),
                            ),
                        }
                    )
        pv_stations = pd.DataFrame(
            {
                "station_id": ["pv_hangzhou", "pv_ningbo", "pv_jiaxing"],
                "lat": [30.1, 30.35, 30.45],
                "lon": [120.1, 120.3, 120.45],
                "capacity_mw": [120.0, 180.0, 200.0],
            }
        )
        return pd.DataFrame(era5_rows), pv_stations

    def _generate_mock_live_data(
        self,
        start: datetime,
        end: datetime,
    ) -> dict[str, pd.DataFrame]:
        """
        高仿真 Mock 备份：叠加正弦周期、电价尖峰噪声与物理可行范围。

        业务要求：在无 Token / 断网 / 限流时仍能闭环运行。
        """
        hours = pd.date_range(start, end, freq="h", inclusive="left")
        n = len(hours)
        if n == 0:
            hours = pd.date_range(end - timedelta(hours=24), end, freq="h", inclusive="left")
            n = len(hours)

        rng = np.random.default_rng(int(end.timestamp()) % 10_000)
        t = np.arange(n)

        # 电价：日周期 + 晚高峰尖峰脉冲 + 噪声
        base_price = 310.0 + 55.0 * np.sin(t / 24.0 * 2 * np.pi)
        peak_mask = ((hours.hour >= 18) & (hours.hour <= 21)).astype(float)
        spike = peak_mask * rng.normal(80, 25, n)
        spot_price = np.clip(base_price + spike + rng.normal(0, 12, n), 50, 1500)

        market = pd.DataFrame(
            {
                "timestamp": hours,
                "spot_price": spot_price,
                "total_load": 10500.0 + 1800.0 * np.sin(t / 24.0) + rng.normal(0, 120, n),
                "wind_forecast": np.clip(750.0 + 180.0 * rng.standard_normal(n), 0, None),
                "solar_forecast": np.clip(
                    550.0 + 280.0 * np.sin(t / 24.0 * 2 * np.pi), 0, None
                ),
                "temperature": 8.0 + 10.0 * np.sin(t / 24.0) + rng.normal(0, 0.6, n),
                "wind_speed": np.clip(3.5 + 1.8 * rng.standard_normal(n), 0, 25),
            }
        )

        era5, pv_stations = self._build_era5_and_pv_static(hours)
        aqi, load_base = self._build_city_aqi_load_mock(hours)
        water = pd.DataFrame(
            {
                "timestamp": hours,
                "water_temp": 9.0 + 7.0 * np.sin(t / 24.0 - 0.5) + rng.normal(0, 0.25, n),
            }
        )

        return {
            "market": market,
            "era5": era5,
            "pv_stations": pv_stations,
            "aqi": aqi,
            "load_base": load_base,
            "water": water,
        }

    def _validate_partial_tables(self, tables: dict[str, pd.DataFrame]) -> bool:
        required = {"market", "era5", "pv_stations", "aqi", "load_base", "water"}
        return required.issubset(tables.keys()) and all(
            not tables[k].empty for k in required
        )

    def _finalize_raw_tables(
        self,
        tables: dict[str, pd.DataFrame],
        start: datetime,
        end: datetime,
    ) -> dict[str, pd.DataFrame]:
        """统一时间戳类型与排序。"""
        ts_col = self.config.schema.timestamp
        out: dict[str, pd.DataFrame] = {}
        for name, df in tables.items():
            frame = df.copy()
            if ts_col in frame.columns:
                frame[ts_col] = pd.to_datetime(frame[ts_col])
                frame = frame.sort_values(ts_col)
            out[name] = frame.reset_index(drop=True)
        meta = {
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "window_start": start.isoformat(),
            "window_end": end.isoformat(),
            "source": self._last_fetch_source,
        }
        meta_path = self.config.root_dir / "last_fetch_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return out

    @property
    def last_fetch_source(self) -> str:
        """最近一次获取数据来源：``live_api`` | ``mock_live``。"""
        return self._last_fetch_source

    # ------------------------------------------------------------------
    # 2. 结构化本地存储
    # ------------------------------------------------------------------
    def _ensure_database_schema(self) -> None:
        """初始化 SQLite 表结构（DataFrame 动态建表，首次写入时创建）。"""
        assert self.config.db_path is not None
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.config.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS pipeline_meta ("
                "key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
            )
            conn.commit()

    def store_to_db(self, tables: dict[str, pd.DataFrame]) -> dict[str, int]:
        """
        去重后增量写入 SQLite，并同步导出 Parquet。

        Returns
        -------
        dict
            各表本次写入后的总行数。
        """
        assert self.config.db_path is not None
        assert self.config.parquet_dir is not None

        row_counts: dict[str, int] = {}
        with sqlite3.connect(self.config.db_path) as conn:
            for table_name, df_new in tables.items():
                if df_new is None or df_new.empty:
                    log.warn(f"表 {table_name} 为空，跳过写入。")
                    continue

                keys = _DEDUP_KEYS.get(table_name, ["timestamp"])
                df_new = self._normalize_for_storage(df_new, keys)

                if self._table_exists(conn, table_name):
                    df_old = pd.read_sql(f'SELECT * FROM "{table_name}"', conn)
                    df_old = self._normalize_for_storage(df_old, keys)
                    df_all = pd.concat([df_old, df_new], ignore_index=True)
                else:
                    df_all = df_new

                df_all = self._normalize_for_storage(df_all, keys)
                df_all = df_all.drop_duplicates(subset=keys, keep="last")
                df_all = df_all.sort_values(keys).reset_index(drop=True)
                df_all.to_sql(table_name, conn, if_exists="replace", index=False)
                row_counts[table_name] = len(df_all)

                pq_path = self.config.parquet_dir / f"{table_name}.parquet"
                try:
                    df_all.to_parquet(pq_path, index=False)
                    pq_msg = f" · Parquet={pq_path.name}"
                except Exception as exc:  # noqa: BLE001
                    pq_msg = f" · Parquet 跳过({type(exc).__name__})"

                log.success(
                    f"表 {table_name} 增量写入完成 · 行数={len(df_all)}{pq_msg}"
                )

            conn.execute(
                "INSERT OR REPLACE INTO pipeline_meta(key, value, updated_at) VALUES (?,?,?)",
                (
                    "last_store",
                    json.dumps(row_counts, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.commit()

        return row_counts

    @staticmethod
    def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        return cur.fetchone() is not None

    @staticmethod
    def _normalize_for_storage(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
        """统一去重键类型，避免 SQLite 历史行与新行 timestamp 类型不一致导致排序失败。"""
        out = df.copy()
        for col in keys:
            if col not in out.columns:
                raise ValueError(f"存储去重键 {col} 不在 DataFrame 列中: {list(out.columns)}")
            if col == "timestamp" or col.endswith("_at"):
                out[col] = pd.to_datetime(out[col], utc=True, errors="coerce").dt.tz_convert(None)
            elif col in ("lat", "lon"):
                out[col] = pd.to_numeric(out[col], errors="coerce")
            elif col == "station_id" or col == "city":
                out[col] = out[col].astype(str)
        return out

    def load_table(self, table_name: str) -> pd.DataFrame:
        """从 SQLite 读取单表；若不存在则返回空 DataFrame。"""
        assert self.config.db_path is not None
        if not self.config.db_path.is_file():
            return pd.DataFrame()
        with sqlite3.connect(self.config.db_path) as conn:
            if not self._table_exists(conn, table_name):
                return pd.DataFrame()
            return pd.read_sql(f'SELECT * FROM "{table_name}"', conn)

    def load_raw_tables_from_db(self) -> dict[str, pd.DataFrame]:
        """加载实验所需的六张原始表。"""
        names = ["market", "era5", "pv_stations", "aqi", "load_base", "water"]
        return {n: self.load_table(n) for n in names}

    # ------------------------------------------------------------------
    # 3. 清洗与特征对齐
    # ------------------------------------------------------------------
    def clean_and_transform(
        self,
        tables: Optional[dict[str, pd.DataFrame]] = None,
    ) -> pd.DataFrame:
        """
        清洗 → 插值 → 特征工程 → 产出 ``df_ready`` 宽表。

        Parameters
        ----------
        tables :
            若为 None，则从本地 SQLite 读取历史全量。
        """
        log.info("开始清洗与特征对齐...")
        raw = tables if tables is not None else self.load_raw_tables_from_db()
        self._assert_raw_tables_nonempty(raw)

        cleaned = {k: self._impute_table(v, k) for k, v in raw.items()}
        df_ready = self._build_feature_wide_table(cleaned)

        # 建模滞后特征（与 run_experiment 保持一致）
        df_ready = self._add_price_and_water_lags(df_ready)
        df_ready = df_ready.sort_values("timestamp").reset_index(drop=True)

        store_tables = {"features_ready": df_ready}
        self.store_to_db(store_tables)

        log.success(f"特征矩阵对齐完成，样本数: {len(df_ready)}")
        return df_ready

    def _assert_raw_tables_nonempty(self, raw: dict[str, pd.DataFrame]) -> None:
        missing = [k for k, v in raw.items() if v is None or v.empty]
        if missing:
            raise ValueError(
                f"本地库缺少原始表: {missing}。请先运行 run_daily_update() 或 fetch_all_data()。"
            )

    def _impute_table(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """
        缺失值处理：数值列线性插值 + 前向填充。

        分类键（city / station_id）保留，按组插值。
        """
        if df.empty:
            return df
        out = df.copy()
        ts_col = self.config.schema.timestamp
        if ts_col in out.columns:
            out[ts_col] = pd.to_datetime(out[ts_col])

        num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        group_keys = [c for c in ("city", "lat", "lon", "station_id") if c in out.columns]

        if group_keys:
            out = out.sort_values(group_keys + ([ts_col] if ts_col in out.columns else []))
            out[num_cols] = (
                out.groupby(group_keys, dropna=False)[num_cols]
                .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
            )
            out[num_cols] = out.groupby(group_keys, dropna=False)[num_cols].ffill()
        else:
            if ts_col in out.columns:
                out = out.sort_values(ts_col)
            out[num_cols] = out[num_cols].interpolate(
                method="linear", limit_direction="both"
            )
            out[num_cols] = out[num_cols].ffill()

        out[num_cols] = out[num_cols].fillna(0)
        return out

    def _build_feature_wide_table(self, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """调用 environmental_feature_engineer 完成辐射/污染/对齐。"""
        schema = self.config.schema
        df_rad = process_radiation_features(
            tables["era5"], tables["pv_stations"], schema=schema
        )
        df_pol = process_pollution_interaction(
            tables["aqi"], tables["load_base"], schema=schema
        )
        return align_and_merge_all(
            tables["market"],
            df_rad,
            df_pol,
            tables["water"],
            schema=schema,
        )

    @staticmethod
    def _add_price_and_water_lags(df: pd.DataFrame) -> pd.DataFrame:
        """追加电价与水温滞后列，供 LightGBM 试验直接使用。"""
        out = df.copy()
        if "spot_price" in out.columns:
            out["spot_price_lag_1h"] = out["spot_price"].shift(1)
            out["spot_price_lag_24h"] = out["spot_price"].shift(24)
        if "water_temp" in out.columns:
            for lag in (1, 3, 6, 24):
                out[f"water_temp_lag_{lag}h"] = out["water_temp"].shift(lag)
        return out

    def load_df_ready(self) -> pd.DataFrame:
        """读取已对齐的特征宽表 ``features_ready``。"""
        return self.load_table("features_ready")

    # ------------------------------------------------------------------
    # 4. 一键日更
    # ------------------------------------------------------------------
    def run_daily_update(self) -> pd.DataFrame:
        """
        一键执行：获取 → 存储 → 清洗对齐。

        Returns
        -------
        pd.DataFrame
            最新 ``df_ready`` 特征宽表。
        """
        log.info("=== 环境微特征智能数据中台 · 日更任务启动 ===")
        tables = self.fetch_all_data()
        log.info(f"数据来源: {self.last_fetch_source}")

        row_counts = self.store_to_db(tables)
        log.success(
            "成功增量写入 SQLite 数据库: "
            + ", ".join(f"{k}={v}" for k, v in row_counts.items())
        )

        df_ready = self.clean_and_transform(tables)
        log.info("=== 日更任务完成 ===")
        return df_ready

    def export_csv_snapshot(self, output_dir: Optional[Path] = None) -> Path:
        """
        将 SQLite 中六张原始表导出为 CSV，供 ``run_experiment.py --data-dir`` 使用。

        Returns
        -------
        Path
            导出目录路径。
        """
        out_dir = output_dir or (_PROJECT_ROOT / "data")
        out_dir.mkdir(parents=True, exist_ok=True)
        mapping = {
            "market": "market.csv",
            "era5": "era5.csv",
            "pv_stations": "pv_stations.csv",
            "aqi": "aqi.csv",
            "load_base": "load_base.csv",
            "water": "water.csv",
        }
        for table, fname in mapping.items():
            df = self.load_table(table)
            if not df.empty:
                df.to_csv(out_dir / fname, index=False)
        ready = self.load_df_ready()
        if not ready.empty:
            ready.to_csv(out_dir / "features_ready.csv", index=False)
        log.success(f"CSV 快照已导出至 {out_dir}")
        return out_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="环境微特征智能数据中台 · 日更")
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=72,
        help="本次抓取/生成的小时数窗口（默认 72）",
    )
    parser.add_argument(
        "--export-csv",
        action="store_true",
        help="日更完成后导出 data/*.csv 快照",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = PipelineConfig(lookback_hours=args.lookback_hours)
    pipeline = EnvironmentalDataPipeline(cfg)
    df_ready = pipeline.run_daily_update()
    log.info(
        f"df_ready 列数={df_ready.shape[1]} · 时间范围="
        f"{df_ready['timestamp'].min()} ~ {df_ready['timestamp'].max()}"
    )
    if args.export_csv:
        pipeline.export_csv_snapshot()


if __name__ == "__main__":
    main()
