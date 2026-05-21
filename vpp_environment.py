# -*- coding: utf-8 -*-
"""
vpp_environment.py — 虚拟电厂 (VPP) 储能控制 Gymnasium 环境

状态空间直接对接 ``v_features_pipeline_ready`` SQL 视图映射后的建模宽表
（基准市场特征 + 环境微特征 + 归一化 SOC），与 LightGBM 增强模型特征列对齐。

数据加载三级容灾（与 Web 试验舱一致，不依赖 Streamlit）：
  1. PostgreSQL 生产库 ``build_wide_table_from_production_db``
  2. 内存 SQLite 沙盒 + Mock 1536h 自愈灌库
  3. 合成大样本管线 ``build_full_feature_wide_table``

用法
----
    from vpp_environment import VPPEnvironment, load_vpp_market_data, ExperimentConfig

    env = VPPEnvironment(split="train", production_db=True)
    obs, info = env.reset()
    obs, reward, term, trunc, info = env.step(action)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "未安装 gymnasium。请执行: pip install gymnasium"
    ) from exc

from run_experiment import (
    BENCHMARK_FEATURE_COLUMNS,
    COL_TARGET,
    COL_TIMESTAMP,
    ENHANCED_ENV_COLUMNS,
    EXPERIMENT_LOOKBACK_HOURS_DEFAULT,
    MIN_MODELING_ROWS,
    STORAGE_CAPACITY_MWH,
    STORAGE_CYCLE_COST,
    STORAGE_ETA,
    STORAGE_MIN_SOC_RATIO,
    STORAGE_POWER_MW,
    ExperimentConfig,
    _fallback_hourly_pnl,
    _robust_fill_enhanced_env_features,
    adapt_sql_pipeline_to_model_frame,
    build_full_feature_wide_table,
    build_wide_table_from_production_db,
    ensure_sufficient_wide_table,
    temporal_train_test_split,
)

# 与增强 LightGBM 擂台一致：SQL 视图特征 + 储能 SOC
VPP_FEATURE_COLUMNS: tuple[str, ...] = (
    *BENCHMARK_FEATURE_COLUMNS,
    *ENHANCED_ENV_COLUMNS,
)
VPP_OBS_COLUMNS: tuple[str, ...] = (*VPP_FEATURE_COLUMNS, "soc_ratio")

DEFAULT_MOCK_NODE = "PJM_HUB"
MOCK_SEED_HOURS = EXPERIMENT_LOOKBACK_HOURS_DEFAULT

SQL_LOAD_PIPELINE_WIDE = """
SELECT *
FROM v_features_pipeline_ready
ORDER BY timestamp ASC
"""

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


@dataclass
class VPPDataBundle:
    """RL 训练/回测用预处理数据包。"""

    observations: np.ndarray  # (T, obs_dim) 已标准化
    settlement_prices: np.ndarray  # (T,) 元/MWh，用于奖励结算
    timestamps: pd.Series
    feature_means: np.ndarray
    feature_stds: np.ndarray
    n_train: int
    n_test: int
    data_source: str


def _generate_mock_seed_tables(n_hours: int = MOCK_SEED_HOURS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """第三级 Mock：1536h 仿真时序（与 experiment_app 语义对齐）。"""
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
            "heat_index": 28.0 + 8.0 * np.sin(t / 24.0) + rng.normal(0, 0.5, n_hours),
            "wind_shear_alpha": np.clip(0.12 + 0.04 * rng.standard_normal(n_hours), 0.05, 0.35),
            "bifacial_gain_index": np.clip(1.0 + 0.05 * np.sin(t / 48.0), 0.9, 1.15),
            "panel_efficiency_discount": np.clip(
                0.02 + 0.01 * np.maximum(0, np.sin(t / 72.0)), 0.0, 0.12
            ),
        }
    )
    return nodes, facts


def _activate_sqlite_feature_view(engine: Engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("DROP VIEW IF EXISTS v_features_pipeline_ready"))
        conn.execute(text(DDL_SQLITE_MARKET_NODES))
        conn.execute(text(DDL_SQLITE_RTO_METRICS))
        conn.execute(text(SQL_CREATE_VIEW_SQLITE))


def _load_via_memory_sqlite(config: ExperimentConfig) -> pd.DataFrame:
    """Tier-2/3：内存 SQLite + Mock 1536h → SQL 窗口宽表。"""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    nodes, facts = _generate_mock_seed_tables(MOCK_SEED_HOURS)
    with engine.begin() as conn:
        conn.execute(text(DDL_SQLITE_MARKET_NODES))
        conn.execute(text(DDL_SQLITE_RTO_METRICS))
    nodes.to_sql("market_nodes", engine, if_exists="append", index=False)
    facts.to_sql("rto_hourly_metrics", engine, if_exists="append", index=False, method="multi")
    _activate_sqlite_feature_view(engine)

    df_sql = pd.read_sql_query(SQL_LOAD_PIPELINE_WIDE, engine)
    if df_sql.empty:
        raise ValueError("内存 SQLite 视图查询为空。")

    df_wide = adapt_sql_pipeline_to_model_frame(df_sql)
    df_wide = _robust_fill_enhanced_env_features(df_wide)
    required = [COL_TARGET] + list(VPP_FEATURE_COLUMNS)
    return ensure_sufficient_wide_table(config, df_wide, required)


def load_vpp_market_frame(config: Optional[ExperimentConfig] = None) -> pd.DataFrame:
    """
    三级容灾加载建模宽表（1536h 纪律，与 run_experiment / Web 一致）。
    """
    config = config or ExperimentConfig(production_db=True, lookback_hours=MOCK_SEED_HOURS)
    required = [COL_TARGET] + list(VPP_FEATURE_COLUMNS)
    sources: list[str] = []

    if config.production_db:
        try:
            df = build_wide_table_from_production_db(config)
            sources.append("postgresql")
            return _finalize_wide_frame(df, config, sources)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"PostgreSQL 加载失败 ({exc!r})，降级内存 SQLite 沙盒。",
                UserWarning,
                stacklevel=2,
            )

    try:
        df = _load_via_memory_sqlite(config)
        sources.append("memory_sqlite_mock")
        return _finalize_wide_frame(df, config, sources)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(
            f"内存 SQLite 失败 ({exc!r})，降级合成大样本管线。",
            UserWarning,
            stacklevel=2,
        )

    df = build_full_feature_wide_table(config)
    sources.append("synthetic_pipeline")
    return _finalize_wide_frame(df, config, sources)


def _finalize_wide_frame(
    df: pd.DataFrame,
    config: ExperimentConfig,
    sources: list[str],
) -> pd.DataFrame:
    """裁剪 lookback、补齐列、记录数据源。"""
    out = _robust_fill_enhanced_env_features(df.copy())
    for col in VPP_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
    out = out.sort_values(COL_TIMESTAMP).reset_index(drop=True)

    if config.lookback_hours > 0 and len(out) > config.lookback_hours:
        out = out.tail(int(config.lookback_hours)).reset_index(drop=True)

    frame = out[[COL_TIMESTAMP, COL_TARGET, *VPP_FEATURE_COLUMNS]].copy()
    frame = frame.dropna(subset=[COL_TARGET, *VPP_FEATURE_COLUMNS]).reset_index(drop=True)
    if len(frame) < MIN_MODELING_ROWS:
        raise ValueError(
            f"VPP 有效样本仅 {len(frame)} 行（目标 ≥{MIN_MODELING_ROWS}）。"
            "请执行 db_injector / data_pipeline 或增大 lookback_hours。"
        )
    frame.attrs["vpp_data_source"] = " → ".join(sources)
    return frame


def build_vpp_data_bundle(
    config: Optional[ExperimentConfig] = None,
    *,
    train_ratio: Optional[float] = None,
) -> VPPDataBundle:
    """构建标准化观测矩阵与结算电价序列（时序 train/test 切分）。"""
    config = config or ExperimentConfig(production_db=True)
    ratio = train_ratio if train_ratio is not None else config.train_ratio

    frame = load_vpp_market_frame(config)
    train_df, test_df = temporal_train_test_split(frame, ratio)

    feat_cols = list(VPP_FEATURE_COLUMNS)
    train_x = train_df[feat_cols].to_numpy(dtype=np.float64)
    test_x = test_df[feat_cols].to_numpy(dtype=np.float64)

    means = train_x.mean(axis=0)
    stds = train_x.std(axis=0)
    stds = np.where(stds < 1e-8, 1.0, stds)

    train_norm = (train_x - means) / stds
    test_norm = (test_x - means) / stds
    obs_all = np.vstack([train_norm, test_norm]).astype(np.float32)

    prices = np.concatenate(
        [
            train_df[COL_TARGET].to_numpy(dtype=np.float64),
            test_df[COL_TARGET].to_numpy(dtype=np.float64),
        ]
    )
    ts = pd.concat([train_df[COL_TIMESTAMP], test_df[COL_TIMESTAMP]], ignore_index=True)

    return VPPDataBundle(
        observations=obs_all,
        settlement_prices=prices.astype(np.float32),
        timestamps=ts,
        feature_means=means.astype(np.float32),
        feature_stds=stds.astype(np.float32),
        n_train=len(train_df),
        n_test=len(test_df),
        data_source=str(frame.attrs.get("vpp_data_source", "unknown")),
    )


def load_vpp_market_data(
    config: Optional[ExperimentConfig] = None,
    split: Literal["train", "test", "all"] = "train",
) -> tuple[np.ndarray, np.ndarray, VPPDataBundle]:
    """
    按 split 返回 (observations_without_soc, settlement_prices, bundle)。
    """
    bundle = build_vpp_data_bundle(config)
    if split == "train":
        sl = slice(0, bundle.n_train)
    elif split == "test":
        sl = slice(bundle.n_train, bundle.n_train + bundle.n_test)
    else:
        sl = slice(0, bundle.n_train + bundle.n_test)

    return (
        bundle.observations[sl],
        bundle.settlement_prices[sl],
        bundle,
    )


class VPPEnvironment(gym.Env):
    """
    虚拟电厂储能套利 Gymnasium 环境。

    - **观测**: 标准化 SQL 特征 + ``soc_ratio ∈ [0, 1]``
    - **动作**: ``[-1, 1]`` 连续标量，线性映射为 ``[-power_mw, +power_mw]``（MW，正=放电）
    - **奖励**: 与 ``run_experiment._fallback_hourly_pnl`` 一致的小时结算 PnL（元）
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[ExperimentConfig] = None,
        *,
        split: Literal["train", "test"] = "train",
        power_mw: float = STORAGE_POWER_MW,
        capacity_mwh: float = STORAGE_CAPACITY_MWH,
        eta: float = STORAGE_ETA,
        min_soc_ratio: float = STORAGE_MIN_SOC_RATIO,
        cycle_cost: float = STORAGE_CYCLE_COST,
        seed: Optional[int] = None,
        data_bundle: Optional[VPPDataBundle] = None,
    ) -> None:
        super().__init__()
        self.config = config or ExperimentConfig(production_db=True)
        self.split = split
        self.power_mw = float(power_mw)
        self.capacity_mwh = float(capacity_mwh)
        self.eta = float(eta)
        self.min_soc_ratio = float(min_soc_ratio)
        self.cycle_cost = float(cycle_cost)

        self._bundle = data_bundle or build_vpp_data_bundle(self.config)
        if split == "train":
            self._start = 0
            self._end = self._bundle.n_train
        else:
            self._start = self._bundle.n_train
            self._end = self._bundle.n_train + self._bundle.n_test

        self._features = self._bundle.observations[self._start : self._end]
        self._prices = self._bundle.settlement_prices[self._start : self._end]
        self._t = 0
        self._soc = (self.min_soc_ratio + 1.0) / 2.0 * self.capacity_mwh

        obs_dim = len(VPP_OBS_COLUMNS)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        self._np_random, _ = gym.utils.seeding.np_random(seed)

    @property
    def data_source(self) -> str:
        return self._bundle.data_source

    def _obs(self) -> np.ndarray:
        feat = self._features[self._t]
        soc_ratio = np.float32(self._soc / self.capacity_mwh)
        return np.concatenate([feat, [soc_ratio]]).astype(np.float32)

    def _apply_soc(self, power_mw: float) -> float:
        """与 run_experiment 储能优化器一致的 SOC 动力学。"""
        if power_mw >= 0.0:
            new_soc = self._soc - power_mw
        else:
            new_soc = self._soc - power_mw * self.eta
        soc_min = self.capacity_mwh * self.min_soc_ratio
        return float(np.clip(new_soc, soc_min, self.capacity_mwh))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)

        opts = options or {}
        self._t = int(opts.get("start_step", 0))
        self._soc = float(
            opts.get(
                "initial_soc",
                (self.min_soc_ratio + 1.0) / 2.0 * self.capacity_mwh,
            )
        )
        return self._obs(), {
            "step": self._t,
            "data_source": self.data_source,
            "split": self.split,
        }

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._t >= len(self._prices):
            raise RuntimeError("Episode 已结束，请先 reset()。")

        raw = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        raw = float(np.clip(raw, -1.0, 1.0))
        power_mw = raw * self.power_mw

        price = float(self._prices[self._t])
        reward = _fallback_hourly_pnl(power_mw, price, self.cycle_cost)

        attempted_soc = self._apply_soc(power_mw)
        soc_min = self.capacity_mwh * self.min_soc_ratio
        if attempted_soc <= soc_min + 1e-6 or attempted_soc >= self.capacity_mwh - 1e-6:
            power_mw = 0.0
            attempted_soc = self._soc

        self._soc = attempted_soc
        self._t += 1
        terminated = self._t >= len(self._prices)
        truncated = False

        info = {
            "power_mw": power_mw,
            "settlement_price": price,
            "soc_mwh": self._soc,
            "hourly_pnl_yuan": reward,
        }
        obs = self._obs() if not terminated else np.zeros(self.observation_space.shape, np.float32)
        return obs, float(reward), terminated, truncated, info
