# -*- coding: utf-8 -*-
"""
experiment_loader.py — 复线实验运行器 (Multi-Track Experiment Loader)

面向对象调度中心：读取 ``experiment_config.yaml``，对接 PostgreSQL / SQLite
三级容灾数据流，按轨道动态装配特征并执行 LightGBM + 储能套利 / PPO DRL 擂台。

用法
----
    python experiment_loader.py
    python experiment_loader.py --demo
    python experiment_loader.py --config experiment_config.yaml --export artifacts/track_matrix.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from run_experiment import (  # noqa: E402
    COL_TIMESTAMP,
    PRODUCTION_DATABASE_URL,
    ExperimentConfig,
    ModelMetrics,
    _fallback_hourly_pnl,
    _fallback_optimize_storage_point,
    _import_lightgbm,
    _import_optimizer,
    _robust_fill_enhanced_env_features,
    _sanitize_xy,
    backtest_storage_revenue_yuan,
    evaluate_regression,
    temporal_train_test_split,
)
from vpp_environment import load_vpp_market_frame  # noqa: E402

# ---------------------------------------------------------------------------
# 铁血防泄露：禁止进入特征矩阵 X 的 contemporaneous 电价列
# ---------------------------------------------------------------------------
FORBIDDEN_LEAKAGE_COLUMNS: frozenset[str] = frozenset(
    {
        "spot_price",       # 标签别名（由 price_rt 映射）
        "price_rt",         # 当前小时实时结算价 — 仅可作 y
        "price_da",         # 当期日前价 — 不作为 contemporaneous 特征
        "basis_spread",     # price_da - price_rt，含当期信息
    }
)

# 允许的历史电价滞后（SQL 窗口 / shift 产出）
ALLOWED_PRICE_HISTORY_COLUMNS: frozenset[str] = frozenset(
    {
        "spot_price_lag_1h",
        "spot_price_lag_24h",
        "price_rt_lag_1h",
        "price_rt_ma_24h",
    }
)

LGBM_TRACK_ORDER: tuple[str, ...] = (
    "A_baseline",
    "B_full_env",
    "C1_radiation_only",
    "C2_pollution_only",
    "C3_sst_only",
    "D_robustness_noise",
)

PPO_TRACK_NAME = "E_ppo_drl"

# Web 财富曲线对决：A 基准 / B 全开 / D 扰动 / E PPO
PNL_CURVE_TRACKS: tuple[str, ...] = (
    "A_baseline",
    "B_full_env",
    "D_robustness_noise",
    PPO_TRACK_NAME,
)


@dataclass
class TrackDataBundle:
    """单轨道建模宽表（已剥离泄露列）。"""

    track_name: str
    feature_columns: list[str]
    frame: pd.DataFrame
    target_column: str
    timestamp_column: str
    data_source: str


@dataclass
class TrackRunResult:
    """单轨道实验结果。"""

    track_name: str
    track_id: str
    model_family: str
    n_train: int
    n_test: int
    rmse: Optional[float]
    mae: Optional[float]
    r2: Optional[float]
    storage_revenue_cny: float
    profitable_hour_ratio: Optional[float] = None
    alpha_vs_idle_pct: Optional[float] = None
    data_source: str = ""
    notes: str = ""


class ExperimentLoader:
    """
    复线实验加载与运行器。

    职责
    ----
    - 解析 YAML 配置与特征词典
    - 三级容灾加载 SQL 窗口宽表
    - 轨道级防泄露特征契约 + 时序切分
    - LightGBM 预测 → 储能优化回测 / PPO 端到端控制
    """

    def __init__(
        self,
        config_path: Path | str = "experiment_config.yaml",
        *,
        production_db: bool = True,
        demo: bool = False,
    ) -> None:
        self.config_path = Path(config_path)
        if not self.config_path.is_file():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as fh:
            self.cfg: dict[str, Any] = yaml.safe_load(fh)

        self._global = self.cfg["global"]
        self._storage = self.cfg["storage"]
        self._lgbm_cfg = self.cfg["models"]["lightgbm"]
        self._lexicon: dict[str, Any] = self.cfg["feature_lexicon"]
        self._tracks: dict[str, Any] = self.cfg["tracks"]

        if self._global.get("split", {}).get("shuffle", False):
            raise ValueError("配置违反防泄露纪律：split.shuffle 必须为 false。")

        self.production_db = production_db and not demo
        self.demo = demo
        self.target_column: str = self._global["target_column"]
        self.timestamp_column: str = self._global["timestamp_column"]
        self.train_ratio: float = float(self._global["split"]["train_ratio"])

        self._wide_frame: Optional[pd.DataFrame] = None
        self._data_source: str = ""
        self._track_cache: dict[str, TrackDataBundle] = {}
        self._results: list[TrackRunResult] = []
        self._pnl_traces: dict[str, pd.DataFrame] = {}

        self.experiment_config = self._build_experiment_config()

    def _build_experiment_config(self) -> ExperimentConfig:
        """将 YAML 全局段映射为 ``ExperimentConfig``（供数据管线复用）。"""
        env_key = self._global["data_source"].get("database_url_env", "MEL_DATABASE_URL")
        db_url = os.getenv(env_key, PRODUCTION_DATABASE_URL)
        return ExperimentConfig(
            train_ratio=self.train_ratio,
            lookback_hours=int(self._global["lookback_hours"]),
            min_train_samples=int(self._global["min_train_samples"]),
            demo=self.demo,
            production_db=self.production_db,
            database_url=db_url,
            node_id=self._global.get("node_id", "PJM_HUB"),
            storage_power_mw=float(self._storage["rated_power_mw"]),
            storage_capacity_mwh=float(self._storage["capacity_mwh"]),
        )

    # ------------------------------------------------------------------
    # 数据加载：三级容灾宽表
    # ------------------------------------------------------------------
    def load_market_wide_table(self, *, force_reload: bool = False) -> pd.DataFrame:
        """PostgreSQL → 内存 SQLite → 合成管线，加载建模宽表。"""
        if self._wide_frame is not None and not force_reload:
            return self._wide_frame

        frame = load_vpp_market_frame(self.experiment_config)
        frame = _robust_fill_enhanced_env_features(frame)
        frame = frame.sort_values(self.timestamp_column).reset_index(drop=True)

        min_rows = int(self._global.get("min_modeling_rows", 1280))
        if len(frame) < min_rows:
            raise ValueError(
                f"宽表仅 {len(frame)} 行，低于 min_modeling_rows={min_rows}。"
            )

        self._data_source = str(frame.attrs.get("vpp_data_source", "unknown"))
        self._wide_frame = frame
        return frame

    # ------------------------------------------------------------------
    # 特征词典解析
    # ------------------------------------------------------------------
    def resolve_track_feature_columns(self, track_name: str) -> list[str]:
        """根据 YAML 轨道定义展开特征列名列表。"""
        if track_name not in self._tracks:
            raise KeyError(f"未知轨道: {track_name!r}。可选: {list(self._tracks)}")

        track = self._tracks[track_name]
        feat_cfg = track.get("features", {})
        use = feat_cfg.get("use") or feat_cfg.get("observation")

        columns: list[str] = []
        if isinstance(use, str):
            columns = self._expand_lexicon_group(use)
        elif isinstance(use, list):
            for group in use:
                columns.extend(self._expand_lexicon_group(str(group)))
        else:
            raise ValueError(f"轨道 {track_name} 的 features.use 配置无效: {use!r}")

        # 去重且保持顺序
        seen: set[str] = set()
        ordered: list[str] = []
        for col in columns:
            if col not in seen:
                seen.add(col)
                ordered.append(col)

        return self._enforce_anti_leakage_columns(ordered, track_name)

    def _expand_lexicon_group(self, group: str) -> list[str]:
        if group not in self._lexicon:
            raise KeyError(f"feature_lexicon 缺少分组: {group!r}")

        entry = self._lexicon[group]
        if isinstance(entry, list):
            return list(entry)
        if isinstance(entry, dict) and "includes" in entry:
            cols: list[str] = []
            for sub in entry["includes"]:
                cols.extend(self._expand_lexicon_group(str(sub)))
            return cols
        raise ValueError(f"无法解析特征组 {group!r}: {entry!r}")

    def _enforce_anti_leakage_columns(
        self, columns: list[str], track_name: str
    ) -> list[str]:
        """剔除 contemporaneous 电价；仅保留历史 LMP 滞后与物理微特征。"""
        leaked = [c for c in columns if c in FORBIDDEN_LEAKAGE_COLUMNS]
        if leaked:
            warnings.warn(
                f"轨道 {track_name}: 已从特征列表移除泄露列 {leaked}。",
                UserWarning,
                stacklevel=2,
            )
        safe = [c for c in columns if c not in FORBIDDEN_LEAKAGE_COLUMNS]
        if not safe:
            raise ValueError(
                f"轨道 {track_name} 特征列表在防泄露过滤后为空，请检查 YAML。"
            )
        return safe

    # ------------------------------------------------------------------
    # load_track_data
    # ------------------------------------------------------------------
    def load_track_data(self, track_name: str) -> TrackDataBundle:
        """
        按轨道动态加载特征矩阵，并将 ``price_rt`` / ``spot_price`` 严格剥离出 X。

        标签 y 仅使用 ``global.target_column``（默认 spot_price ← price_rt）。
        """
        if track_name in self._track_cache:
            return self._track_cache[track_name]

        wide = self.load_market_wide_table()
        feature_cols = self.resolve_track_feature_columns(track_name)

        missing = [c for c in feature_cols if c not in wide.columns]
        for col in missing:
            wide[col] = 0.0
            warnings.warn(
                f"轨道 {track_name}: 宽表缺少列 {col!r}，已填 0。",
                UserWarning,
                stacklevel=2,
            )

        if self.target_column not in wide.columns:
            raise ValueError(
                f"宽表缺少标签列 {self.target_column!r}。"
                "请确认 SQL 视图已映射 price_rt → spot_price。"
            )

        work = wide[[self.timestamp_column, self.target_column, *feature_cols]].copy()

        # 二次铁血审计：X 中不得含当期实时价
        forbidden_in_x = [
            c
            for c in feature_cols
            if c in FORBIDDEN_LEAKAGE_COLUMNS or c == self.target_column
        ]
        if forbidden_in_x:
            raise RuntimeError(
                f"防泄露审计失败，特征矩阵仍含禁止列: {forbidden_in_x}"
            )

        # Track D：马尔可夫 ±5% 噪声（仅作用于物理微特征组）
        if track_name == "D_robustness_noise":
            work = self._apply_markov_noise(work, track_name)

        work = work.dropna(subset=[self.target_column, *feature_cols]).reset_index(
            drop=True
        )
        if work.empty:
            raise ValueError(f"轨道 {track_name} dropna 后无有效样本。")

        bundle = TrackDataBundle(
            track_name=track_name,
            feature_columns=feature_cols,
            frame=work,
            target_column=self.target_column,
            timestamp_column=self.timestamp_column,
            data_source=self._data_source,
        )
        self._track_cache[track_name] = bundle
        return bundle

    def _apply_markov_noise(self, df: pd.DataFrame, track_name: str) -> pd.DataFrame:
        """对配置指定的物理特征组施加马尔可夫均匀扰动 ±relative_pct。"""
        track = self._tracks[track_name]
        noise_cfg = track.get("robustness", {}).get("noise", {})
        rel = float(noise_cfg.get("relative_pct", 0.05))
        seed = int(noise_cfg.get("seed", self._global.get("random_seed", 42)))
        groups = noise_cfg.get("apply_to_groups", [])

        cols: list[str] = []
        for g in groups:
            cols.extend(self._expand_lexicon_group(str(g)))
        cols = [c for c in cols if c in df.columns and c not in FORBIDDEN_LEAKAGE_COLUMNS]
        if not cols:
            return df

        out = df.copy()
        rng = np.random.default_rng(seed)
        n = len(out)
        state = rng.uniform(-rel, rel, size=len(cols))

        for i in range(n):
            multiplier = 1.0 + state
            out.loc[out.index[i], cols] = out.loc[out.index[i], cols] * multiplier
            # 马尔可夫惯性：下一步 = 0.85 * 当前扰动 + 0.15 * 新噪声
            state = 0.85 * state + 0.15 * rng.uniform(-rel, rel, size=len(cols))

        return out

    # ------------------------------------------------------------------
    # 时序切分（严禁 shuffle）
    # ------------------------------------------------------------------
    def get_train_test_split(
        self, track_name: str
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Chronological Split：前 ``train_ratio`` 训练，后段测试。

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        bundle = self.load_track_data(track_name)
        split_method = self._global.get("split", {}).get("method", "chronological_ratio")

        if split_method == "calendar_range":
            train_df, test_df = self._calendar_split(bundle.frame)
        else:
            train_df, test_df = temporal_train_test_split(
                bundle.frame, self.train_ratio
            )

        feat = bundle.feature_columns
        X_train = train_df[feat]
        y_train = train_df[bundle.target_column]
        X_test = test_df[feat]
        y_test = test_df[bundle.target_column]
        return X_train, X_test, y_train, y_test

    def _calendar_split(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """可选日历切分（配置 calendar_fallback）。"""
        cal = self._global["split"]["calendar_fallback"]
        ts = pd.to_datetime(frame[self.timestamp_column], utc=True, errors="coerce")
        work = frame.copy()
        work["_ts"] = ts

        train_start = pd.Timestamp(cal["train_start"])
        train_end = pd.Timestamp(cal["train_end"])
        test_start = pd.Timestamp(cal["test_start"])
        test_end = pd.Timestamp(cal["test_end"]) if cal.get("test_end") else None

        train_mask = (work["_ts"] >= train_start) & (work["_ts"] <= train_end)
        test_mask = work["_ts"] >= test_start
        if test_end is not None:
            test_mask &= work["_ts"] <= test_end

        train_df = work.loc[train_mask].drop(columns=["_ts"])
        test_df = work.loc[test_mask].drop(columns=["_ts"])
        if train_df.empty or test_df.empty:
            raise ValueError("日历切分产生空训练集或测试集，请检查 calendar_fallback。")
        return train_df, test_df

    # ------------------------------------------------------------------
    # LightGBM + 储能回测
    # ------------------------------------------------------------------
    def _lgbm_params(self) -> dict[str, Any]:
        p = dict(self._lgbm_cfg)
        p.pop("framework", None)
        return p

    def _storage_backtest(
        self,
        predicted: np.ndarray,
        settlement: np.ndarray,
    ) -> float:
        """按 YAML 储能物理参数执行滚动视界套利回测。"""
        backend, optimize_fn, hourly_pnl_fn = _import_optimizer()

        eta = float(self._storage["round_trip_efficiency"])
        deg = float(self._storage["cycle_degradation_cost_cny_per_mwh"])
        min_soc = float(self._storage["min_soc_ratio"])
        power = float(self._storage["rated_power_mw"])
        cap = float(self._storage["capacity_mwh"])
        horizon = int(self._storage.get("horizon_hours", 24))

        if backend != "power_quant":
            optimize_fn = partial(
                _fallback_optimize_storage_point,
                eta=eta,
                cycle_degradation_cost=deg,
                min_soc_ratio=min_soc,
            )

        def pnl_fn(power_mw: float, price: float, _deg: float) -> float:
            return _fallback_hourly_pnl(power_mw, price, deg)

        return backtest_storage_revenue_yuan(
            predicted,
            settlement,
            backend,
            optimize_fn,
            pnl_fn,
            power_mw=power,
            capacity_mwh=cap,
            horizon_hours=horizon,
        )

    def backtest_hourly_pnl_series(
        self,
        predicted_prices: np.ndarray,
        settlement_prices: np.ndarray,
    ) -> np.ndarray:
        """逐小时储能套利 PnL（与总收益回测同逻辑，供累计曲线绘制）。"""
        backend, optimize_fn, hourly_pnl_fn = _import_optimizer()
        pred = np.asarray(predicted_prices, dtype=float)
        settle = np.asarray(settlement_prices, dtype=float)
        power = float(self._storage["rated_power_mw"])
        cap = float(self._storage["capacity_mwh"])
        eta = float(self._storage["round_trip_efficiency"])
        deg = float(self._storage["cycle_degradation_cost_cny_per_mwh"])
        min_soc = float(self._storage["min_soc_ratio"])
        horizon = int(self._storage.get("horizon_hours", 24))

        if backend != "power_quant":
            optimize_fn = partial(
                _fallback_optimize_storage_point,
                eta=eta,
                cycle_degradation_cost=deg,
                min_soc_ratio=min_soc,
            )

        def pnl_fn(power_mw: float, price: float, _deg: float) -> float:
            return _fallback_hourly_pnl(power_mw, price, deg)

        soc_min = cap * min_soc
        soc_init: Optional[float] = (soc_min + cap) / 2.0
        pnls: list[float] = []

        for start in range(0, len(pred), horizon):
            end = min(start + horizon, len(pred))
            pred_h = pred[start:end]
            settle_h = settle[start:end]
            if len(pred_h) < 2:
                continue
            if backend == "power_quant":
                opt = optimize_fn(
                    pred_h,
                    pred_h,
                    pred_h,
                    power,
                    cap,
                    risk_mode="balanced",
                    eta=eta,
                    cycle_degradation_cost=deg,
                    min_soc_ratio=min_soc,
                    soc_init=soc_init,
                )
            else:
                opt = optimize_fn(pred_h, power, cap, eta=eta, cycle_degradation_cost=deg, min_soc_ratio=min_soc)
            power_arr = np.asarray(opt["power"], dtype=float)
            for t in range(len(settle_h)):
                pnls.append(pnl_fn(power_arr[t], settle_h[t], deg))
            if "soc" in opt and len(opt["soc"]) > 0:
                soc_init = float(np.clip(opt["soc"][-1], soc_min, cap))
        return np.asarray(pnls, dtype=np.float64)

    def _fit_lgbm_and_predict(
        self, track_name: str
    ) -> tuple[np.ndarray, np.ndarray, pd.Series]:
        """训练 LightGBM 并返回 (y_test, pred, test_timestamps)。"""
        bundle = self.load_track_data(track_name)
        X_train, X_test, y_train, y_test = self.get_train_test_split(track_name)
        X_train, y_train = _sanitize_xy(X_train, y_train)
        X_test, y_test = _sanitize_xy(X_test, y_test)

        valid_size = max(1, int(len(X_train) * 0.1))
        X_tr, y_tr = X_train.iloc[:-valid_size], y_train.iloc[:-valid_size]
        X_va, y_va = X_train.iloc[-valid_size:], y_train.iloc[-valid_size:]

        lgb = _import_lightgbm()
        model = lgb.LGBMRegressor(**self._lgbm_params())
        fit_kwargs: dict[str, Any] = {}
        if hasattr(lgb, "early_stopping"):
            rounds = int(self._lgbm_cfg.get("early_stopping_rounds", 30))
            fit_kwargs["eval_set"] = [(X_va, y_va)]
            fit_kwargs["callbacks"] = [
                lgb.early_stopping(stopping_rounds=rounds, verbose=False)
            ]
        model.fit(X_tr, y_tr, **fit_kwargs)
        pred = model.predict(X_test)

        _, test_df = temporal_train_test_split(bundle.frame, self.train_ratio)
        test_df = test_df.dropna(subset=bundle.feature_columns + [bundle.target_column])
        test_aligned = test_df.loc[X_test.index]
        ts = test_aligned[self.timestamp_column].reset_index(drop=True)
        return y_test.values.astype(float), pred.astype(float), ts

    def _record_lgbm_pnl_trace(self, track_name: str) -> None:
        """缓存单轨测试集逐小时 PnL 与累计收益。"""
        y_test, pred, ts = self._fit_lgbm_and_predict(track_name)
        hourly = self.backtest_hourly_pnl_series(pred, y_test)
        n = min(len(ts), len(hourly))
        self._pnl_traces[track_name] = pd.DataFrame(
            {
                self.timestamp_column: ts.iloc[:n].values,
                "hourly_pnl": hourly[:n],
                "cumulative_pnl": np.cumsum(hourly[:n]),
            }
        )

    def _record_ppo_pnl_trace(self) -> None:
        """缓存 PPO 测试集逐小时 PnL。"""
        from ppo_agent import PPOAgent, PPOHyperParams, build_vpp_data_bundle, run_policy_on_env_with_trace
        from vpp_environment import VPPEnvironment, VPP_OBS_COLUMNS

        track_meta = self._tracks[PPO_TRACK_NAME]
        ppo_yaml = track_meta.get("ppo", {})
        policy_path = Path(ppo_yaml.get("policy_artifact", "artifacts/vpp_ppo_policy.pt"))
        if not policy_path.is_absolute():
            policy_path = _PROJECT_ROOT / policy_path

        vpp_bundle = build_vpp_data_bundle(self.experiment_config)
        test_env = VPPEnvironment(
            config=self.experiment_config,
            split="test",
            power_mw=self.experiment_config.storage_power_mw,
            capacity_mwh=self.experiment_config.storage_capacity_mwh,
            data_bundle=vpp_bundle,
        )
        hp = PPOHyperParams(
            total_epochs=int(ppo_yaml.get("total_epochs_web", 5)),
            rollout_steps=int(ppo_yaml.get("rollout_steps", 256)),
            eval_interval=999,
        )
        agent = PPOAgent(len(VPP_OBS_COLUMNS), device="cpu", hparams=hp)
        if policy_path.is_file():
            agent.load(policy_path)
        else:
            from ppo_agent import train_ppo

            agent, _ = train_ppo(
                self.experiment_config,
                hparams=hp,
                device="cpu",
                save_path=policy_path,
                verbose=False,
            )

        _, hourly = run_policy_on_env_with_trace(agent, test_env, deterministic=True)
        ts = vpp_bundle.timestamps.iloc[vpp_bundle.n_train :].reset_index(drop=True)
        n = min(len(ts), len(hourly))
        self._pnl_traces[PPO_TRACK_NAME] = pd.DataFrame(
            {
                self.timestamp_column: ts.iloc[:n].values,
                "hourly_pnl": hourly[:n],
                "cumulative_pnl": np.cumsum(hourly[:n]),
            }
        )

    def build_multitrack_pnl_dataframe(self) -> pd.DataFrame:
        """合并 A / B / D / E 累计财富曲线（宽表，供 Plotly）。"""
        if not self._pnl_traces:
            return pd.DataFrame()

        base_key = PNL_CURVE_TRACKS[0]
        if base_key not in self._pnl_traces:
            return pd.DataFrame()

        out = self._pnl_traces[base_key][[self.timestamp_column]].copy()
        for track in PNL_CURVE_TRACKS:
            if track not in self._pnl_traces:
                continue
            col = f"cum_{self._tracks[track].get('id', track)}"
            trace = self._pnl_traces[track]
            n = min(len(out), len(trace))
            out[col] = trace["cumulative_pnl"].values[:n]
        return out

    def run_lgbm_track(self, track_name: str) -> TrackRunResult:
        """单轨 LightGBM 分位数回归 + 储能套利。"""
        if track_name not in self._tracks:
            raise KeyError(track_name)
        if self._tracks[track_name].get("model_family") != "lightgbm":
            raise ValueError(f"{track_name} 非 LightGBM 轨道。")

        bundle = self.load_track_data(track_name)
        X_train, X_test, y_train, y_test = self.get_train_test_split(track_name)
        X_train, y_train = _sanitize_xy(X_train, y_train)
        X_test, y_test = _sanitize_xy(X_test, y_test)

        min_train = int(self._global.get("min_train_samples", 1000))
        if len(X_train) < min_train:
            raise ValueError(
                f"轨道 {track_name} 训练样本 {len(X_train)} < {min_train}。"
            )

        y_arr, pred, ts = self._fit_lgbm_and_predict(track_name)
        metrics = evaluate_regression(y_arr, pred)
        revenue = self._storage_backtest(pred, y_arr)

        n = min(len(ts), len(pred))
        hourly = self.backtest_hourly_pnl_series(pred[:n], y_arr[:n])
        self._pnl_traces[track_name] = pd.DataFrame(
            {
                self.timestamp_column: ts.iloc[:n].values,
                "hourly_pnl": hourly[:n],
                "cumulative_pnl": np.cumsum(hourly[:n]),
            }
        )

        track_meta = self._tracks[track_name]
        return TrackRunResult(
            track_name=track_name,
            track_id=str(track_meta.get("id", "")),
            model_family="lightgbm",
            n_train=len(X_train),
            n_test=len(X_test),
            rmse=metrics.rmse,
            mae=metrics.mae,
            r2=metrics.r2,
            storage_revenue_cny=revenue,
            data_source=bundle.data_source,
            notes=str(track_meta.get("name", "")),
        )

    # ------------------------------------------------------------------
    # PPO 轨道 E
    # ------------------------------------------------------------------
    def run_ppo_track(self, track_name: str = PPO_TRACK_NAME) -> TrackRunResult:
        """Track E：PyTorch PPO 端到端控制回测（测试集 PnL）。"""
        if track_name not in self._tracks:
            raise KeyError(track_name)

        track_meta = self._tracks[track_name]
        if track_meta.get("model_family") != "pytorch_ppo":
            raise ValueError(f"{track_name} 非 PPO 轨道。")

        try:
            from ppo_agent import (
                BacktestReport,
                PPOAgent,
                PPOHyperParams,
                build_vpp_data_bundle,
                run_policy_on_env,
            )
            from vpp_environment import VPPEnvironment, VPP_OBS_COLUMNS
        except ImportError as exc:
            raise ImportError(
                "PPO 轨道需要 torch、gymnasium、ppo_agent、vpp_environment。"
            ) from exc

        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("OMP_NUM_THREADS", "1")

        ppo_yaml = track_meta.get("ppo", {})
        policy_path = Path(
            ppo_yaml.get(
                "policy_artifact",
                self.cfg.get("artifacts", {}).get(
                    "ppo_checkpoint", "artifacts/vpp_ppo_policy.pt"
                ),
            )
        )
        if not policy_path.is_absolute():
            policy_path = _PROJECT_ROOT / policy_path

        vpp_bundle = build_vpp_data_bundle(self.experiment_config)
        test_env = VPPEnvironment(
            config=self.experiment_config,
            split="test",
            power_mw=self.experiment_config.storage_power_mw,
            capacity_mwh=self.experiment_config.storage_capacity_mwh,
            data_bundle=vpp_bundle,
        )

        obs_dim = len(VPP_OBS_COLUMNS)
        hp = PPOHyperParams(
            hidden_dim=int(ppo_yaml.get("hidden_dim", 256)),
            lr=float(ppo_yaml.get("learning_rate", 3e-4)),
            gamma=float(ppo_yaml.get("gamma", 0.99)),
            gae_lambda=float(ppo_yaml.get("gae_lambda", 0.95)),
            clip_eps=float(ppo_yaml.get("clip_eps", 0.2)),
            entropy_coef=float(ppo_yaml.get("entropy_coef", 0.01)),
            total_epochs=int(ppo_yaml.get("total_epochs_cli", 60)),
            rollout_steps=int(ppo_yaml.get("rollout_steps", 512)),
            eval_interval=999,
        )

        if policy_path.is_file():
            agent = PPOAgent(obs_dim, device="cpu", hparams=hp)
            agent.load(policy_path)
        else:
            from ppo_agent import train_ppo

            warnings.warn(
                f"未找到预训练权重 {policy_path}，将执行 CLI 训练（耗时较长）。",
                UserWarning,
                stacklevel=2,
            )
            agent, _ = train_ppo(
                self.experiment_config,
                hparams=hp,
                device="cpu",
                save_path=policy_path,
                verbose=True,
            )

        ppo_report: BacktestReport = run_policy_on_env(agent, test_env, deterministic=True)
        try:
            if PPO_TRACK_NAME not in self._pnl_traces:
                self._record_ppo_pnl_trace()
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"PPO PnL 曲线缓存失败: {exc}", UserWarning)

        return TrackRunResult(
            track_name=track_name,
            track_id=str(track_meta.get("id", "E")),
            model_family="pytorch_ppo",
            n_train=vpp_bundle.n_train,
            n_test=vpp_bundle.n_test,
            rmse=None,
            mae=None,
            r2=None,
            storage_revenue_cny=float(ppo_report.total_revenue_yuan),
            profitable_hour_ratio=float(ppo_report.profitable_hour_ratio),
            alpha_vs_idle_pct=float(ppo_report.alpha_vs_idle_pct),
            data_source=ppo_report.data_source,
            notes=str(track_meta.get("name", "")),
        )

    # ------------------------------------------------------------------
    # 全矩阵擂台
    # ------------------------------------------------------------------
    def run_all_tracks(
        self,
        *,
        include_ppo: bool = True,
        skip_ppo_if_missing_torch: bool = True,
    ) -> pd.DataFrame:
        """
        依次运行 Track A–D（LightGBM）及 Track E（PPO），汇总对照矩阵。
        """
        self._results.clear()
        print("\n" + "=" * 72)
        print("  MEL-F 复线实验矩阵 — Multi-Track Loader".center(72))
        print("=" * 72)
        print(f"  配置: {self.config_path.name}")
        self.load_market_wide_table()
        print(f"  数据源: {self._data_source}")

        for track_name in LGBM_TRACK_ORDER:
            print(f"\n▶ 运行轨道 {track_name} ...")
            try:
                result = self.run_lgbm_track(track_name)
                self._results.append(result)
                print(
                    f"   RMSE={result.rmse:.2f} | R²={result.r2:.4f} | "
                    f"储能收益=¥{result.storage_revenue_cny:,.0f}"
                )
            except Exception as exc:
                warnings.warn(f"轨道 {track_name} 失败: {exc}", UserWarning)
                self._results.append(
                    TrackRunResult(
                        track_name=track_name,
                        track_id=self._tracks.get(track_name, {}).get("id", "?"),
                        model_family="lightgbm",
                        n_train=0,
                        n_test=0,
                        rmse=None,
                        mae=None,
                        r2=None,
                        storage_revenue_cny=0.0,
                        notes=f"FAILED: {exc}",
                    )
                )

        if include_ppo:
            print(f"\n▶ 运行轨道 {PPO_TRACK_NAME} (PyTorch PPO) ...")
            try:
                result = self.run_ppo_track(PPO_TRACK_NAME)
                self._results.append(result)
                print(
                    f"   储能收益=¥{result.storage_revenue_cny:,.0f} | "
                    f"盈利小时占比={result.profitable_hour_ratio:.1%} | "
                    f"Alpha vs 空仓={result.alpha_vs_idle_pct:+.2f}%"
                )
            except ImportError as exc:
                if skip_ppo_if_missing_torch:
                    warnings.warn(f"PPO 轨道跳过: {exc}", UserWarning)
                else:
                    raise
            except Exception as exc:
                warnings.warn(f"PPO 轨道失败: {exc}", UserWarning)
                self._results.append(
                    TrackRunResult(
                        track_name=PPO_TRACK_NAME,
                        track_id="E",
                        model_family="pytorch_ppo",
                        n_train=0,
                        n_test=0,
                        rmse=None,
                        mae=None,
                        r2=None,
                        storage_revenue_cny=0.0,
                        notes=f"FAILED: {exc}",
                    )
                )

        _ = self.build_multitrack_pnl_dataframe()
        return self.build_comparison_matrix()

    def build_comparison_matrix(self) -> pd.DataFrame:
        """将全部轨道结果整合为标准 DataFrame。"""
        rows = []
        for r in self._results:
            rows.append(
                {
                    "track_name": r.track_name,
                    "track_id": r.track_id,
                    "model_family": r.model_family,
                    "n_train": r.n_train,
                    "n_test": r.n_test,
                    "rmse": r.rmse,
                    "mae": r.mae,
                    "r2": r.r2,
                    "storage_revenue_cny": r.storage_revenue_cny,
                    "profitable_hour_ratio": r.profitable_hour_ratio,
                    "alpha_vs_idle_pct": r.alpha_vs_idle_pct,
                    "data_source": r.data_source,
                    "notes": r.notes,
                }
            )
        return pd.DataFrame(rows)

    def export_comparison_matrix(
        self,
        df: Optional[pd.DataFrame] = None,
        export_path: Optional[Path | str] = None,
    ) -> Path:
        """导出全矩阵 CSV。"""
        matrix = df if df is not None else self.build_comparison_matrix()
        if export_path is None:
            out_dir = Path(self.cfg.get("artifacts", {}).get("output_dir", "artifacts"))
            export_path = out_dir / "track_comparison_matrix.csv"
        path = Path(export_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"\n✓ 复线对照矩阵已导出 → {path}")
        return path


# Web / CLI 统一别名
ExperimentRunner = ExperimentLoader


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MEL-F 复线实验运行器")
    p.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "experiment_config.yaml",
        help="YAML 配置路径",
    )
    p.add_argument("--demo", action="store_true", help="强制合成大样本管线")
    p.add_argument(
        "--export",
        type=Path,
        default=None,
        help="对照矩阵 CSV 输出路径",
    )
    p.add_argument("--no-ppo", action="store_true", help="跳过 PPO 轨道 E")
    return p.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    loader = ExperimentLoader(
        args.config,
        production_db=not args.demo,
        demo=args.demo,
    )
    matrix = loader.run_all_tracks(include_ppo=not args.no_ppo)
    print("\n" + matrix.to_string(index=False))
    loader.export_comparison_matrix(matrix, args.export)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
