# -*- coding: utf-8 -*-
"""
环境微特征电力现货预测 — 对比实验与精度 / 经济价值评估脚本。

闭环流程
--------
1. 特征工程宽表合成（``align_and_merge_all``）→ 按时间 80/20 划分（无未来信息泄露）
2. 基准 vs 增强 LightGBM 分位数回归（P50）擂台
3. 测试集统计精度（RMSE / MAE / R²）+ 储能套利回测（元）
4. 终端输出专业实验报告与增强模型环境因子重要性排名

用法
----
    python run_experiment.py --demo
    python run_experiment.py --data-dir ./data
"""

from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 项目根目录加入 sys.path，便于导入本仓库模块及主项目 optimizer
# ---------------------------------------------------------------------------
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
# 列名常量（与 FeatureSchema 默认一致，可按主项目覆盖）
# ---------------------------------------------------------------------------
COL_TIMESTAMP = "timestamp"
COL_TARGET = "spot_price"

# 基准模型特征：历史电价（滞后）、系统负荷、新能源预测、常规气象
BENCHMARK_FEATURE_COLUMNS: tuple[str, ...] = (
    "spot_price_lag_1h",
    "spot_price_lag_24h",
    "total_load",
    "wind_forecast",
    "solar_forecast",
    "temperature",
    "wind_speed",
)

# 增强模型新增：空间加权辐射、辐射突变率、污染×负荷交叉、水温时滞
ENHANCED_ENV_COLUMNS: tuple[str, ...] = (
    "effective_pv_radiation",
    "radiation_mutation_rate",
    "pm25_load_cross_prov_weighted",
    "water_temp_lag_1h",
    "water_temp_lag_3h",
    "water_temp_lag_6h",
    "water_temp_lag_24h",
)

# 统一 LightGBM 超参数（两套模型完全一致，仅特征不同）
LGBM_PARAMS: dict[str, Any] = {
    "objective": "quantile",
    "alpha": 0.5,  # 50% 分位数 → 中位数电价
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 8,
    "num_leaves": 64,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": 1,
    "verbose": -1,
}

# 储能回测默认参数（与主项目 optimizer 典型配置对齐）
STORAGE_POWER_MW: float = 50.0
STORAGE_CAPACITY_MWH: float = 200.0
STORAGE_ETA: float = 0.95
STORAGE_CYCLE_COST: float = 50.0
STORAGE_MIN_SOC_RATIO: float = 0.10

_WATER_LAG_HOURS: tuple[int, ...] = (1, 3, 6, 24)
_PRICE_LAG_HOURS: tuple[int, ...] = (1, 24)


@dataclass
class ExperimentConfig:
    """实验配置。"""

    train_ratio: float = 0.80
    data_dir: Optional[Path] = None
    demo: bool = False
    schema: FeatureSchema = field(default_factory=FeatureSchema)
    storage_power_mw: float = STORAGE_POWER_MW
    storage_capacity_mwh: float = STORAGE_CAPACITY_MWH


@dataclass
class ModelMetrics:
    """单模型测试集精度指标。"""

    rmse: float
    mae: float
    r2: float


@dataclass
class ExperimentReport:
    """完整实验报告载体。"""

    n_train: int
    n_test: int
    benchmark_metrics: ModelMetrics
    enhanced_metrics: ModelMetrics
    benchmark_revenue_yuan: float
    enhanced_revenue_yuan: float
    env_importance: pd.DataFrame
    # 供 Web 可视化：测试集时序预测与增强模型全量特征重要性
    test_forecast: Optional[pd.DataFrame] = None
    enhanced_full_importance: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# 依赖导入（带友好报错）
# ---------------------------------------------------------------------------
def _import_lightgbm() -> Any:
    try:
        import lightgbm as lgb  # type: ignore

        return lgb
    except ImportError as exc:
        raise ImportError(
            "未安装 lightgbm。请执行: pip install lightgbm"
        ) from exc


def _import_optimizer() -> tuple[str, Any, Callable[[float, float, float], float]]:
    """
    尝试加载主项目 ``power_quant/optimizer.py`` 中的储能优化函数。

    返回 (backend_name, optimize_callable, hourly_pnl_fn)。
    若不可用，回退到本脚本内置 scipy 实现。
    """
    candidates = [
        _PROJECT_ROOT / "optimizer.py",
        _PROJECT_ROOT.parent / "power_quant" / "optimizer.py",
    ]
    for path in candidates:
        parent = str(path.parent)
        if path.is_file() and parent not in sys.path:
            sys.path.insert(0, parent)

    try:
        from optimizer import optimize_storage_robust  # type: ignore

        def hourly_pnl(power_mw: float, price: float, deg_cost: float) -> float:
            return float(price * power_mw - deg_cost * max(-power_mw, 0.0))

        return "power_quant", optimize_storage_robust, hourly_pnl
    except ImportError:
        warnings.warn(
            "未找到主项目 optimizer.py，将使用内置 scipy 储能优化回退实现。",
            UserWarning,
            stacklevel=2,
        )
        return "fallback", _fallback_optimize_storage_point, _fallback_hourly_pnl


def _fallback_hourly_pnl(power_mw: float, price: float, deg_cost: float) -> float:
    return float(price * power_mw - deg_cost * max(-power_mw, 0.0))


def _fallback_optimize_storage_point(
    prices: np.ndarray,
    power_mw: float,
    capacity_mwh: float,
    eta: float = STORAGE_ETA,
    cycle_degradation_cost: float = STORAGE_CYCLE_COST,
    min_soc_ratio: float = STORAGE_MIN_SOC_RATIO,
) -> dict:
    """轻量级储能套利优化（SLSQP），接口与主项目 optimize_storage_point 对齐。"""
    from scipy.optimize import minimize

    p = np.asarray(prices, dtype=float)
    n = len(p)
    soc_min = capacity_mwh * min_soc_ratio
    soc_max = capacity_mwh
    soc_init = (soc_min + soc_max) / 2.0

    def soc_path(power: np.ndarray) -> np.ndarray:
        soc = np.zeros(n + 1)
        soc[0] = soc_init
        for t in range(n):
            if power[t] >= 0:
                soc[t + 1] = soc[t] - power[t]
            else:
                soc[t + 1] = soc[t] - power[t] * eta
        return soc

    def objective(power: np.ndarray) -> float:
        rev = float(np.dot(p, power))
        deg = cycle_degradation_cost * float(np.sum(np.maximum(-power, 0.0)))
        return -(rev - deg)

    bounds = [(-power_mw, power_mw)] * n
    constraints = []
    for t in range(n + 1):

        def lo(pw, idx=t):
            return soc_path(pw)[idx] - soc_min

        def hi(pw, idx=t):
            return soc_max - soc_path(pw)[idx]

        constraints.append({"type": "ineq", "fun": lo})
        constraints.append({"type": "ineq", "fun": hi})

    res = minimize(
        objective,
        np.zeros(n),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 400, "ftol": 1e-6},
    )
    p_opt = res.x if res.success else np.zeros(n)
    return {"power": p_opt, "soc": soc_path(p_opt), "success": res.success}


# ---------------------------------------------------------------------------
# 数据加载与特征衍生（防泄露：目标列不进入滞后以外的 contemporaneous 特征）
# ---------------------------------------------------------------------------
def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if path.is_file():
        return pd.read_csv(path)
    return None


def load_raw_tables(data_dir: Path) -> dict[str, pd.DataFrame]:
    """从 data_dir 加载各源表；缺失文件抛出明确错误。"""
    mapping = {
        "market": "market.csv",
        "era5": "era5.csv",
        "pv_stations": "pv_stations.csv",
        "aqi": "aqi.csv",
        "load_base": "load_base.csv",
        "water": "water.csv",
    }
    tables: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for key, fname in mapping.items():
        fp = data_dir / fname
        df = _read_csv_if_exists(fp)
        if df is None:
            missing.append(str(fp))
        else:
            tables[key] = df
    if missing:
        raise FileNotFoundError(
            f"数据目录 {data_dir} 缺少文件: {missing}。"
            "请补齐 CSV 或使用 --demo 运行合成数据实验。"
        )
    return tables


def build_synthetic_raw_tables(n_hours: int = 720) -> dict[str, pd.DataFrame]:
    """
    构造可跑通全流程的合成数据（默认 30 天 × 24 小时）。

    用于无真实 CSV 时的闭环自检；统计与经济结论仅供管线验证。
    """
    hours = pd.date_range("2024-01-01", periods=n_hours, freq="h")
    rng = np.random.default_rng(42)

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
                        "ssrd": 150.0 + 80.0 * np.sin(i / 24.0) + rng.normal(0, 10),
                    }
                )

    cities = ["city_a", "city_b", "city_c"]
    aqi_rows, load_rows = [], []
    for i, ts in enumerate(hours):
        for j, city in enumerate(cities):
            aqi_rows.append(
                {
                    "timestamp": ts,
                    "city": city,
                    "pm25": 25.0 + 10 * np.sin(i / 12.0) + j * 3 + rng.normal(0, 2),
                    "no2": 18.0 + 5 * np.cos(i / 18.0) + j * 2,
                    "aod": 0.25 + 0.05 * j + 0.02 * np.sin(i / 24.0),
                }
            )
            load_rows.append(
                {
                    "timestamp": ts,
                    "city": city,
                    "industrial_load_base": 400.0 + j * 80 + 50 * np.sin(i / 24.0),
                }
            )

    base_price = 320.0 + 60.0 * np.sin(np.arange(n_hours) / 24.0)
    market = pd.DataFrame(
        {
            "timestamp": hours,
            "spot_price": base_price + rng.normal(0, 15, n_hours),
            "total_load": 11000.0 + 1500 * np.sin(np.arange(n_hours) / 24.0) + rng.normal(0, 100, n_hours),
            "wind_forecast": 800.0 + 200 * rng.standard_normal(n_hours),
            "solar_forecast": np.clip(600.0 + 300 * np.sin(np.arange(n_hours) / 24.0), 0, None),
            "temperature": 12.0 + 8.0 * np.sin(np.arange(n_hours) / 24.0) + rng.normal(0, 0.5, n_hours),
            "wind_speed": 4.0 + 2.0 * rng.standard_normal(n_hours),
        }
    )

    water = pd.DataFrame(
        {
            "timestamp": hours,
            "water_temp": 10.0 + 6.0 * np.sin(np.arange(n_hours) / 24.0) + rng.normal(0, 0.3, n_hours),
        }
    )

    return {
        "market": market,
        "era5": pd.DataFrame(era5_rows),
        "pv_stations": pd.DataFrame(
            {
                "station_id": ["pv1", "pv2", "pv3"],
                "lat": [30.1, 30.35, 30.45],
                "lon": [120.1, 120.3, 120.45],
                "capacity_mw": [80.0, 120.0, 200.0],
            }
        ),
        "aqi": pd.DataFrame(aqi_rows),
        "load_base": pd.DataFrame(load_rows),
        "water": water,
    }


def build_enhanced_wide_table(
    tables: dict[str, pd.DataFrame],
    schema: Optional[FeatureSchema] = None,
) -> pd.DataFrame:
    """
    调用特征工程流水线生成总宽表。

    严格走 ``process_radiation_features`` → ``process_pollution_interaction``
    → ``align_and_merge_all``，与生产特征一致。
    """
    schema = schema or FeatureSchema()
    df_rad = process_radiation_features(tables["era5"], tables["pv_stations"], schema=schema)
    df_pol = process_pollution_interaction(tables["aqi"], tables["load_base"], schema=schema)
    return align_and_merge_all(
        tables["market"],
        df_rad,
        df_pol,
        tables["water"],
        schema=schema,
    )


def add_model_ready_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    在宽表上追加建模用滞后特征。

    - 历史电价滞后：避免将 contemporaneous ``spot_price`` 作为特征（泄露）
    - 水温时滞：增强模型专用，仅使用过去观测
    """
    out = df.copy()
    out = out.sort_values(COL_TIMESTAMP).reset_index(drop=True)

    if COL_TARGET not in out.columns:
        raise ValueError(f"宽表必须包含目标列 '{COL_TARGET}'。")

    for lag in _PRICE_LAG_HOURS:
        out[f"spot_price_lag_{lag}h"] = out[COL_TARGET].shift(lag)

    if "water_temp" in out.columns:
        for lag in _WATER_LAG_HOURS:
            out[f"water_temp_lag_{lag}h"] = out["water_temp"].shift(lag)

    return out


def temporal_train_test_split(
    df: pd.DataFrame,
    train_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    按时间顺序划分训练 / 测试集，杜绝随机打乱造成未来信息泄露。
    """
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio 必须在 (0, 1) 内。")
    df = df.sort_values(COL_TIMESTAMP).reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)
    if split_idx < 1 or split_idx >= len(df):
        raise ValueError(
            f"样本量 {len(df)} 过小或 train_ratio={train_ratio} 不合理，"
            "无法得到非空的训练集与测试集。"
        )
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def _sanitize_xy(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """剔除含 NaN / inf 的样本行。"""
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1) & y.notna()
    return X.loc[mask], y.loc[mask]


def train_quantile_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: Optional[pd.DataFrame] = None,
    y_valid: Optional[pd.Series] = None,
) -> Any:
    """训练 LightGBM P50 分位数回归模型。"""
    lgb = _import_lightgbm()
    model = lgb.LGBMRegressor(**LGBM_PARAMS)

    fit_kwargs: dict[str, Any] = {}
    if X_valid is not None and y_valid is not None and len(X_valid) > 0:
        fit_kwargs["eval_set"] = [(X_valid, y_valid)]
        fit_kwargs["callbacks"] = [lgb.early_stopping(stopping_rounds=30, verbose=False)]

    model.fit(X_train, y_train, **fit_kwargs)
    return model


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> ModelMetrics:
    """计算 RMSE / MAE / R²。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err**2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
    return ModelMetrics(rmse=rmse, mae=mae, r2=r2)


def backtest_storage_revenue_yuan(
    predicted_prices: np.ndarray,
    settlement_prices: np.ndarray,
    backend: str,
    optimize_fn: Callable[..., dict],
    hourly_pnl_fn: Callable[[float, float, float], float],
    power_mw: float = STORAGE_POWER_MW,
    capacity_mwh: float = STORAGE_CAPACITY_MWH,
    horizon_hours: int = 24,
) -> float:
    """
    储能套利回测：用**预测电价**做优化，用**实际结算电价**计算收益。

    按 ``horizon_hours``（默认 24h）滚动求解，SOC 在日界间传递，兼顾算力与日内交易节奏。
    更准的电价预测 → 更优充放电计划 → 更高的实测套利现金流。
    """
    pred = np.asarray(predicted_prices, dtype=float)
    settle = np.asarray(settlement_prices, dtype=float)
    if len(pred) != len(settle) or len(pred) == 0:
        raise ValueError("预测电价与结算电价长度必须一致且非空。")
    if np.any(~np.isfinite(pred)) or np.any(~np.isfinite(settle)):
        raise ValueError("电价序列含 NaN/inf，无法优化。")

    soc_min = capacity_mwh * STORAGE_MIN_SOC_RATIO
    soc_init: Optional[float] = (soc_min + capacity_mwh) / 2.0
    total = 0.0

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
            total += hourly_pnl_fn(power[t], settle_h[t], STORAGE_CYCLE_COST)

        if "soc" in opt and len(opt["soc"]) > 0:
            soc_init = float(np.clip(opt["soc"][-1], soc_min, capacity_mwh))

    return float(total)


def extract_env_feature_importance(
    model: Any,
    feature_names: list[str],
    env_columns: tuple[str, ...],
) -> pd.DataFrame:
    """提取增强模型中环境因子的重要性排名。"""
    importances = np.asarray(model.feature_importances_, dtype=float)
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    env_set = set(env_columns)
    df_env = df_imp[df_imp["feature"].isin(env_set)].copy()
    df_env = df_env.sort_values("importance", ascending=False).reset_index(drop=True)
    df_env["rank"] = np.arange(1, len(df_env) + 1)
    return df_env


def _pct_improvement(baseline: float, enhanced: float) -> float:
    """正值表示增强模型相对基准的改善（误差类指标 baseline>enhanced 为改善）。"""
    if abs(baseline) < 1e-12:
        return 0.0
    return (baseline - enhanced) / abs(baseline) * 100.0


def _pct_alpha(benchmark: float, enhanced: float) -> float:
    """储能收益增量 Alpha（%）。"""
    if abs(benchmark) < 1e-12:
        return 0.0
    return (enhanced - benchmark) / abs(benchmark) * 100.0


def print_experiment_report(report: ExperimentReport) -> None:
    """在终端打印专业对比实验报告。"""
    bm, em = report.benchmark_metrics, report.enhanced_metrics
    rmse_imp = _pct_improvement(bm.rmse, em.rmse)
    mae_imp = _pct_improvement(bm.mae, em.mae)
    rev_alpha = _pct_alpha(report.benchmark_revenue_yuan, report.enhanced_revenue_yuan)

    width = 72
    print("\n" + "=" * width)
    print("  环境微特征 × 电力现货预测 — 对比实验报告".center(width))
    print("=" * width)
    print(f"  训练样本数: {report.n_train:<8d}  测试样本数: {report.n_test}")
    print("-" * width)
    print(f"  {'指标':<22} {'基准模型':>14} {'增强模型':>14} {'变化':>12}")
    print("-" * width)
    print(f"  {'RMSE (元/MWh)':<22} {bm.rmse:>14.4f} {em.rmse:>14.4f} {rmse_imp:>+11.2f}%")
    print(f"  {'MAE (元/MWh)':<22} {bm.mae:>14.4f} {em.mae:>14.4f} {mae_imp:>+11.2f}%")
    print(f"  {'R2':<22} {bm.r2:>14.4f} {em.r2:>14.4f} {'—':>12}")
    print("-" * width)
    print(f"  {'储能套利收益 (元)':<22} {report.benchmark_revenue_yuan:>14.2f} "
          f"{report.enhanced_revenue_yuan:>14.2f} {rev_alpha:>+11.2f}%")
    print("=" * width)
    print("\n【结论摘要】")
    rmse_verb = "降低" if rmse_imp >= 0 else "上升"
    mae_verb = "降低" if mae_imp >= 0 else "上升"
    rev_verb = "提升" if rev_alpha >= 0 else "下降"
    print(f"  · 加入环境微特征后，测试集 RMSE {rmse_verb} {abs(rmse_imp):.2f}%")
    print(f"  · 测试集 MAE  {mae_verb} {abs(mae_imp):.2f}%")
    print(f"  · 储能模拟套利收益{rev_verb} {abs(rev_alpha):.2f}%（增量 Alpha）")
    print("\n【增强模型 · 环境因子重要性排名】")
    if report.env_importance.empty:
        print("  （无环境因子进入特征列表）")
    else:
        for _, row in report.env_importance.iterrows():
            print(f"  {int(row['rank']):>2}. {row['feature']:<35} importance={row['importance']:.4f}")
    print("=" * width + "\n")


def run_experiment(config: ExperimentConfig) -> ExperimentReport:
    """执行完整对比实验。"""
    # ---------- 1. 加载原始表 & 特征工程宽表 ----------
    if config.demo:
        tables = build_synthetic_raw_tables()
    else:
        if config.data_dir is None:
            raise ValueError("非 demo 模式必须指定 --data-dir。")
        tables = load_raw_tables(config.data_dir)

    df_wide = build_enhanced_wide_table(tables, schema=config.schema)
    df_wide = add_model_ready_features(df_wide)

    benchmark_cols = list(BENCHMARK_FEATURE_COLUMNS)
    enhanced_cols = benchmark_cols + list(ENHANCED_ENV_COLUMNS)
    required_cols = [COL_TARGET] + enhanced_cols

    missing = [c for c in required_cols if c not in df_wide.columns]
    if missing:
        raise ValueError(
            f"宽表缺少建模列: {missing}。"
            "请检查 market / 环境特征工程输出或合成数据字段。"
        )

    # 在全表上 drop 滞后 NaN 后，再按时间切分（滞后仅依赖过去，不泄露未来）
    feature_frame = df_wide[required_cols + [COL_TIMESTAMP]].copy()
    feature_frame = feature_frame.dropna(subset=required_cols).reset_index(drop=True)

    train_df, test_df = temporal_train_test_split(feature_frame, config.train_ratio)

    X_train_b, y_train = _sanitize_xy(train_df[benchmark_cols], train_df[COL_TARGET])
    X_train_e, y_train = _sanitize_xy(train_df[enhanced_cols], train_df[COL_TARGET])
    X_test_b, y_test = _sanitize_xy(test_df[benchmark_cols], test_df[COL_TARGET])
    X_test_e, y_test = _sanitize_xy(test_df[enhanced_cols], test_df[COL_TARGET])

    # 用训练集末尾作 early stopping 验证（仍在训练时段内，不触碰测试集）
    valid_size = max(1, int(len(X_train_e) * 0.1))
    X_tr_e, X_va_e = X_train_e.iloc[:-valid_size], X_train_e.iloc[-valid_size:]
    y_tr_e, y_va_e = y_train.iloc[:-valid_size], y_train.iloc[-valid_size:]
    X_tr_b, X_va_b = X_train_b.iloc[:-valid_size], X_train_b.iloc[-valid_size:]
    y_tr_b, y_va_b = y_train.iloc[:-valid_size], y_train.iloc[-valid_size:]

    # ---------- 2. 训练两套 LightGBM 擂台 ----------
    model_bench = train_quantile_lgbm(X_tr_b, y_tr_b, X_va_b, y_va_b)
    model_enh = train_quantile_lgbm(X_tr_e, y_tr_e, X_va_e, y_va_e)

    pred_bench = model_bench.predict(X_test_b)
    pred_enh = model_enh.predict(X_test_e)

    # ---------- 3. 统计精度 ----------
    metrics_bench = evaluate_regression(y_test.values, pred_bench)
    metrics_enh = evaluate_regression(y_test.values, pred_enh)

    # ---------- 4. 储能经济价值回测 ----------
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


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="环境微特征电力现货预测 — 基准 vs 增强对比实验",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="使用合成数据跑通闭环（无需 CSV）",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="存放 market.csv / era5.csv / ... 的目录",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="按时间顺序的训练集占比（默认 0.80）",
    )
    parser.add_argument(
        "--storage-power-mw",
        type=float,
        default=STORAGE_POWER_MW,
        help="储能额定功率 (MW)",
    )
    parser.add_argument(
        "--storage-capacity-mwh",
        type=float,
        default=STORAGE_CAPACITY_MWH,
        help="储能容量 (MWh)",
    )
    args = parser.parse_args()
    return ExperimentConfig(
        demo=args.demo,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        train_ratio=args.train_ratio,
        storage_power_mw=args.storage_power_mw,
        storage_capacity_mwh=args.storage_capacity_mwh,
    )


def main() -> None:
    config = parse_args()
    if not config.demo and config.data_dir is None:
        print("未指定 --data-dir，将使用 --demo 合成数据模式。")
        config = ExperimentConfig(demo=True, train_ratio=config.train_ratio)

    try:
        report = run_experiment(config)
        print_experiment_report(report)
    except Exception as exc:
        print(f"\n[实验失败] {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
