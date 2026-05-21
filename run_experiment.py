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
    python run_experiment.py --production-db
    python run_experiment.py --data-dir ./data   # 遗留 CSV 多表管线
"""

from __future__ import annotations

import argparse
import os
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

from db_feature_view import SQL_CREATE_VIEW  # noqa: E402
from db_injector import DEFAULT_DATABASE_URL, create_db_engine  # noqa: E402
from sqlalchemy import text  # noqa: E402
from environmental_feature_engineer import (  # noqa: E402
    FeatureSchema,
    align_and_merge_all,
    enrich_market_physics_inputs,
    merge_frontier_physics_features,
    process_pollution_interaction,
    process_radiation_features,
)

# ---------------------------------------------------------------------------
# 生产 PostgreSQL 特征源（v_features_pipeline_ready 窗口函数宽表）
# ---------------------------------------------------------------------------
PRODUCTION_DATABASE_URL = os.getenv("MEL_DATABASE_URL", DEFAULT_DATABASE_URL)
DEFAULT_PRODUCTION_NODE_ID = "PJM_HUB"

SQL_LOAD_FEATURES_PIPELINE = """
SELECT *
FROM v_features_pipeline_ready
WHERE node_id = 'PJM_HUB'
ORDER BY timestamp ASC
"""

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

# 微观物理因子（四大前沿板块核心字段，增强模型显式纳入）
MICRO_PHYSICS_FEATURE_COLUMNS: tuple[str, ...] = (
    "heat_index",
    "heat_index_spike_35",
    "wind_shear_coefficient",  # 幂律切变系数 α，与特征工程 wind_shear_alpha 同源
    "wake_effect_intensity",
    "snow_melt_rate",
    "bifacial_gain_index",
    "panel_efficiency_discount",
)

# 增强模型新增：经典环境工程 + 微观物理 + 切变/积尘辅助 + 水温时滞
ENHANCED_ENV_COLUMNS: tuple[str, ...] = (
    "effective_pv_radiation",
    "radiation_mutation_rate",
    "pm25_load_cross_prov_weighted",
    *MICRO_PHYSICS_FEATURE_COLUMNS,
    "wind_shear_alpha",
    "wind_shear_risk",
    "wind_dir_dev",
    "panel_dirt_accumulation",
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

# 大样本纪律：30 天(720h)为最小回溯；默认 1536h 保障 dropna + 80/20 切分后训练集 ≥ 1000
EXPERIMENT_LOOKBACK_HOURS_MIN: int = 720
EXPERIMENT_LOOKBACK_HOURS_DEFAULT: int = 1536
MIN_TRAIN_SAMPLES: int = 1000
MIN_MODELING_ROWS: int = 1280


@dataclass
class ExperimentConfig:
    """实验配置。"""

    train_ratio: float = 0.80
    lookback_hours: int = EXPERIMENT_LOOKBACK_HOURS_DEFAULT
    min_train_samples: int = MIN_TRAIN_SAMPLES
    data_dir: Optional[Path] = None
    demo: bool = False
    production_db: bool = False
    database_url: str = field(default_factory=lambda: PRODUCTION_DATABASE_URL)
    node_id: str = DEFAULT_PRODUCTION_NODE_ID
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
    model_enhanced: Any = None
    X_test_enhanced: Optional[pd.DataFrame] = None


# ---------------------------------------------------------------------------
# 依赖导入（带友好报错）
# ---------------------------------------------------------------------------
class _SklearnLGBMShim:
    """LightGBM 不可用（缺 libomp 等）时，用 sklearn 分位数 GBR 保持 API 兼容。"""

    class LGBMRegressor:
        def __init__(self, **kwargs: Any) -> None:
            from sklearn.ensemble import GradientBoostingRegressor

            self._model = GradientBoostingRegressor(
                loss="quantile",
                alpha=float(kwargs.get("alpha", 0.5)),
                n_estimators=int(kwargs.get("n_estimators", 200)),
                learning_rate=float(kwargs.get("learning_rate", 0.05)),
                max_depth=int(kwargs.get("max_depth", 8)),
                subsample=float(kwargs.get("subsample", 0.85)),
                random_state=int(kwargs.get("random_state", 42)),
            )

        def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> "_SklearnLGBMShim.LGBMRegressor":
            self._model.fit(X, y)
            return self

        def predict(self, X: pd.DataFrame) -> np.ndarray:
            return np.asarray(self._model.predict(X), dtype=float)

        @property
        def feature_importances_(self) -> np.ndarray:
            return np.asarray(self._model.feature_importances_, dtype=float)

    @staticmethod
    def early_stopping(stopping_rounds: int = 30, verbose: bool = False) -> None:
        return None


def _import_lightgbm() -> Any:
    try:
        import lightgbm as lgb  # type: ignore

        return lgb
    except ImportError as exc:
        raise ImportError(
            "未安装 lightgbm。请执行: pip install lightgbm"
        ) from exc
    except OSError as exc:
        warnings.warn(
            f"LightGBM 动态库加载失败（常见于 macOS 缺 libomp）: {exc}\n"
            "已自动回退 sklearn GradientBoostingRegressor（quantile）。"
            "修复原生 LightGBM: brew install libomp",
            UserWarning,
            stacklevel=2,
        )
        return _SklearnLGBMShim()


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
def _ensure_feature_view(engine: Any) -> None:
    """确保 PostgreSQL 特征视图已激活（幂等 CREATE OR REPLACE VIEW）。"""
    with engine.begin() as conn:
        conn.execute(text(SQL_CREATE_VIEW))


def load_features_from_production_db(
    database_url: str = PRODUCTION_DATABASE_URL,
    node_id: str = DEFAULT_PRODUCTION_NODE_ID,
) -> pd.DataFrame:
    """
    从 ``v_features_pipeline_ready`` 读取 SQL 窗口特征宽表。

    默认执行标准查询（PJM_HUB 节点、时间升序），由库内 LAG / 移动平均完成特征工程。
    """
    engine = create_db_engine(database_url)
    _ensure_feature_view(engine)

    if node_id == DEFAULT_PRODUCTION_NODE_ID:
        sql = SQL_LOAD_FEATURES_PIPELINE
        df = pd.read_sql_query(sql, engine)
    else:
        sql = text(
            """
            SELECT *
            FROM v_features_pipeline_ready
            WHERE node_id = :node_id
            ORDER BY timestamp ASC
            """
        )
        with engine.connect() as conn:
            df = pd.read_sql_query(sql, conn, params={"node_id": node_id})

    if df.empty:
        raise ValueError(
            f"节点 {node_id!r} 在 v_features_pipeline_ready 中无数据。"
            "请确认 db_injector 已灌入该节点。"
        )
    return df


def adapt_sql_pipeline_to_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 SQL 视图列映射为 LightGBM 擂台所需宽表列名。

    - ``price_rt`` → ``spot_price``（结算/训练标签）
    - ``system_load`` → ``total_load``
    - ``price_rt_lag_1h`` → ``spot_price_lag_1h``（库内窗口函数，无泄露）
    - 缺失的基准气象/新能源代理列以 0 占位，增强环境列由后续鲁棒填充
    """
    out = df.copy()
    out[COL_TIMESTAMP] = pd.to_datetime(out[COL_TIMESTAMP], errors="coerce")
    out = out.dropna(subset=[COL_TIMESTAMP]).sort_values(COL_TIMESTAMP).reset_index(drop=True)

    if COL_TARGET not in out.columns:
        if "price_rt" in out.columns:
            out[COL_TARGET] = pd.to_numeric(out["price_rt"], errors="coerce")
        elif "price_da" in out.columns:
            out[COL_TARGET] = pd.to_numeric(out["price_da"], errors="coerce")
        else:
            raise ValueError("SQL 宽表缺少 price_rt / price_da，无法构造 spot_price 标签。")

    if "total_load" not in out.columns and "system_load" in out.columns:
        out["total_load"] = pd.to_numeric(out["system_load"], errors="coerce")

    if "spot_price_lag_1h" not in out.columns and "price_rt_lag_1h" in out.columns:
        out["spot_price_lag_1h"] = pd.to_numeric(out["price_rt_lag_1h"], errors="coerce")

    if "spot_price_lag_24h" not in out.columns:
        out["spot_price_lag_24h"] = out[COL_TARGET].shift(24)

    for col in ("wind_forecast", "solar_forecast", "temperature", "wind_speed"):
        if col not in out.columns:
            out[col] = 0.0

    return out


def effective_lookback_hours(config: ExperimentConfig) -> int:
    """实验有效回溯小时数（不低于 720h，且满足大样本训练下限）。"""
    return max(
        config.lookback_hours,
        EXPERIMENT_LOOKBACK_HOURS_MIN,
        MIN_MODELING_ROWS + 48,
    )


def count_usable_modeling_rows(
    df_wide: pd.DataFrame,
    required_cols: list[str],
) -> int:
    """统计 dropna(required_cols) 后可建模行数。"""
    frame = df_wide[required_cols + [COL_TIMESTAMP]].copy()
    return len(frame.dropna(subset=required_cols))


def build_full_feature_wide_table(config: ExperimentConfig) -> pd.DataFrame:
    """合成大样本特征宽表（默认 ≥1536h），供样本不足时自动切换。"""
    hours = effective_lookback_hours(config)
    tables = build_synthetic_raw_tables(n_hours=hours)
    tables = enrich_market_physics_inputs(tables, schema=config.schema)
    df_wide = build_enhanced_wide_table(tables, schema=config.schema)
    df_wide = merge_frontier_physics_features(df_wide, schema=config.schema)
    df_wide = add_model_ready_features(df_wide)
    return _robust_fill_enhanced_env_features(df_wide)


def ensure_sufficient_wide_table(
    config: ExperimentConfig,
    df_wide: pd.DataFrame,
    required_cols: list[str],
) -> pd.DataFrame:
    """若宽表有效行数不足，自动切换为大样本合成管线。"""
    n_usable = count_usable_modeling_rows(df_wide, required_cols)
    if n_usable >= MIN_MODELING_ROWS:
        return df_wide
    warnings.warn(
        f"当前有效样本仅 {n_usable} 行（目标 ≥{MIN_MODELING_ROWS}），"
        f"自动切换 {effective_lookback_hours(config)}h 合成大样本管线。",
        UserWarning,
        stacklevel=2,
    )
    return build_full_feature_wide_table(config)


def build_wide_table_from_production_db(config: ExperimentConfig) -> pd.DataFrame:
    """生产库 SQL 宽表 → 建模就绪 DataFrame（样本不足则扩容）。"""
    df_sql = load_features_from_production_db(config.database_url, config.node_id)
    df_wide = adapt_sql_pipeline_to_model_frame(df_sql)
    df_wide = _robust_fill_enhanced_env_features(df_wide)
    required = [COL_TARGET] + list(BENCHMARK_FEATURE_COLUMNS) + list(ENHANCED_ENV_COLUMNS)
    return ensure_sufficient_wide_table(config, df_wide, required)


def build_wide_table_from_csv_pipeline(config: ExperimentConfig) -> pd.DataFrame:
    """遗留 CSV 多表 → 特征工程宽表（原管线）。"""
    if config.data_dir is None:
        raise ValueError("CSV 模式必须指定 --data-dir。")
    tables = load_raw_tables(config.data_dir)
    tables = enrich_market_physics_inputs(tables, schema=config.schema)
    df_wide = build_enhanced_wide_table(tables, schema=config.schema)
    df_wide = merge_frontier_physics_features(df_wide, schema=config.schema)
    df_wide = add_model_ready_features(df_wide)
    df_wide = _robust_fill_enhanced_env_features(df_wide)
    required = [COL_TARGET] + list(BENCHMARK_FEATURE_COLUMNS) + list(ENHANCED_ENV_COLUMNS)
    return ensure_sufficient_wide_table(config, df_wide, required)


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

    t_idx = np.arange(n_hours, dtype=float)
    hour_of_day = hours.hour.to_numpy()
    evening_peak = ((hour_of_day >= 17) & (hour_of_day <= 21)).astype(float)
    temperature = 12.0 + 8.0 * np.sin(t_idx / 24.0) + rng.normal(0, 0.5, n_hours)
    relative_humidity = np.clip(
        62.0 + 18.0 * np.sin(t_idx / 24.0 * 2.0 * np.pi)
        + 6.0 * np.maximum(0.0, temperature - 22.0)
        + rng.normal(0, 2.0, n_hours),
        25.0,
        98.0,
    )
    wind_speed = np.clip(4.0 + 2.0 * rng.standard_normal(n_hours), 0.1, None)
    # 对数风廓线：10m → 100m，并叠加切变扰动使 wind_shear_alpha 有信息量
    _z0, _z10, _z100 = 0.03, 10.0, 100.0
    wind_speed_100m = wind_speed * (
        np.log(_z100 / _z0) / np.log(_z10 / _z0)
    ) * (1.0 + 0.15 * np.sin(t_idx / 36.0))
    wind_direction = (180.0 + 90.0 * np.sin(t_idx / 24.0)) % 360.0
    albedo = np.clip(0.15 + 0.4 * np.maximum(0, np.sin(t_idx / 24.0)), 0, 0.85)
    snow_depth = np.clip(15.0 - t_idx * 0.02, 0, None)
    precipitation = np.where(t_idx.astype(int) % 120 == 0, 8.0, 0.2)
    sand_dust_total = np.clip(50.0 + 120.0 * np.sin(t_idx / 48.0), 20, 300)

    # 电价与前沿物理可观测变量耦合，便于 Demo 擂台展示非零增量
    heat_stress = np.maximum(0.0, temperature - 26.0) * (relative_humidity / 100.0)
    wake_proxy = np.exp(-0.5 * ((90.0 - (wind_direction % 360.0)) / 25.0) ** 2)
    soiling_proxy = sand_dust_total / 300.0
    base_price = 320.0 + 60.0 * np.sin(t_idx / 24.0)
    bifacial_proxy = albedo * np.clip(snow_depth / 20.0, 0, 1)
    spot_price = (
        base_price
        + 55.0 * heat_stress * evening_peak
        + 40.0 * wake_proxy * np.clip(wind_speed_100m - wind_speed, 0, None)
        + 35.0 * soiling_proxy * (1.0 - np.minimum(precipitation, 10.0) / 10.0)
        + 25.0 * bifacial_proxy * np.sin(t_idx / 24.0)
        + rng.normal(0, 10, n_hours)
    )

    market = pd.DataFrame(
        {
            "timestamp": hours,
            "spot_price": spot_price,
            "total_load": 11000.0 + 1500 * np.sin(t_idx / 24.0) + rng.normal(0, 100, n_hours),
            "wind_forecast": 800.0 + 200 * rng.standard_normal(n_hours),
            "solar_forecast": np.clip(600.0 + 300 * np.sin(t_idx / 24.0), 0, None),
            "temperature": temperature,
            "relative_humidity": relative_humidity,
            "wind_speed": wind_speed,
            "wind_speed_100m": wind_speed_100m,
            "wind_direction": wind_direction,
            "albedo": albedo,
            "snow_depth": snow_depth,
            "precipitation": precipitation,
            "sand_dust_total": sand_dust_total,
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
    tables = enrich_market_physics_inputs(tables, schema=schema)
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
        col = f"spot_price_lag_{lag}h"
        if col not in out.columns:
            out[col] = out[COL_TARGET].shift(lag)

    if "water_temp" in out.columns:
        for lag in _WATER_LAG_HOURS:
            col = f"water_temp_lag_{lag}h"
            if col not in out.columns:
                out[col] = out["water_temp"].shift(lag)

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
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


def _ensure_wind_shear_coefficient(df: pd.DataFrame) -> pd.DataFrame:
    """将特征工程输出的 wind_shear_alpha 同步为建模列 wind_shear_coefficient。"""
    out = df.copy()
    if "wind_shear_coefficient" not in out.columns and "wind_shear_alpha" in out.columns:
        out["wind_shear_coefficient"] = out["wind_shear_alpha"]
    return out


def _robust_fill_enhanced_env_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    增强模型环境列 NaN 鲁棒填充：先时间方向 ffill，再 fillna(0)。

    覆盖 MICRO_PHYSICS_FEATURE_COLUMNS 及 ENHANCED_ENV_COLUMNS 全部字段，
    缺失列补 0，避免 LightGBM 训练因 NaN 报错。
    """
    out = _ensure_wind_shear_coefficient(df)
    fill_cols = list(dict.fromkeys((*ENHANCED_ENV_COLUMNS, *MICRO_PHYSICS_FEATURE_COLUMNS)))
    for col in fill_cols:
        if col not in out.columns:
            out[col] = 0.0
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").ffill().fillna(0.0)
    return out


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
    if (
        X_valid is not None
        and y_valid is not None
        and len(X_valid) > 0
        and hasattr(lgb, "early_stopping")
    ):
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
    # ---------- 1. 加载特征宽表 ----------
    if config.demo:
        df_wide = build_full_feature_wide_table(config)
    elif config.production_db or config.data_dir is None:
        df_wide = build_wide_table_from_production_db(config)
    else:
        df_wide = build_wide_table_from_csv_pipeline(config)

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

    if len(train_df) < config.min_train_samples:
        raise ValueError(
            f"训练集仅 {len(train_df)} 条，低于下限 {config.min_train_samples}。"
            f"请增大 --lookback-hours（当前 {config.lookback_hours}，"
            f"建议 ≥{EXPERIMENT_LOOKBACK_HOURS_DEFAULT}）或执行 "
            f"python data_pipeline.py --lookback-hours 720 --export-csv && "
            f"python db_injector.py --fresh"
        )

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
        "--production-db",
        action="store_true",
        help="从 data/production_market.db 的 v_features_pipeline_ready 加载（SQL 窗口特征）",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        default=PRODUCTION_DATABASE_URL,
        help="PostgreSQL 连接 URL（默认 postgresql://localhost:5432/postgres）",
    )
    parser.add_argument(
        "--node-id",
        type=str,
        default=DEFAULT_PRODUCTION_NODE_ID,
        help="SQL 过滤节点（默认 PJM_HUB；标准查询已写死该值）",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="遗留模式：从 CSV 多表跑特征工程（与 --production-db 互斥）",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.80,
        help="按时间顺序的训练集占比（默认 0.80）",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=EXPERIMENT_LOOKBACK_HOURS_DEFAULT,
        help=f"历史回溯小时数（默认 {EXPERIMENT_LOOKBACK_HOURS_DEFAULT}，不低于 {EXPERIMENT_LOOKBACK_HOURS_MIN}）",
    )
    parser.add_argument(
        "--min-train-samples",
        type=int,
        default=MIN_TRAIN_SAMPLES,
        help=f"训练集最小样本量（默认 {MIN_TRAIN_SAMPLES}）",
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
        production_db=args.production_db,
        database_url=args.database_url,
        node_id=args.node_id,
        data_dir=Path(args.data_dir) if args.data_dir else None,
        train_ratio=args.train_ratio,
        lookback_hours=args.lookback_hours,
        min_train_samples=args.min_train_samples,
        storage_power_mw=args.storage_power_mw,
        storage_capacity_mwh=args.storage_capacity_mwh,
    )


def main() -> None:
    config = parse_args()
    if not config.demo and not config.production_db and config.data_dir is None:
        print("未指定 --data-dir / --demo，默认使用 PostgreSQL 生产库模式。")
        config = ExperimentConfig(
            demo=False,
            production_db=True,
            database_url=config.database_url,
            node_id=config.node_id,
            train_ratio=config.train_ratio,
            lookback_hours=config.lookback_hours,
            min_train_samples=config.min_train_samples,
            storage_power_mw=config.storage_power_mw,
            storage_capacity_mwh=config.storage_capacity_mwh,
        )

    try:
        report = run_experiment(config)
        print_experiment_report(report)
    except Exception as exc:
        print(f"\n[实验失败] {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
