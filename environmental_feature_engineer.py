"""
环境微特征对电力现货价格影响的预测增强 — 特征工程模块。

本模块负责将 ERA5 辐射、AQI/污染、水温等环境因子清洗、衍生并与市场主表对齐，
输出可直接喂给 LightGBM 的宽表 ``df_enhanced_features``。
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from frontier_physics_constants import (  # noqa: E402
    FRONTIER_PHYSICS_BLOCKS,
    audit_frontier_physics_features,
    frontier_physics_feature_names,
)

# ---------------------------------------------------------------------------
# 地球半径（米），用于 Haversine 最近邻网格匹配
# ---------------------------------------------------------------------------
_EARTH_RADIUS_M: float = 6_371_000.0

# AOD 默认时滞步长（逐小时观测个数）
_DEFAULT_AOD_LAG_HOURS: tuple[int, ...] = (1, 3, 6, 24)


@dataclass(frozen=True)
class FeatureSchema:
    """
    各输入表的列名契约；主项目若列名不同，可实例化本类并传入各处理函数。

    注意：ERA5 ``ssrd`` 单位需在整条流水线中保持一致（常见为 J/m² 累积量或 W/m² 瞬时值）。
  """

    timestamp: str = "timestamp"
    lat: str = "lat"
    lon: str = "lon"
    ssrd: str = "ssrd"
    station_id: str = "station_id"
    capacity_mw: str = "capacity_mw"
    city: str = "city"
    pm25: str = "pm25"
    no2: str = "no2"
    aod: str = "aod"
    industrial_load_base: str = "industrial_load_base"
    water_temp: str = "water_temp"
    temperature: str = "temperature"
    relative_humidity: str = "relative_humidity"
    dew_point_temperature: str = "dew_point_temperature"
    wind_speed_10m: str = "wind_speed_10m"
    wind_speed_100m: str = "wind_speed_100m"
    wind_speed: str = "wind_speed"
    wind_direction: str = "wind_direction"
    albedo: str = "albedo"
    snow_depth: str = "snow_depth"
    sand_dust_total: str = "sand_dust_total"
    pm10: str = "pm10"
    precipitation: str = "precipitation"
    aod_lag_hours: tuple[int, ...] = field(default_factory=lambda: _DEFAULT_AOD_LAG_HOURS)
    radiation_roll_window: int = 3


# 风资源工程常量
_WIND_HEIGHT_LOW_M: float = 10.0
_WIND_HEIGHT_HUB_M: float = 100.0
_SURFACE_ROUGHNESS_Z0_M: float = 0.1
_SHEAR_ALPHA_UNSTABLE_THRESHOLD: float = 0.3
_DEFAULT_WAKE_ARRAY_AXES_DEG: tuple[float, float] = (45.0, 225.0)

# 酷热指数输出列名
_COL_HEAT_INDEX = "heat_index"
_COL_HEAT_INDEX_SPIKE_35 = "heat_index_spike_35"
_HEAT_INDEX_SPIKE_THRESHOLD_C: float = 35.0

# 高级风电特征输出列名
_COL_WIND_SHEAR_ALPHA = "wind_shear_alpha"
_COL_WIND_SHEAR_RISK = "wind_shear_risk"
_COL_WIND_DIR_DEV = "wind_dir_dev"
_COL_WAKE_EFFECT_INTENSITY = "wake_effect_intensity"

# 反照率 / 积雪特征输出列名
_COL_SNOW_MELT_RATE = "snow_melt_rate"
_COL_BIFACIAL_GAIN_INDEX = "bifacial_gain_index"
_SNOW_MELT_TEMP_COEFF: float = 0.2
_ALBEDO_SNOW_HIGH: float = 0.7

# 积尘状态机参数
_COL_PANEL_DIRT = "panel_dirt_accumulation"
_COL_PANEL_EFFICIENCY_DISCOUNT = "panel_efficiency_discount"
_RAIN_WASH_THRESHOLD_MM: float = 5.0
_DIRT_MEMORY_DECAY: float = 0.99
_MAX_DIRT_ACCUMULATION: float = 1.0
_SOILING_EFFICIENCY_PENALTY: float = 0.15
_DEFAULT_SAND_STORM_THRESHOLD: float = 150.0
_DEFAULT_DUST_JUMP_SCALE: float = 200.0

# 辐射衍生列名（输出）
_COL_EFFECTIVE_PV_RADIATION = "effective_pv_radiation"
_COL_RADIATION_MUTATION = "radiation_mutation_rate"
_COL_RADIATION_INSTABILITY = "radiation_instability_3h"

# 污染全省加权列名前缀
_PROV_SUFFIX = "_prov_weighted"

def _require_columns(df: pd.DataFrame, columns: Sequence[str], table_name: str) -> None:
    """校验 DataFrame 是否包含必需列。"""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{table_name} 缺少必需列: {missing}")


def _ensure_datetime(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    将时间列转为 datetime64[ns]，按时间排序并去除完全重复行。

    业务说明：环境与市场数据对齐前必须统一时间轴粒度（建议均为逐小时整点）。
    """
    out = df.copy()
    out[col] = pd.to_datetime(out[col], errors="coerce")
    if out[col].isna().any():
        n_bad = int(out[col].isna().sum())
        raise ValueError(f"列 '{col}' 存在 {n_bad} 条无法解析的时间戳。")
    return out.sort_values(col).reset_index(drop=True)


def _haversine_distance_m(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """计算两点间大圆距离（米）。"""
    lat1_r = np.radians(lat1)
    lon1_r = np.radians(lon1)
    lat2_r = np.radians(lat2)
    lon2_r = np.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return _EARTH_RADIUS_M * c


def _build_era5_grid_index(df_era5: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    """
    从 ERA5 长表中提取唯一网格点坐标，并赋予稳定的 grid_key。

    同一 (lat, lon) 在多时刻重复出现，仅保留一份坐标用于最近邻匹配。
    """
    grid = (
        df_era5[[schema.lat, schema.lon]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    grid["grid_key"] = np.arange(len(grid), dtype=np.int64)
    return grid


def _match_stations_to_grid(
    df_stations: pd.DataFrame,
    grid: pd.DataFrame,
    schema: FeatureSchema,
) -> pd.DataFrame:
    """
    用电站坐标对 ERA5 网格做 Haversine 最近邻匹配。

    业务机理：光伏电站位置往往不在规则网格中心，最近邻比简单四舍五入更稳健。
    """
    stations = df_stations.copy()
    station_lats = stations[schema.lat].to_numpy(dtype=float)
    station_lons = stations[schema.lon].to_numpy(dtype=float)
    grid_lats = grid[schema.lat].to_numpy(dtype=float)
    grid_lons = grid[schema.lon].to_numpy(dtype=float)

    grid_keys: list[int] = []
    for lat, lon in zip(station_lats, station_lons, strict=True):
        dist = _haversine_distance_m(
            np.array([lat]),
            np.array([lon]),
            grid_lats,
            grid_lons,
        )
        nearest_idx = int(np.argmin(dist))
        grid_keys.append(int(grid["grid_key"].iloc[nearest_idx]))

    stations["grid_key"] = grid_keys
    return stations[[schema.station_id, "grid_key", schema.capacity_mw]]


def _nearest_grid_radiation(
    df_era5: pd.DataFrame,
    df_pv_stations: pd.DataFrame,
    schema: FeatureSchema,
) -> pd.DataFrame:
    """
    为每个电站 × 时刻绑定最近 ERA5 网格点的短波辐射 ssrd。

    返回列：timestamp, station_id, ssrd, capacity_mw
    """
    era5 = _ensure_datetime(df_era5, schema.timestamp)
    grid = _build_era5_grid_index(era5, schema)
    era5 = era5.merge(grid, on=[schema.lat, schema.lon], how="left")

    station_map = _match_stations_to_grid(df_pv_stations, grid, schema)

    station_rad = era5.merge(
        station_map,
        on="grid_key",
        how="inner",
    )
    return station_rad[
        [schema.timestamp, schema.station_id, schema.ssrd, schema.capacity_mw]
    ]


def _load_weighted_province(
    df_city: pd.DataFrame,
    value_cols: Iterable[str],
    load_col: str,
    timestamp_col: str,
) -> pd.DataFrame:
    """
    按时间戳对城市级特征做工业负荷加权全省聚合。

    公式：prov_feature(t) = Σ_i feature_i(t) * load_i(t) / Σ_i load_i(t)

    当某时刻全省工业负荷合计为 0 时，回退为各城市等权算术平均，并发出警告。
    """
    value_cols = list(value_cols)
    records: list[dict[str, object]] = []

    for ts, grp in df_city.groupby(timestamp_col, sort=True):
        loads = grp[load_col].astype(float)
        total_load = float(loads.sum())
        row: dict[str, object] = {timestamp_col: ts}

        if total_load <= 0.0:
            warnings.warn(
                f"时刻 {ts} 全省 industrial_load_base 合计为 0，"
                f"对 {value_cols} 使用等权平均回退。",
                UserWarning,
                stacklevel=3,
            )
            for col in value_cols:
                row[f"{col}{_PROV_SUFFIX}"] = float(grp[col].astype(float).mean())
        else:
            weights = loads / total_load
            for col in value_cols:
                row[f"{col}{_PROV_SUFFIX}"] = float(
                    (grp[col].astype(float) * weights).sum()
                )
        records.append(row)

    return pd.DataFrame.from_records(records)


def process_radiation_features(
    df_era5: pd.DataFrame,
    df_pv_stations: pd.DataFrame,
    schema: Optional[FeatureSchema] = None,
) -> pd.DataFrame:
    """
    处理 ERA5 辐射数据，生成全省有效光伏辐射时序及稳定性衍生特征。

    业务机理
    --------
    1. **容量加权有效辐射**：全省光伏装机分布不均，用装机容量对各站暴露辐射加权，
       近似全省光伏出力潜势的辐射驱动项，优于单格点代表。
    2. **辐射突变率（一阶差分）**：刻画云系快速移动导致的辐照跳变，影响分布式预测误差
       与现货价格波动。
    3. **3 小时滚动标准差**：刻画辐照不稳定性，与备用、平衡市场紧张程度相关。

    Parameters
    ----------
    df_era5 :
        ERA5 网格化辐射，需含 timestamp, lat, lon, ssrd。
    df_pv_stations :
        光伏电站列表，需含 station_id, lat, lon, capacity_mw（>0）。
    schema :
        列名配置，默认使用 FeatureSchema 标准列名。

    Returns
    -------
    pd.DataFrame
        列：timestamp, effective_pv_radiation, radiation_mutation_rate,
        radiation_instability_3h。

    Notes
    -----
    - 输入频率假定为**逐小时**；若为 15 分钟数据，请先 ``resample('1h')`` 或调整
      ``schema.radiation_roll_window``。
    - ``ssrd`` 单位需在 ERA5 提取与建模全链路保持一致。
    """
    schema = schema or FeatureSchema()
    if df_era5.empty or df_pv_stations.empty:
        raise ValueError("df_era5 与 df_pv_stations 均不可为空。")

    _require_columns(
        df_era5,
        [schema.timestamp, schema.lat, schema.lon, schema.ssrd],
        "df_era5",
    )
    _require_columns(
        df_pv_stations,
        [schema.station_id, schema.lat, schema.lon, schema.capacity_mw],
        "df_pv_stations",
    )

    capacities = df_pv_stations[schema.capacity_mw].astype(float)
    if (capacities <= 0).any():
        raise ValueError(f"{schema.capacity_mw} 必须全部大于 0。")

    station_rad = _nearest_grid_radiation(df_era5, df_pv_stations, schema)
    station_rad["weighted_ssrd"] = (
        station_rad[schema.ssrd].astype(float) * station_rad[schema.capacity_mw].astype(float)
    )

    prov = (
        station_rad.groupby(schema.timestamp, as_index=False)
        .agg(
            _weighted_sum=("weighted_ssrd", "sum"),
            _total_cap=(schema.capacity_mw, "sum"),
        )
    )
    prov[_COL_EFFECTIVE_PV_RADIATION] = prov["_weighted_sum"] / prov["_total_cap"]
    prov = prov.drop(columns=["_weighted_sum", "_total_cap"])
    prov = prov.sort_values(schema.timestamp).reset_index(drop=True)

    # 突变率：相邻时刻有效辐射之差
    prov[_COL_RADIATION_MUTATION] = prov[_COL_EFFECTIVE_PV_RADIATION].diff()

    # 不稳定性：3 个逐小时观测的滚动标准差
    prov[_COL_RADIATION_INSTABILITY] = (
        prov[_COL_EFFECTIVE_PV_RADIATION]
        .rolling(window=schema.radiation_roll_window, min_periods=1)
        .std()
    )

    return prov[
        [
            schema.timestamp,
            _COL_EFFECTIVE_PV_RADIATION,
            _COL_RADIATION_MUTATION,
            _COL_RADIATION_INSTABILITY,
        ]
    ]


def process_pollution_interaction(
    df_aqi: pd.DataFrame,
    df_load_base: pd.DataFrame,
    schema: Optional[FeatureSchema] = None,
) -> pd.DataFrame:
    """
    构建污染浓度与工业负荷基准的交叉特征，并生成 AOD 时滞与全省负荷加权聚合结果。

    业务机理
    --------
    1. **污染 × 负荷交叉项**：高 PM2.5/NO₂ 叠加高工业负荷时，环保限产概率上升，
       负荷相对线性基准出现非线性下挫；乘积项帮助树模型捕捉该交互。
    2. **AOD 时滞**：气溶胶光学厚度削弱地表太阳辐射并影响采暖/体感负荷，
       对光伏出力与需求均有滞后效应，故按城市分组 shift 构造滞后列。
    3. **全省负荷加权**：各城市污染水平对全省电价影响权重不同，以工业负荷基准占比加权。

    Parameters
    ----------
    df_aqi :
        城市逐小时 AQI，含 timestamp, city, pm25, no2, aod。
    df_load_base :
        城市逐小时工业负荷基准，含 timestamp, city, industrial_load_base。
    schema :
        列名配置。

    Returns
    -------
    pd.DataFrame
        全省逐小时特征宽表（列名带 ``_prov_weighted`` 后缀）。

    Notes
    -----
    - 默认在 (timestamp, city) 上**内连接**，保证交叉项语义正确；
      若需保留无负荷城市，可改为 left join 并自行处理 NaN。
  """
    schema = schema or FeatureSchema()
    if df_aqi.empty or df_load_base.empty:
        raise ValueError("df_aqi 与 df_load_base 均不可为空。")

    _require_columns(
        df_aqi,
        [schema.timestamp, schema.city, schema.pm25, schema.no2, schema.aod],
        "df_aqi",
    )
    _require_columns(
        df_load_base,
        [schema.timestamp, schema.city, schema.industrial_load_base],
        "df_load_base",
    )

    aqi = _ensure_datetime(df_aqi, schema.timestamp)
    load = _ensure_datetime(df_load_base, schema.timestamp)

    merged = aqi.merge(
        load,
        on=[schema.timestamp, schema.city],
        how="inner",
    )
    if merged.empty:
        raise ValueError("df_aqi 与 df_load_base 内连接后无记录，请检查城市与时间对齐。")

    # 交叉特征：捕捉环保限产导致的负荷非线性跌落
    merged["pm25_load_cross"] = (
        merged[schema.pm25].astype(float) * merged[schema.industrial_load_base].astype(float)
    )
    merged["no2_load_cross"] = (
        merged[schema.no2].astype(float) * merged[schema.industrial_load_base].astype(float)
    )

    # AOD 时滞：按城市分组 shift，避免城市间错位
    merged = merged.sort_values([schema.city, schema.timestamp]).reset_index(drop=True)
    lag_cols: list[str] = []
    for lag_h in schema.aod_lag_hours:
        col_name = f"aod_lag_{lag_h}h"
        merged[col_name] = merged.groupby(schema.city, sort=False)[schema.aod].shift(lag_h)
        lag_cols.append(col_name)

    value_cols = [
        schema.pm25,
        schema.no2,
        schema.aod,
        *lag_cols,
        "pm25_load_cross",
        "no2_load_cross",
    ]

    prov = _load_weighted_province(
        merged,
        value_cols=value_cols,
        load_col=schema.industrial_load_base,
        timestamp_col=schema.timestamp,
    )
    return prov.sort_values(schema.timestamp).reset_index(drop=True)


def _celsius_to_fahrenheit(temp_c: np.ndarray) -> np.ndarray:
    return temp_c * 1.8 + 32.0


def _fahrenheit_to_celsius(temp_f: np.ndarray) -> np.ndarray:
    return (temp_f - 32.0) / 1.8


def _magnus_saturation_vapor_pressure_hpa(temp_c: np.ndarray) -> np.ndarray:
    """
    Magnus-Tetens 近似：计算饱和水汽压（hPa）。

    用于由露点反推实际水汽压，进而得到相对湿度。
    """
    temp_c = np.asarray(temp_c, dtype=float)
    return 6.112 * np.exp((17.67 * temp_c) / (temp_c + 243.5))


def _relative_humidity_from_dewpoint(
    temp_c: np.ndarray,
    dewpoint_c: np.ndarray,
) -> np.ndarray:
    """由气温与露点温度（Magnus-Tetens）反推相对湿度（%）。"""
    es_air = _magnus_saturation_vapor_pressure_hpa(temp_c)
    e_actual = _magnus_saturation_vapor_pressure_hpa(dewpoint_c)
    rh = 100.0 * e_actual / np.maximum(es_air, 1e-6)
    return np.clip(rh, 0.0, 100.0)


def _mock_relative_humidity(
    temp_c: np.ndarray,
    timestamps: pd.Series,
) -> np.ndarray:
    """
    在缺乏湿度观测时，基于日周期与气温 Mock 相对湿度（40%–80%）。

    业务假设（夏季典型日变化）：
    - 正午气温高、相对湿度低（蒸发增强）；
    - 早晚相对湿度回升。
    """
    hours = pd.to_datetime(timestamps).dt.hour.to_numpy(dtype=float)
    # 以 12:00 为谷值：约 40%；0:00/24:00 附近约 80%
    rh = 60.0 - 20.0 * np.cos((hours - 12.0) / 24.0 * 2.0 * np.pi)
    # 高温日略微降低 Mock 湿度
    rh -= np.clip((temp_c - 25.0) * 0.5, 0.0, 10.0)
    return np.clip(rh, 40.0, 80.0)


def _nws_heat_index_fahrenheit(temp_f: np.ndarray, rh_pct: np.ndarray) -> np.ndarray:
    """
    美国国家气象局（NWS）酷热指数（°F）。

    采用 Rothfusz 多项式回归（9 项系数），刻画高湿环境下蒸发散热受阻的非线性效应；
    并对低湿/高湿区间做经验修正。
    """
    t = np.asarray(temp_f, dtype=float)
    rh = np.asarray(rh_pct, dtype=float)
    hi = np.full_like(t, np.nan, dtype=float)

    # 低温区简化公式
    cool = t < 80.0
    hi[cool] = 0.5 * (t[cool] + 61.0 + ((t[cool] - 68.0) * 1.2) + (rh[cool] * 0.094))

    # 主回归区
    hot = ~cool
    if np.any(hot):
        th, rhh = t[hot], rh[hot]
        hi_hot = (
            -42.379
            + 2.04901523 * th
            + 10.14333127 * rhh
            - 0.22475541 * th * rhh
            - 0.00683783 * th**2
            - 0.05481717 * rhh**2
            + 0.00122874 * th**2 * rhh
            + 0.00085282 * th * rhh**2
            - 0.00000199 * th**2 * rhh**2
        )
        # 低湿修正
        low_rh = (rhh < 13.0) & (th >= 80.0) & (th <= 112.0)
        if np.any(low_rh):
            adj = ((13.0 - rhh[low_rh]) / 4.0) * np.sqrt(
                (17.0 - np.abs(th[low_rh] - 95.0)) / 17.0
            )
            hi_hot[low_rh] -= adj
        # 高湿修正
        high_rh = (rhh > 85.0) & (th >= 80.0) & (th <= 87.0)
        if np.any(high_rh):
            adj = ((rhh[high_rh] - 85.0) / 10.0) * ((87.0 - th[high_rh]) / 5.0)
            hi_hot[high_rh] += adj
        hi[hot] = hi_hot

    # 酷热指数不应低于实际气温（华氏）
    hi = np.maximum(hi, t)
    return hi


def calculate_heat_index_features(
    df_weather: pd.DataFrame,
    schema: Optional[FeatureSchema] = None,
) -> pd.DataFrame:
    """
    计算酷热指数（Heat Index）及晚高峰空调负荷弹性阈值特征。

    业务机理
    --------
    酷热指数综合气温与相对湿度，反映人体蒸发散热效率。
    当 ``heat_index`` 超过 35°C 时，空调负荷往往在晚高峰出现 **非线性暴涨**；
    ``heat_index_spike_35`` 显式编码超出阈值的部分，引导树模型捕捉尾部弹性。

    Parameters
    ----------
    df_weather :
        气象表，必须含 ``temperature``（℃）。
        可选 ``relative_humidity``（%）或 ``dew_point_temperature``（℃）。
        若含 ``timestamp``，Mock 湿度时将利用小时周期。
    schema :
        列名配置。

    Returns
    -------
    pd.DataFrame
        列：``timestamp``（若输入有）、``heat_index``、``heat_index_spike_35``、
        ``relative_humidity_used``（实际参与计算的湿度，便于审计）。
    """
    schema = schema or FeatureSchema()
    _require_columns(df_weather, [schema.temperature], "df_weather")

    frame = df_weather.copy()
    temp_c = frame[schema.temperature].astype(float).to_numpy()

    rh_pct: Optional[np.ndarray] = None
    if schema.relative_humidity in frame.columns:
        rh_pct = frame[schema.relative_humidity].astype(float).to_numpy()
        rh_pct = np.clip(rh_pct, 0.0, 100.0)
    elif schema.dew_point_temperature in frame.columns:
        dew_c = frame[schema.dew_point_temperature].astype(float).to_numpy()
        rh_pct = _relative_humidity_from_dewpoint(temp_c, dew_c)
    else:
        warnings.warn(
            "df_weather 缺少相对湿度与露点列，将基于时间周期与气温 Mock 湿度 (40%~80%)。",
            UserWarning,
            stacklevel=2,
        )
        ts_series = (
            frame[schema.timestamp]
            if schema.timestamp in frame.columns
            else pd.Series(pd.date_range("2000-06-01", periods=len(frame), freq="h"))
        )
        rh_pct = _mock_relative_humidity(temp_c, ts_series)

    temp_f = _celsius_to_fahrenheit(temp_c)
    hi_f = _nws_heat_index_fahrenheit(temp_f, rh_pct)
    hi_c = _fahrenheit_to_celsius(hi_f)

    spike = np.where(
        hi_c > _HEAT_INDEX_SPIKE_THRESHOLD_C,
        hi_c - _HEAT_INDEX_SPIKE_THRESHOLD_C,
        0.0,
    )

    out = pd.DataFrame(
        {
            _COL_HEAT_INDEX: hi_c,
            _COL_HEAT_INDEX_SPIKE_35: spike,
            "relative_humidity_used": rh_pct,
        }
    )
    if schema.timestamp in frame.columns:
        out.insert(0, schema.timestamp, pd.to_datetime(frame[schema.timestamp]))
    return out


def _log_profile_wind_speed(
    v_ref: np.ndarray,
    z_ref: float,
    z_target: float,
    z0: float = _SURFACE_ROUGHNESS_Z0_M,
) -> np.ndarray:
    """
    对数风廓线外推：$v(z_2) = v(z_1) \\cdot \\dfrac{\\ln(z_2/z_0)}{\\ln(z_1/z_0)}$。

    用于在缺乏测风塔 100m 数据时，由 10m 风速高仿真推导轮毂高度风速。
    """
    v_ref = np.maximum(np.asarray(v_ref, dtype=float), 0.1)
    z_ref = max(z_ref, z0 + 1e-3)
    z_target = max(z_target, z0 + 1e-3)
    return v_ref * (np.log(z_target / z0) / np.log(z_ref / z0))


def _resolve_hub_height_wind_speeds(
    frame: pd.DataFrame,
    schema: FeatureSchema,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    解析 10m / 100m 风速；缺失 100m 时以对数律 Mock。

    Returns
    -------
    v10, v100, mocked_mask
    """
    if schema.wind_speed_10m in frame.columns:
        v10 = frame[schema.wind_speed_10m].astype(float).to_numpy()
    elif schema.wind_speed in frame.columns:
        v10 = frame[schema.wind_speed].astype(float).to_numpy()
    else:
        raise ValueError(
            f"df_wind 需包含 {schema.wind_speed_10m} 或 {schema.wind_speed}。"
        )

    mocked = np.zeros(len(frame), dtype=bool)
    if schema.wind_speed_100m in frame.columns:
        v100 = frame[schema.wind_speed_100m].astype(float).to_numpy()
        missing = ~np.isfinite(v100) | (v100 <= 0)
        if missing.any():
            v100 = v100.copy()
            v100[missing] = _log_profile_wind_speed(
                v10[missing], _WIND_HEIGHT_LOW_M, _WIND_HEIGHT_HUB_M
            )
            mocked[missing] = True
    else:
        warnings.warn(
            "缺少 100m 轮毂风速，将基于 10m 风速与对数风廓线进行高仿真 Mock。",
            UserWarning,
            stacklevel=3,
        )
        v100 = _log_profile_wind_speed(v10, _WIND_HEIGHT_LOW_M, _WIND_HEIGHT_HUB_M)
        mocked[:] = True

    v10 = np.maximum(v10, 0.1)
    v100 = np.maximum(v100, 0.1)
    return v10, v100, mocked


def _wind_shear_exponent(v10: np.ndarray, v100: np.ndarray) -> np.ndarray:
    """
    幂律切变指数：$\\alpha = \\dfrac{\\ln(v_{100}/v_{10})}{\\ln(100/10)}$。
    """
    ratio = v100 / v10
    return np.log(ratio) / np.log(_WIND_HEIGHT_HUB_M / _WIND_HEIGHT_LOW_M)


def _wind_shear_risk_signal(alpha: np.ndarray) -> np.ndarray:
    """
    风机疲劳与保护停机风险信号。

    - $\\alpha > 0.3$：极端不稳定层结，轮毂高度突风加剧；
    - $\\alpha < 0$：逆温层结，低空与高空风速关系反常，脱网/爬坡风险上升。
    """
    risk = np.zeros_like(alpha, dtype=float)
    unstable = alpha > _SHEAR_ALPHA_UNSTABLE_THRESHOLD
    inversion = alpha < 0.0
    risk[unstable | inversion] = 1.0
    # 连续强度：越偏离正常层结，信号越强（上限 1）
    excess = np.maximum(alpha - _SHEAR_ALPHA_UNSTABLE_THRESHOLD, 0.0)
    deficit = np.maximum(-alpha, 0.0)
    risk = np.maximum(risk, np.clip(excess + deficit, 0.0, 1.0))
    return risk


def _angle_to_array_axis(wind_direction_deg: np.ndarray, axis_deg: float) -> np.ndarray:
    """
    风向与阵列轴线（无限长直线）的最小夹角，范围 [0°, 90°]。

    45° 与 225° 为同一轴线方向，故取对 45° 的线对称角即可。
    """
    w = np.asarray(wind_direction_deg, dtype=float) % 360.0
    delta = np.abs(w - axis_deg) % 360.0
    delta = np.where(delta > 180.0, 360.0 - delta, delta)
    return np.where(delta > 90.0, 180.0 - delta, delta)


def _wake_effect_intensity(wind_dir_dev: np.ndarray, sigma_deg: float = 15.0) -> np.ndarray:
    """
    尾流效应强度惩罚特征。

    当风向与阵列轴线夹角接近 **0°**（正对尾流带）或 **90°**（横向强遮挡）时取极大值，
    提示出力将低于传统气象预报的外推。
    """
    dev = np.clip(np.asarray(wind_dir_dev, dtype=float), 0.0, 90.0)
    aligned = np.exp(-0.5 * (dev / sigma_deg) ** 2)
    cross = np.exp(-0.5 * ((90.0 - dev) / sigma_deg) ** 2)
    return np.clip(aligned + cross, 0.0, 1.0)


def calculate_advanced_wind_features(
    df_wind: pd.DataFrame,
    df_wind_farms: Optional[pd.DataFrame] = None,
    schema: Optional[FeatureSchema] = None,
    array_axes_deg: Optional[tuple[float, float]] = None,
) -> pd.DataFrame:
    """
    计算风电高级微特征：垂直切变风险与阵列尾流惩罚。

    业务机理
    --------
    1. **Wind Shear**：由 10m/100m 风速估计切变指数 $\\alpha$，识别极端层结下
       风机保护停机与 **日前-日内价格爬坡** 风险（``wind_shear_risk``）。
    2. **Wake Effect**：风向相对本地主导阵列轴线（默认 45°/225°）的偏角
       ``wind_dir_dev`` 与尾流强度 ``wake_effect_intensity``，刻画流体力学损耗。

    Parameters
    ----------
    df_wind :
        风场表，需含 10m 风速（``wind_speed_10m`` 或 ``wind_speed``）；
        可选 ``wind_speed_100m``、``wind_direction``（0–360°）。
    df_wind_farms :
        可选风电场元数据；若含 ``array_axis_deg`` 列则覆盖默认轴线方向。
    schema :
        列名配置。
    array_axes_deg :
        主导阵列轴线方向（度），默认 (45, 225)。

    Returns
    -------
    pd.DataFrame
        ``wind_shear_alpha``, ``wind_shear_risk``, ``wind_dir_dev``,
        ``wake_effect_intensity`` 及审计列。
    """
    schema = schema or FeatureSchema()
    frame = df_wind.copy()

    v10, v100, mocked = _resolve_hub_height_wind_speeds(frame, schema)
    alpha = _wind_shear_exponent(v10, v100)
    shear_risk = _wind_shear_risk_signal(alpha)

    axes = array_axes_deg or _DEFAULT_WAKE_ARRAY_AXES_DEG
    if df_wind_farms is not None and not df_wind_farms.empty:
        if "array_axis_deg" in df_wind_farms.columns:
            axes = (float(df_wind_farms["array_axis_deg"].iloc[0]),)

    if schema.wind_direction in frame.columns:
        wdir = frame[schema.wind_direction].astype(float).to_numpy() % 360.0
        dev_list = [_angle_to_array_axis(wdir, ax) for ax in axes]
        wind_dir_dev = np.min(np.vstack(dev_list), axis=0)
        wake_intensity = _wake_effect_intensity(wind_dir_dev)
    else:
        warnings.warn(
            "缺少 wind_direction，wind_dir_dev / wake_effect_intensity 置 0。",
            UserWarning,
            stacklevel=2,
        )
        wind_dir_dev = np.zeros(len(frame), dtype=float)
        wake_intensity = np.zeros(len(frame), dtype=float)

    out = pd.DataFrame(
        {
            _COL_WIND_SHEAR_ALPHA: alpha,
            _COL_WIND_SHEAR_RISK: shear_risk,
            _COL_WIND_DIR_DEV: wind_dir_dev,
            _COL_WAKE_EFFECT_INTENSITY: wake_intensity,
            "wind_speed_100m_used": v100,
            "wind_speed_100m_mocked": mocked.astype(int),
        }
    )
    if schema.timestamp in frame.columns:
        out.insert(0, schema.timestamp, pd.to_datetime(frame[schema.timestamp]))
    return out


def calculate_albedo_snow_features(
    df_env: pd.DataFrame,
    schema: Optional[FeatureSchema] = None,
) -> pd.DataFrame:
    """
    计算地表反照率 / 积雪消融与双面光伏增益交叉特征。

    业务机理（北方现货市场：山西 / 山东等）
    ----------------------------------------
    冬季降雪后，若遭遇 **雪后初晴（Post-Snow Clear-Sky）** 脉冲：

    1. **积雪 + 正温**：``snow_melt_rate`` 刻画融雪速率，反映反照率即将回落的过渡态；
    2. **高反照率（0.7–0.8）× 强辐射**：地面反射显著增强 **双面光伏（Bifacial PV）**
       背面受光，出力可超预期爆发，导致：

       - 日前市场（DA）光伏预测系统性 **低估** 实际出力；
       - 午间实时出清（RT）供给过剩，价格极易砸向 **负电价深谷（Negative Price Valley）**，
         形成长尾极端形态——统计 RMSE 未必改善，但 **经济 PnL / 储能套利** 对形状极度敏感。

    本函数通过 ``bifacial_gain_index = effective_pv_radiation × albedo`` 将上述物理链条
    显式编码为可分裂特征，供 LightGBM 在尾部样本上学习非线性折价。

    Parameters
    ----------
    df_env :
        环境表，必须含 ``albedo``（0–1）、``snow_depth``（cm）、``temperature``（℃）。
        须已合并 ``effective_pv_radiation``（装机加权有效辐射）。
    schema :
        列名配置。

    Returns
    -------
    pd.DataFrame
        ``snow_melt_rate``, ``bifacial_gain_index``（及可选 ``timestamp``）。
    """
    schema = schema or FeatureSchema()
    _require_columns(
        df_env,
        [schema.albedo, schema.snow_depth, schema.temperature],
        "df_env",
    )

    frame = df_env.copy()
    albedo = np.clip(frame[schema.albedo].astype(float).to_numpy(), 0.0, 1.0)
    snow_cm = np.maximum(frame[schema.snow_depth].astype(float).to_numpy(), 0.0)
    temp_c = frame[schema.temperature].astype(float).to_numpy()

    # 正温融雪：仅当存在积雪且气温高于 0℃ 时激活
    snow_melt_rate = np.where(
        (snow_cm > 0.0) & (temp_c > 0.0),
        temp_c * _SNOW_MELT_TEMP_COEFF,
        0.0,
    )

    if _COL_EFFECTIVE_PV_RADIATION not in frame.columns:
        warnings.warn(
            f"df_env 缺少 {_COL_EFFECTIVE_PV_RADIATION}，bifacial_gain_index 置 0。",
            UserWarning,
            stacklevel=2,
        )
        effective_rad = np.zeros(len(frame), dtype=float)
    else:
        effective_rad = np.maximum(
            frame[_COL_EFFECTIVE_PV_RADIATION].astype(float).to_numpy(),
            0.0,
        )

    # 双面增益交叉项：高 albedo × 强辐射 → 午间负电价尾部风险代理
    bifacial_gain_index = effective_rad * albedo

    # 雪后初晴强化：积雪尚未融化且反照率已进入高反射区间时，放大交叉信号
    snow_clear_sky_boost = (
        (snow_cm > 0.0)
        & (albedo >= _ALBEDO_SNOW_HIGH)
        & (effective_rad > np.percentile(effective_rad, 50) if len(effective_rad) else 0.0)
    )
    bifacial_gain_index = np.where(
        snow_clear_sky_boost,
        bifacial_gain_index * 1.25,
        bifacial_gain_index,
    )

    out = pd.DataFrame(
        {
            _COL_SNOW_MELT_RATE: snow_melt_rate,
            _COL_BIFACIAL_GAIN_INDEX: bifacial_gain_index,
        }
    )
    if schema.timestamp in frame.columns:
        out.insert(0, schema.timestamp, pd.to_datetime(frame[schema.timestamp]))
    return out


def _resolve_dust_proxy(
    frame: pd.DataFrame,
    schema: FeatureSchema,
    dust_threshold: Optional[float],
) -> tuple[np.ndarray, float]:
    """
    解析沙尘/尘负荷代理序列（sand_dust_total 优先，否则 PM10）。

    Returns
    -------
    dust_signal, threshold
    """
    if schema.sand_dust_total in frame.columns:
        signal = frame[schema.sand_dust_total].astype(float).to_numpy()
    elif schema.pm10 in frame.columns:
        signal = frame[schema.pm10].astype(float).to_numpy()
        warnings.warn(
            "缺少 sand_dust_total，使用 PM10 作为沙尘暴代理指标。",
            UserWarning,
            stacklevel=3,
        )
    else:
        raise ValueError(
            f"df_timeline 需包含 {schema.sand_dust_total} 或 {schema.pm10}。"
        )

    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    if dust_threshold is None:
        if len(signal) >= 8:
            threshold = float(np.nanpercentile(signal, 75))
        else:
            threshold = _DEFAULT_SAND_STORM_THRESHOLD
    else:
        threshold = float(dust_threshold)
    return signal, max(threshold, 1e-6)


def _soiling_state_machine_forward(
    dust_signal: np.ndarray,
    precipitation_mm: np.ndarray,
    dust_threshold: float,
    rain_wash_mm: float = _RAIN_WASH_THRESHOLD_MM,
    memory_decay: float = _DIRT_MEMORY_DECAY,
    max_dirt: float = _MAX_DIRT_ACCUMULATION,
) -> np.ndarray:
    """
    因果单向积尘状态机（仅依赖 $t$ 与 $t-1$ 状态，杜绝未来信息泄露）。

    优先级（同一时刻）
    ----------------
    1. 大雨冲刷：``precipitation > rain_wash_mm`` → $\\text{Dirt}_t = 0$
    2. 沙尘突变：``dust_signal > threshold`` → $\\text{Dirt}_t = \\min(1, \\text{Dirt}_{t-1} + \\Delta)$
    3. 常态长记忆：$\\text{Dirt}_t = \\min(1, \\text{Dirt}_{t-1} \\times \\text{decay})$
    """
    n = len(dust_signal)
    dirt = np.zeros(n, dtype=float)
    prev = 0.0

    for t in range(n):
        precip = float(precipitation_mm[t])
        dust = float(dust_signal[t])

        if precip > rain_wash_mm:
            current = 0.0
        elif dust > dust_threshold:
            excess = (dust - dust_threshold) / _DEFAULT_DUST_JUMP_SCALE
            delta = min(0.35, max(0.05, excess * 0.15))
            current = min(max_dirt, prev + delta)
        else:
            current = min(max_dirt, prev * memory_decay)

        dirt[t] = current
        prev = current

    return dirt


def calculate_soiling_decay_effect(
    df_timeline: pd.DataFrame,
    schema: Optional[FeatureSchema] = None,
    dust_storm_threshold: Optional[float] = None,
    rain_wash_threshold_mm: float = _RAIN_WASH_THRESHOLD_MM,
    memory_decay: float = _DIRT_MEMORY_DECAY,
) -> pd.DataFrame:
    """
    光伏面板积尘长记忆状态机（Soiling Decay State Machine）。

    物理机理
    --------
    沙尘暴（全大气柱沙尘总量 ``sand_dust_total`` 或重度 PM10）使组件表面积尘，
    效率可在数周内持续折价；仅当 **降雨量 > 5mm** 时发生「硬重置（Hard Reset）」
    冲刷洗净。沙尘暴后、大雨前，光伏存在 **隐形减产（Invisible Derate）**，
    抬高日内现货价格中枢——尤其在中长期合约与滚动优化场景中显著。

    算法说明
    --------
    - 按 ``timestamp`` **严格升序** 单向递推，状态变量仅依赖当前与上一时刻；
    - 输出 ``panel_efficiency_discount = 1.0 - 0.15 × panel_dirt_accumulation``，
      作为供给侧折价系数供 LightGBM 学习。

    Parameters
    ----------
    df_timeline :
        时序表，需含 ``precipitation``（mm）及 ``sand_dust_total`` 或 ``pm10``。
        强烈建议含 ``timestamp`` 以保证排序因果性。
    schema :
        列名配置。
    dust_storm_threshold :
        沙尘突变阈值；默认取样本 P75。
    rain_wash_threshold_mm :
        冲刷降雨阈值（mm），默认 5.0。
    memory_decay :
        无雨无沙日的长记忆衰减系数，默认 0.99。

    Returns
    -------
    pd.DataFrame
        ``panel_dirt_accumulation``, ``panel_efficiency_discount``，
        及原始索引对齐字段。
    """
    schema = schema or FeatureSchema()
    _require_columns(df_timeline, [schema.precipitation], "df_timeline")

    if schema.timestamp not in df_timeline.columns:
        warnings.warn(
            "df_timeline 无 timestamp，将按当前行序递推；请确保数据已按时间升序排列。",
            UserWarning,
            stacklevel=2,
        )

    frame = df_timeline.copy()
    if schema.timestamp in frame.columns:
        frame[schema.timestamp] = pd.to_datetime(frame[schema.timestamp])
        frame = frame.sort_values(schema.timestamp)

    dust_signal, threshold = _resolve_dust_proxy(frame, schema, dust_storm_threshold)
    precip = np.maximum(
        frame[schema.precipitation].astype(float).to_numpy(),
        0.0,
    )

    dirt = _soiling_state_machine_forward(
        dust_signal=dust_signal,
        precipitation_mm=precip,
        dust_threshold=threshold,
        rain_wash_mm=rain_wash_threshold_mm,
        memory_decay=memory_decay,
    )

    discount = 1.0 - _SOILING_EFFICIENCY_PENALTY * dirt

    out_sorted = pd.DataFrame(
        {
            _COL_PANEL_DIRT: dirt,
            _COL_PANEL_EFFICIENCY_DISCOUNT: discount,
        }
    )
    if schema.timestamp in frame.columns:
        out_sorted[schema.timestamp] = frame[schema.timestamp].values
        return df_timeline[[schema.timestamp]].merge(
            out_sorted,
            on=schema.timestamp,
            how="left",
        )

    out_sorted.index = df_timeline.index
    return out_sorted


def enrich_market_physics_inputs(
    tables: dict[str, pd.DataFrame],
    schema: Optional[FeatureSchema] = None,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, pd.DataFrame]:
    """
    为 market 主表补齐四大前沿物理因子所需的原始输入列。

    覆盖：``relative_humidity``、``wind_speed_100m``、``wind_direction``、
    ``albedo``、``snow_depth``、``precipitation``、``sand_dust_total`` 等，
    确保后续 ``merge_frontier_physics_features`` 可触发全部计算分支。
    """
    schema = schema or FeatureSchema()
    if "market" not in tables or tables["market"] is None or tables["market"].empty:
        return tables

    rng = rng or np.random.default_rng(42)
    out_tables = {k: (v.copy() if isinstance(v, pd.DataFrame) else v) for k, v in tables.items()}
    market = out_tables["market"].copy()
    ts_col = schema.timestamp
    if ts_col in market.columns:
        market[ts_col] = pd.to_datetime(market[ts_col])
        market = market.sort_values(ts_col).reset_index(drop=True)

    n = len(market)
    t = np.arange(n, dtype=float)
    hours = (
        market[ts_col].dt.hour.to_numpy()
        if ts_col in market.columns
        else (t % 24).astype(int)
    )
    temp_c = (
        market[schema.temperature].astype(float).to_numpy()
        if schema.temperature in market.columns
        else 15.0 + 8.0 * np.sin(t / 24.0)
    )

    if schema.wind_speed in market.columns and schema.wind_speed_10m not in market.columns:
        market[schema.wind_speed_10m] = market[schema.wind_speed].astype(float)
    elif schema.wind_speed_10m in market.columns and schema.wind_speed not in market.columns:
        market[schema.wind_speed] = market[schema.wind_speed_10m].astype(float)

    if schema.relative_humidity not in market.columns:
        rh = 62.0 + 18.0 * np.sin(t / 24.0 * 2.0 * np.pi)
        rh += 6.0 * np.maximum(0.0, temp_c - 22.0)
        rh += rng.normal(0, 2.0, n)
        market[schema.relative_humidity] = np.clip(rh, 25.0, 98.0)

    ws_col = (
        schema.wind_speed_10m
        if schema.wind_speed_10m in market.columns
        else schema.wind_speed
    )
    if schema.wind_speed_100m not in market.columns and ws_col in market.columns:
        v10 = market[ws_col].astype(float).to_numpy()
        v100 = _log_profile_wind_speed(v10, _WIND_HEIGHT_LOW_M, _WIND_HEIGHT_HUB_M)
        shear_jitter = 1.0 + 0.12 * np.sin(t / 36.0) + 0.05 * rng.standard_normal(n)
        market[schema.wind_speed_100m] = np.clip(v100 * shear_jitter, 0.1, None)

    if schema.wind_direction not in market.columns:
        market[schema.wind_direction] = (180.0 + 90.0 * np.sin(t / 24.0)) % 360.0
    if schema.albedo not in market.columns:
        market[schema.albedo] = np.clip(
            0.12 + 0.45 * np.maximum(0, np.sin(t / 24.0)), 0, 0.85
        )
    if schema.snow_depth not in market.columns:
        market[schema.snow_depth] = np.clip(20.0 - t * 0.05, 0, None)
    if schema.precipitation not in market.columns:
        rain_events = ((t.astype(int) % 168) == 36).astype(float)
        market[schema.precipitation] = np.where(
            rain_events > 0,
            6.0 + 4.0 * np.sin(t),
            0.2 + 0.3 * (hours >= 18),
        )
    if schema.sand_dust_total not in market.columns:
        aqi = out_tables.get("aqi")
        if (
            isinstance(aqi, pd.DataFrame)
            and not aqi.empty
            and schema.pm10 in aqi.columns
            and ts_col in aqi.columns
        ):
            pm10_prov = (
                aqi.groupby(ts_col, as_index=False)[schema.pm10]
                .mean()
                .rename(columns={schema.pm10: schema.sand_dust_total})
            )
            market = market.merge(pm10_prov, on=ts_col, how="left")
            market[schema.sand_dust_total] = market[schema.sand_dust_total].fillna(80.0)
        else:
            market[schema.sand_dust_total] = np.clip(
                35.0 + 150.0 * np.maximum(0, np.sin(t / 80.0)), 10, 350
            )

    out_tables["market"] = market
    return out_tables


def merge_frontier_physics_features(
    df_wide: pd.DataFrame,
    schema: Optional[FeatureSchema] = None,
) -> pd.DataFrame:
    """
    在已合并辐射/污染/水温的宽表上，显式调用四大前沿物理特征计算函数。

    依次执行：酷热指数 → 风切变/尾流 → 反照率/融雪 → 积尘状态机。
    Web 端 Demo / 数据中台可在 ``build_enhanced_wide_table`` 之后二次调用以确保列完整。
    """
    schema = schema or FeatureSchema()
    result = df_wide.copy()
    if schema.timestamp in result.columns:
        result[schema.timestamp] = pd.to_datetime(result[schema.timestamp])

    if schema.temperature in result.columns:
        weather_cols = [schema.temperature]
        if schema.relative_humidity in result.columns:
            weather_cols.append(schema.relative_humidity)
        if schema.dew_point_temperature in result.columns:
            weather_cols.append(schema.dew_point_temperature)
        if schema.timestamp in result.columns:
            weather_cols = [schema.timestamp] + weather_cols
        hi_feats = calculate_heat_index_features(
            result[list(dict.fromkeys(weather_cols))],
            schema=schema,
        )
        merge_on = [schema.timestamp] if schema.timestamp in hi_feats.columns else None
        if merge_on:
            hi_merge = hi_feats.drop(columns=["relative_humidity_used"], errors="ignore")
            result = result.drop(
                columns=[c for c in (_COL_HEAT_INDEX, _COL_HEAT_INDEX_SPIKE_35) if c in result.columns],
                errors="ignore",
            )
            result = result.merge(hi_merge, on=merge_on[0], how="left")
        else:
            for col in (_COL_HEAT_INDEX, _COL_HEAT_INDEX_SPIKE_35):
                if col in hi_feats.columns:
                    result[col] = hi_feats[col].values
        for col in (_COL_HEAT_INDEX, _COL_HEAT_INDEX_SPIKE_35):
            if col in result.columns:
                result[col] = result[col].ffill().fillna(0)

    has_wind = schema.wind_speed_10m in result.columns or schema.wind_speed in result.columns
    if has_wind:
        wind_cols: list[str] = []
        if schema.timestamp in result.columns:
            wind_cols.append(schema.timestamp)
        if schema.wind_speed_10m in result.columns:
            wind_cols.append(schema.wind_speed_10m)
        elif schema.wind_speed in result.columns:
            wind_cols.append(schema.wind_speed)
        if schema.wind_speed_100m in result.columns:
            wind_cols.append(schema.wind_speed_100m)
        if schema.wind_direction in result.columns:
            wind_cols.append(schema.wind_direction)
        wind_feats = calculate_advanced_wind_features(
            result[list(dict.fromkeys(wind_cols))],
            schema=schema,
        )
        audit_drop = {"wind_speed_100m_used", "wind_speed_100m_mocked"}
        merge_cols = [c for c in wind_feats.columns if c not in audit_drop]
        if schema.timestamp in wind_feats.columns:
            result = result.drop(
                columns=[
                    c
                    for c in (
                        _COL_WIND_SHEAR_ALPHA,
                        _COL_WIND_SHEAR_RISK,
                        _COL_WIND_DIR_DEV,
                        _COL_WAKE_EFFECT_INTENSITY,
                    )
                    if c in result.columns
                ],
                errors="ignore",
            )
            result = result.merge(
                wind_feats[merge_cols],
                on=schema.timestamp,
                how="left",
            )
        else:
            for col in merge_cols:
                if col != schema.timestamp:
                    result[col] = wind_feats[col].values
        for col in (
            _COL_WIND_SHEAR_ALPHA,
            _COL_WIND_SHEAR_RISK,
            _COL_WIND_DIR_DEV,
            _COL_WAKE_EFFECT_INTENSITY,
        ):
            if col in result.columns:
                result[col] = result[col].ffill().fillna(0)

    has_albedo_snow = (
        schema.albedo in result.columns
        and schema.snow_depth in result.columns
        and schema.temperature in result.columns
    )
    if has_albedo_snow:
        env_cols_list = [
            schema.albedo,
            schema.snow_depth,
            schema.temperature,
            _COL_EFFECTIVE_PV_RADIATION,
        ]
        if schema.timestamp in result.columns:
            env_cols_list = [schema.timestamp] + env_cols_list
        env_cols_list = [c for c in env_cols_list if c in result.columns]
        albedo_feats = calculate_albedo_snow_features(
            result[env_cols_list],
            schema=schema,
        )
        if schema.timestamp in albedo_feats.columns:
            result = result.drop(
                columns=[c for c in (_COL_SNOW_MELT_RATE, _COL_BIFACIAL_GAIN_INDEX) if c in result.columns],
                errors="ignore",
            )
            result = result.merge(albedo_feats, on=schema.timestamp, how="left")
        else:
            for col in (_COL_SNOW_MELT_RATE, _COL_BIFACIAL_GAIN_INDEX):
                result[col] = albedo_feats[col].values
        for col in (_COL_SNOW_MELT_RATE, _COL_BIFACIAL_GAIN_INDEX):
            if col in result.columns:
                result[col] = result[col].ffill().fillna(0)

    has_soiling = schema.precipitation in result.columns and (
        schema.sand_dust_total in result.columns or schema.pm10 in result.columns
    )
    if has_soiling:
        soil_cols = [schema.precipitation]
        if schema.sand_dust_total in result.columns:
            soil_cols.append(schema.sand_dust_total)
        elif schema.pm10 in result.columns:
            soil_cols.append(schema.pm10)
        if schema.timestamp in result.columns:
            soil_cols = [schema.timestamp] + soil_cols
        soil_feats = calculate_soiling_decay_effect(
            result[list(dict.fromkeys(soil_cols))],
            schema=schema,
        )
        merge_key = schema.timestamp if schema.timestamp in soil_feats.columns else None
        feat_cols = [_COL_PANEL_DIRT, _COL_PANEL_EFFICIENCY_DISCOUNT]
        if merge_key:
            result = result.drop(columns=[c for c in feat_cols if c in result.columns], errors="ignore")
            result = result.merge(
                soil_feats[[merge_key] + feat_cols],
                on=merge_key,
                how="left",
            )
        else:
            for col in feat_cols:
                result[col] = soil_feats[col].values
        for col in feat_cols:
            if col in result.columns:
                result[col] = result[col].ffill().fillna(
                    1.0 if col == _COL_PANEL_EFFICIENCY_DISCOUNT else 0.0
                )

    return result


def align_and_merge_all(
    df_market: pd.DataFrame,
    df_rad: pd.DataFrame,
    df_pol: pd.DataFrame,
    df_water: pd.DataFrame,
    schema: Optional[FeatureSchema] = None,
) -> pd.DataFrame:
    """
    对齐中台：以市场主表时间轴为基准左连接环境因子，并做缺失值鲁棒填充。

    业务机理
    --------
    - **左连接**：保证历史电价/负荷行不丢失，环境因子为增强特征而非主键。
    - **ffill + 0 填充**：环境数据空洞先沿用最近有效观测（物理连续性），
      仍缺失则置 0，表示“无额外环境冲击信号”，便于 LightGBM 分裂。

    Parameters
    ----------
    df_market :
        主项目历史电价与负荷，必须含 timestamp。
    df_rad :
        ``process_radiation_features`` 输出。
    df_pol :
        ``process_pollution_interaction`` 输出。
    df_water :
        水温序列，至少含 timestamp, water_temp（可含更多预计算列）。
    schema :
        列名配置。

    Returns
    -------
    pd.DataFrame
        ``df_enhanced_features`` 宽表，可直接用于 LightGBM 训练。
    """
    schema = schema or FeatureSchema()
    if df_market.empty:
        raise ValueError("df_market 不可为空。")

    _require_columns(df_market, [schema.timestamp], "df_market")

    market = _ensure_datetime(df_market, schema.timestamp)
    market_cols = set(market.columns)

    # 环境表去重：同一时刻保留最后一条观测
    env_frames: list[pd.DataFrame] = []
    for env_df, name in ((df_rad, "df_rad"), (df_pol, "df_pol"), (df_water, "df_water")):
        if env_df is None or env_df.empty:
            continue
        _require_columns(env_df, [schema.timestamp], name)
        env = _ensure_datetime(env_df, schema.timestamp)
        env = env.drop_duplicates(subset=[schema.timestamp], keep="last")
        env_frames.append(env)

    result = market.copy()
    for env in env_frames:
        env_feature_cols = [c for c in env.columns if c != schema.timestamp]
        result = result.merge(env, on=schema.timestamp, how="left", suffixes=("", "_dup"))
        dup_cols = [c for c in result.columns if c.endswith("_dup")]
        if dup_cols:
            result = result.drop(columns=dup_cols)

    # 环境衍生列 = 合并后新增的列（不含 timestamp 与 market 原有列）
    env_cols = [c for c in result.columns if c not in market_cols and c != schema.timestamp]

    if env_cols:
        result[env_cols] = result[env_cols].ffill()
        result[env_cols] = result[env_cols].fillna(0)

    result = merge_frontier_physics_features(result, schema=schema)

    # 列顺序：时间戳 + 市场原始列 + 环境增强列
    _audit_cols = {"relative_humidity_used", "wind_speed_100m_used", "wind_speed_100m_mocked"}
    ordered = [schema.timestamp] + [c for c in market.columns if c != schema.timestamp]
    ordered += [
        c
        for c in result.columns
        if c not in ordered and c not in _audit_cols
    ]
    return result[ordered].reset_index(drop=True)


def build_enhanced_features(
    df_market: pd.DataFrame,
    df_era5: pd.DataFrame,
    df_pv_stations: pd.DataFrame,
    df_aqi: pd.DataFrame,
    df_load_base: pd.DataFrame,
    df_water: pd.DataFrame,
    schema: Optional[FeatureSchema] = None,
) -> pd.DataFrame:
    """
    一站式串联：辐射 → 污染交互 → 对齐合并。

    便于 Notebook 或脚本单行调用完成第一步数据合成。
    """
    schema = schema or FeatureSchema()
    df_rad = process_radiation_features(df_era5, df_pv_stations, schema=schema)
    df_pol = process_pollution_interaction(df_aqi, df_load_base, schema=schema)
    return align_and_merge_all(df_market, df_rad, df_pol, df_water, schema=schema)


def _toy_demo() -> pd.DataFrame:
    """构造最小 toy 数据并跑通全流程，供本地自检。"""
    hours = pd.date_range("2024-06-01", periods=24, freq="h")
    grid_lats = [30.0, 30.25, 30.5]
    grid_lons = [120.0, 120.25, 120.5]

    era5_rows = []
    for ts in hours:
        for lat in grid_lats:
            for lon in grid_lons:
                era5_rows.append(
                    {
                        "timestamp": ts,
                        "lat": lat,
                        "lon": lon,
                        "ssrd": 200.0 + hash((ts, lat, lon)) % 50,
                    }
                )
    df_era5 = pd.DataFrame(era5_rows)

    df_pv = pd.DataFrame(
        {
            "station_id": ["s1", "s2"],
            "lat": [30.1, 30.4],
            "lon": [120.1, 120.4],
            "capacity_mw": [100.0, 200.0],
        }
    )

    cities = ["hangzhou", "ningbo"]
    aqi_rows = []
    load_rows = []
    for ts in hours:
        for i, city in enumerate(cities):
            aqi_rows.append(
                {
                    "timestamp": ts,
                    "city": city,
                    "pm25": 30.0 + i * 5,
                    "no2": 20.0 + i * 3,
                    "aod": 0.3 + i * 0.05,
                }
            )
            load_rows.append(
                {
                    "timestamp": ts,
                    "city": city,
                    "industrial_load_base": 500.0 + i * 100,
                }
            )

    df_market = pd.DataFrame(
        {
            "timestamp": hours,
            "spot_price": np.linspace(300, 400, len(hours)),
            "total_load": np.linspace(10000, 12000, len(hours)),
            "temperature": 28.0 + 6.0 * np.sin(np.arange(len(hours)) / 24.0),
            "wind_speed": np.clip(4.0 + 2.0 * np.sin(np.arange(len(hours)) / 24.0), 0.5, None),
            "wind_direction": (180.0 + 90.0 * np.sin(np.arange(len(hours)) / 24.0)) % 360.0,
            "albedo": np.clip(0.15 + 0.5 * np.maximum(0, np.sin(np.arange(len(hours)) / 24.0)), 0, 0.85),
            "snow_depth": np.clip(20.0 - np.arange(len(hours)) * 0.5, 0, None),
            "precipitation": np.where(np.arange(len(hours)) == 12, 8.0, 0.2),
            "sand_dust_total": np.where(np.arange(len(hours)) < 6, 220.0, 40.0),
        }
    )
    df_water = pd.DataFrame({"timestamp": hours, "water_temp": 18.0 + np.sin(np.arange(24))})

    return build_enhanced_features(
        df_market=df_market,
        df_era5=df_era5,
        df_pv_stations=df_pv,
        df_aqi=pd.DataFrame(aqi_rows),
        df_load_base=pd.DataFrame(load_rows),
        df_water=df_water,
    )


if __name__ == "__main__":
    enhanced = _toy_demo()
    print(f"合成宽表形状: {enhanced.shape}")
    print(enhanced.head())
    print("\n环境增强列示例:")
    env_sample = [c for c in enhanced.columns if c not in ("timestamp", "spot_price", "total_load")]
    print(enhanced[env_sample].head())
