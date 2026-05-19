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
    aod_lag_hours: tuple[int, ...] = field(default_factory=lambda: _DEFAULT_AOD_LAG_HOURS)
    radiation_roll_window: int = 3


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

    # 列顺序：时间戳 + 市场原始列 + 环境增强列
    ordered = [schema.timestamp] + [c for c in market.columns if c != schema.timestamp]
    ordered += [c for c in env_cols if c not in ordered]
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
