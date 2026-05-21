# -*- coding: utf-8 -*-
"""
frontier_physics_constants.py — 四大前沿物理因子板块定义（轻量模块）

独立拆分以避免 Streamlit / 数据中台 / 特征工程之间的循环导入与热重载缓存问题。
"""

from __future__ import annotations

import pandas as pd

# 四大前沿物理因子板块（宽表审计 + Web SHAP）
FRONTIER_PHYSICS_BLOCKS: dict[str, tuple[str, ...]] = {
    "酷热指数 Heat Index": ("heat_index", "heat_index_spike_35"),
    "风速切变与尾流 Wind Shear & Wake": (
        "wind_shear_alpha",
        "wind_shear_risk",
        "wind_dir_dev",
        "wake_effect_intensity",
    ),
    "双面光伏反照率 Bifacial Albedo": ("snow_melt_rate", "bifacial_gain_index"),
    "积尘长记忆 Soiling State Machine": (
        "panel_dirt_accumulation",
        "panel_efficiency_discount",
    ),
}


def frontier_physics_feature_names() -> tuple[str, ...]:
    """返回四大前沿物理板块全部特征列名（去重）。"""
    names: list[str] = []
    for cols in FRONTIER_PHYSICS_BLOCKS.values():
        names.extend(cols)
    return tuple(dict.fromkeys(names))


def audit_frontier_physics_features(df: pd.DataFrame) -> dict[str, bool]:
    """检查宽表是否包含各前沿物理板块特征。"""
    present = set(df.columns)
    return {
        block: all(c in present for c in cols)
        for block, cols in FRONTIER_PHYSICS_BLOCKS.items()
    }
