"""
t3_edge.py – Task 3：森林边缘 (0–100 m) / 内部 (>100 m) 分类、边缘/内部比、未来 20 年外推。

思路：和 Task 2 一样，重活交给公开库。
    scipy.ndimage.binary_erosion   腐蚀：一个森林像元的邻居全是森林才保留 -> 内部，
                                   被腐蚀掉的森林像元 -> 边缘
    t1_grid.polygons_to_mask       单元 mask（和 Task 2 用同一份 units.geojson）
    pandas.crosstab                A -> B 的类别转移矩阵，看新增的边缘是从哪里来的

读取：IV_data_processed/landscape_A.tif, _B.tif  （Task 2 的三类景观，1 = 森林）
      IV_data_processed/units.geojson
      VI_outputs/tables/T1_forest_summary.csv    （每个单元两期的平均调查年份）
写出：
      IV_data_processed/edge_A.tif, edge_B.tif     0 = 非森林, 1 = 边缘, 2 = 内部, 255 = 无数据
      VI_outputs/tables/T3_edge_interior.csv        长表：每个单元 × 每个时间一行
      VI_outputs/tables/T3_outlook.csv              每个单元：A、B、每 10 年变化、外推值
      VI_outputs/tables/T3_transition_canton.csv    伯尔尼州 A -> B 转移矩阵（ha）
      VI_outputs/figures/T3_edge_interior.png

为什么用 Task 2 的 landscape 而不是 Task 1 的 forest：
    landscape 里，主要道路和铁路穿过森林的像元已经是"障碍"，不再是森林。
    这样道路切开森林也会产生边缘，和 Task 2 的边缘密度 ED 口径一致。
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from V_scripts import utils, t1_grid, t2_metrics

# edge 栅格的编码
NONFOREST = 0
EDGE = 1
INTERIOR = 2
CLASS_NAMES = {NONFOREST: "non-forest", EDGE: "edge", INTERIOR: "interior"}


# ---------------------------------------------------------------------------
# 1. 边缘 / 内部分类
# ---------------------------------------------------------------------------
def classify_edge(land, cfg):
    """
    三类景观 -> 0 非森林 / 1 边缘 / 2 内部。

    腐蚀 (erosion) 的规则：一个像元只有当结构元素覆盖到的邻居全是 True 时才保留。
        4 邻域：上下左右四个邻居都是森林 -> 内部
        8 邻域：再加四个对角邻居
    在 100 m 网格上，上下左右邻居的中心距离是 100 m，对角邻居是 141 m。
    所以 4 邻域对应的正好是"距非森林 ≤ 100 m"，8 邻域会把边缘带稍微放宽。

    边缘宽度大于 100 m 时（比如 200 m），重复腐蚀 edge_width_m / 100 次。

    无数据像元（瑞士国界以外）在腐蚀时当作森林：
    国界不是真实的森林边缘，不能因为数据到头了就把那里的森林算成边缘。
    """
    nodata = cfg["nodata"]
    forest = land == t2_metrics.FOREST
    forest_or_nodata = forest | (land == nodata)

    if cfg["neighbourhood"] == 4:
        structure = ndimage.generate_binary_structure(2, 1)   # 十字形
    else:
        structure = ndimage.generate_binary_structure(2, 2)   # 3×3

    iterations = int(cfg["edge_width_m"] / t1_grid.RES)
    core = ndimage.binary_erosion(forest_or_nodata, structure=structure,
                                  iterations=iterations, border_value=1)

    interior = forest & core                       # 森林，并且腐蚀后还在
    edge = forest & np.logical_not(core)           # 森林，但被腐蚀掉了

    classes = np.full(land.shape, nodata, dtype=np.uint8)
    classes[land != nodata] = NONFOREST
    classes[edge] = EDGE
    classes[interior] = INTERIOR
    return classes


def summarise(classes, unit_mask):
    """统计一个单元内的边缘、内部面积（1 像元 = 1 ha）和边缘/内部比。"""
    edge_ha = int(np.sum(unit_mask & (classes == EDGE)))
    interior_ha = int(np.sum(unit_mask & (classes == INTERIOR)))
    forest_ha = edge_ha + interior_ha

    ratio = np.nan
    edge_share = np.nan
    if interior_ha > 0:
        ratio = edge_ha / interior_ha
    if forest_ha > 0:
        edge_share = 100.0 * edge_ha / forest_ha

    result = {
        "forest_ha": forest_ha,
        "edge_ha": edge_ha,
        "interior_ha": interior_ha,
        "edge_share_pct": edge_share,       # 边缘占森林的百分比，比 ratio 直观
        "edge_interior_ratio": ratio,
    }
    return result


# ---------------------------------------------------------------------------
# 2. 外推：如果 A -> B 的变化速度保持不变
# ---------------------------------------------------------------------------
def summary_key(unit_name):
    """
    T1_forest_summary.csv 里只有 canton / urban / rural 三行。
    生物地理区没有单独的调查年份，用全州的平均年份代替（差别不到一年）。
    """
    if unit_name.endswith("(urban)"):
        return "urban"
    if unit_name.endswith("(rural)"):
        return "rural"
    return "canton"


def outlook(table, cfg):
    """
    线性外推：每年的变化量 = (ratio_B - ratio_A) / 两期之间的年数，
    从 B 的调查年份再往后推 outlook_years 年。

    用 Task 1 统计的平均调查年份而不是 1985、2018 这两个"名义年份"：
    伯尔尼州的 AS85 实际大多在 1981 年前后调查，AS18 在 2014 年前后。
    """
    years = pd.read_csv(cfg["paths"]["tables"] / "T1_forest_summary.csv", index_col="unit")
    horizon = cfg["outlook_years"]

    rows = []
    for unit_name in table["unit"].unique():
        a = table[(table["unit"] == unit_name) & (table["time"] == "A")].iloc[0]
        b = table[(table["unit"] == unit_name) & (table["time"] == "B")].iloc[0]
        key = summary_key(unit_name)
        year_a = years.loc[key, "mean_year_A"]
        year_b = years.loc[key, "mean_year_B"]

        rate_per_year = (b["edge_interior_ratio"] - a["edge_interior_ratio"]) / (year_b - year_a)
        share_rate = (b["edge_share_pct"] - a["edge_share_pct"]) / (year_b - year_a)

        rows.append({
            "unit": unit_name,
            "scale": a["scale"],
            "year_A": year_a,
            "year_B": year_b,
            "ratio_A": a["edge_interior_ratio"],
            "ratio_B": b["edge_interior_ratio"],
            "ratio_chg_per_10yr": 10 * rate_per_year,
            "year_future": year_b + horizon,
            "ratio_future": b["edge_interior_ratio"] + horizon * rate_per_year,
            "edge_share_pct_A": a["edge_share_pct"],
            "edge_share_pct_B": b["edge_share_pct"],
            "edge_share_pct_future": b["edge_share_pct"] + horizon * share_rate,
        })
    result = pd.DataFrame(rows).round(3)
    result.to_csv(cfg["paths"]["tables"] / "T3_outlook.csv", index=False)
    return result


# ---------------------------------------------------------------------------
# 3. 转移矩阵和检查图
# ---------------------------------------------------------------------------
def transition_matrix(classes_a, classes_b, unit_mask, nodata, out_path):
    """
    A 期类别（行）× B 期类别（列）的像元数 = 面积 (ha)。
    例如 "interior -> edge" 是内部变成了边缘：森林内部被开了口子，或旁边的森林没了。
    "non-forest -> edge" 是新长出或新种的森林，一般先以边缘的形式出现。
    """
    inside = unit_mask & (classes_a != nodata) & (classes_b != nodata)
    a = pd.Series(classes_a[inside]).map(CLASS_NAMES)
    b = pd.Series(classes_b[inside]).map(CLASS_NAMES)
    matrix = pd.crosstab(a, b, rownames=["A"], colnames=["B"])
    matrix.to_csv(out_path)
    return matrix


def quicklook(paths, labels, units, nodata, out_path):
    """州、城市市镇、乡村市镇三行，A、B 两列，显示非森林 / 边缘 / 内部。"""
    colours = ListedColormap(["#ece8dc", "#e08a2c", "#2f6b34"])   # 非森林 / 边缘 / 内部
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    for row in range(3):
        geometry = units.geometry.iloc[row]
        for col in range(2):
            array, _ = t2_metrics.clip_to_unit(paths[col], geometry, nodata)
            shown = np.ma.masked_equal(array, nodata)
            axes[row, col].imshow(shown, cmap=colours, vmin=-0.5, vmax=2.5, interpolation="nearest")
            axes[row, col].set_title(units["unit"].iloc[row] + " – " + labels[col])
            axes[row, col].set_axis_off()
    fig.suptitle("Forest edge (orange, 0–100 m) and interior (green, > 100 m)")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def run(cfg):
    processed = cfg["paths"]["processed"]
    nodata = cfg["nodata"]
    print("  邻域规则：", cfg["neighbourhood"], "  边缘宽度：", cfg["edge_width_m"], "m")

    units = gpd.read_file(processed / "units.geojson")
    _, profile = utils.read_tif(processed / "lulc_A.tif")   # 参考栅格

    unit_masks = []
    for i in range(len(units)):
        unit_masks.append(t1_grid.polygons_to_mask(units.iloc[[i]], profile) == 1)

    records = []
    edge_paths = []
    classes_by_time = {}
    for time in ["A", "B"]:
        land, _ = utils.read_tif(processed / ("landscape_" + time + ".tif"))
        classes = classify_edge(land, cfg)
        classes_by_time[time] = classes

        path = processed / ("edge_" + time + ".tif")
        utils.save_tif(classes, path, profile, nodata=nodata)
        edge_paths.append(path)

        for i in range(len(units)):
            record = {"unit": units["unit"].iloc[i], "scale": units["scale"].iloc[i], "time": time}
            record.update(summarise(classes, unit_masks[i]))
            records.append(record)

    tables = cfg["paths"]["tables"]
    table = pd.DataFrame(records).round(3)
    table.to_csv(tables / "T3_edge_interior.csv", index=False)
    future = outlook(table, cfg)
    matrix = transition_matrix(classes_by_time["A"], classes_by_time["B"], unit_masks[0],
                               nodata, tables / "T3_transition_canton.csv")
    labels = ["AS" + cfg["time_A"], "AS" + cfg["time_B"]]
    quicklook(edge_paths, labels, units, nodata, cfg["paths"]["figures"] / "T3_edge_interior.png")

    print(table.to_string(index=False))
    print()
    print(future[["unit", "ratio_A", "ratio_B", "ratio_chg_per_10yr", "year_future", "ratio_future"]].to_string(index=False))
    print()
    print("  伯尔尼州 A -> B 转移 (ha)：")
    print(matrix)
    print("  Task 3 完成。")
