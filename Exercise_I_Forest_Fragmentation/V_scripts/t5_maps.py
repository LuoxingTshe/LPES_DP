"""
t5_maps.py – Task 5：森林变化地图 + 信息图要用的四个数字。

地图：伯尔尼州 A -> B 的森林增加 / 减少 / 稳定，叠加生物地理区边界，
      带图例、比例尺、指北针、标题和数据来源。
      matplotlib 画栅格（imshow + extent 用 LV95 坐标），geopandas 画边界，
      matplotlib-scalebar 画比例尺（坐标单位是 m，所以 dx = 1）。

信息图的数字：Task 2 的 meff 变化 -> Task 3 的边缘/内部比 -> Task 4 的最强驱动因子。
      信息图本身是设计工作，这里只把数字整理成一张表，第四个元素（规划建议）你自己写。

读取：IV_data_processed/change_AB.tif（Task 4）, units.geojson（Task 2）
      VI_outputs/tables/T2_change.csv, T3_outlook.csv, T4_forest_change.csv, T4_coefficients.csv
写出：VI_outputs/maps/T5_forest_change.png（300 dpi）和 .pdf（矢量边界，适合排版）
      VI_outputs/tables/T5_infographic_numbers.csv
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib_scalebar.scalebar import ScaleBar

from V_scripts import t2_metrics, t4_drivers

# change_AB.tif 的四类：稳定非森林 / 稳定森林 / 增加 / 减少
COLOURS = ["#f3f1ea", "#bcd3ab", "#2c7bb6", "#d7191c"]
REGION_COLOUR = "#7b3294"

SOURCES = ("Data: FSO Arealstatistik (AS85, AS18, 100 m); FOEN biogeographical regions; "
           "swisstopo swissBOUNDARIES3D. CRS: CH1903+/LV95 (EPSG:2056).")


def raster_extent(array, transform):
    """imshow 的 extent = (左, 右, 下, 上)，用地图坐标，这样矢量边界能直接叠上去。"""
    nrow, ncol = array.shape
    left = transform.c
    top = transform.f
    right = left + ncol * transform.a
    bottom = top + nrow * transform.e          # e 是负数
    return (left, right, bottom, top)


def north_arrow(ax):
    """右上角画一个指北箭头。LV95 的网格北和真北在瑞士几乎重合。"""
    ax.annotate("N", xy=(0.94, 0.95), xytext=(0.94, 0.86), xycoords="axes fraction",
                ha="center", va="center", fontsize=13, fontweight="bold",
                arrowprops={"facecolor": "black", "width": 4, "headwidth": 11})


def change_map(cfg):
    processed = cfg["paths"]["processed"]
    nodata = cfg["nodata"]
    units = gpd.read_file(processed / "units.geojson")
    canton = units[units["scale"] == "canton"]
    regions = units[units["scale"] == "region"]
    stats = pd.read_csv(cfg["paths"]["tables"] / "T4_forest_change.csv", index_col="unit")

    array, transform = t2_metrics.clip_to_unit(processed / "change_AB.tif", canton.geometry.iloc[0], nodata)
    shown = np.ma.masked_equal(array, nodata)

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.imshow(shown, cmap=ListedColormap(COLOURS), vmin=-0.5, vmax=3.5,
              extent=raster_extent(array, transform), interpolation="nearest")
    # 区界用颜色区分而不是虚线：边界顶点很密，虚线会被画成实线
    regions.boundary.plot(ax=ax, color=REGION_COLOUR, linewidth=1.3)
    canton.boundary.plot(ax=ax, color="black", linewidth=1.6)

    # 区名写在每个区内部的一个点上（representative_point 保证点在面内）
    for i in range(len(regions)):
        point = regions.geometry.iloc[i].representative_point()
        name = regions["unit"].iloc[i].replace(" (BE part)", "")
        row = stats.loc[regions["unit"].iloc[i]]
        label = name + "\nnet " + format(int(row["net_ha"]), "+d") + " ha (" + format(row["net_pct"], "+.1f") + " %)"
        ax.text(point.x, point.y, label, ha="center", va="center", fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2})

    canton_row = stats.loc[canton["unit"].iloc[0]]
    survey_a = "AS" + cfg["time_A"]
    survey_b = "AS" + cfg["time_B"]
    handles = [
        Patch(facecolor=COLOURS[2], label="Forest gain (" + format(int(canton_row["gross_gain_ha"]), ",") + " ha)"),
        Patch(facecolor=COLOURS[3], label="Forest loss (" + format(int(canton_row["gross_loss_ha"]), ",") + " ha)"),
        Patch(facecolor=COLOURS[1], label="Stable forest"),
        Patch(facecolor=COLOURS[0], edgecolor="#999999", label="Stable non-forest"),
        Line2D([0], [0], color=REGION_COLOUR, linewidth=1.3, label="Biogeographical region (FOEN)"),
        Line2D([0], [0], color="black", linewidth=1.6, label="Canton of Bern"),
    ]
    # 图例放在右上角指北针下面：州的东北方向那块是空白
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 0.82), fontsize=10, frameon=True,
              title=survey_a + " → " + survey_b + " (1 cell = 1 ha)")

    ax.add_artist(ScaleBar(1, units="m", location="lower right", length_fraction=0.2, box_alpha=0.8))
    north_arrow(ax)
    ax.set_title("Forest change in the Canton of Bern, " + survey_a + " (≈1981) – " + survey_b + " (≈2014)",
                 fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0.0, -0.01, SOURCES, transform=ax.transAxes, ha="left", va="top", fontsize=8, color="#555555")

    maps = cfg["paths"]["maps"]
    fig.savefig(maps / "T5_forest_change.png", dpi=300, bbox_inches="tight")
    fig.savefig(maps / "T5_forest_change.pdf", bbox_inches="tight")
    plt.close(fig)


def infographic_numbers(cfg):
    """
    信息图链条的前三环，每个尺度一行：
        meff 变化（CBC）-> 边缘占比 / 边缘内部比的变化 -> 最强驱动因子（|系数| 最大的变量）
    """
    tables = cfg["paths"]["tables"]
    t2 = pd.read_csv(tables / "T2_change.csv", index_col="unit")
    t3 = pd.read_csv(tables / "T3_outlook.csv", index_col="unit")
    coef = pd.read_csv(tables / "T4_coefficients.csv")
    coef = coef[coef["variable"] != "const"]

    rows = []
    for unit_name in t2.index:
        row = {
            "unit": unit_name,
            "meff_cbc_A_km2": t2.loc[unit_name, "meff_forest_cbc_km2_A"],
            "meff_cbc_B_km2": t2.loc[unit_name, "meff_forest_cbc_km2_B"],
            "meff_chg_pct": t2.loc[unit_name, "meff_forest_cbc_km2_chg_pct"],
            "edge_share_A_pct": t3.loc[unit_name, "edge_share_pct_A"],
            "edge_share_B_pct": t3.loc[unit_name, "edge_share_pct_B"],
            "ratio_A": t3.loc[unit_name, "ratio_A"],
            "ratio_B": t3.loc[unit_name, "ratio_B"],
            "ratio_future": t3.loc[unit_name, "ratio_future"],
            "key_driver": "",
            "key_driver_or": np.nan,
        }
        # 只有建了模型的两个区才有驱动因子
        key = unit_name.split(" ")[0]
        part = coef[coef["region"] == key]
        if len(part) > 0:
            strongest = part.loc[part["coef_per_sd"].abs().idxmax()]
            row["key_driver"] = strongest["variable"] + " (forest " + strongest["model"] + ")"
            row["key_driver_or"] = strongest["odds_ratio"]
        rows.append(row)
    table = pd.DataFrame(rows).round(3)
    table.to_csv(tables / "T5_infographic_numbers.csv", index=False)
    return table


def run(cfg):
    change_map(cfg)
    table = infographic_numbers(cfg)
    print(table.to_string(index=False))
    print("  Task 5 完成。地图在 VI_outputs/maps/，信息图数字在 T5_infographic_numbers.csv")
