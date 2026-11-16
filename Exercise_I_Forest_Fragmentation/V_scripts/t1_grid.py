import numpy as np
import geopandas as gpd
import pandas as pd
import pyogrio
from rasterio.features import rasterize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from V_scripts import utils

RES = 100
CRS = "EPSG:2056"  # CH1903+ / LV95
LULC_LAYER = "arealstatistik_all"
CANTON_LAYER = "TLM_KANTONSGEBIET"    
MUNICIPALITY_LAYER = "TLM_HOHEITSGEBIET"  

def loadptgrid(cfg):
    """load point grid path from config file"""
    gpkg_path = cfg["paths"]["raw"] / cfg["files"]["lulc"]
    time_a = cfg["time_A"]
    time_b = cfg["time_B"]
    columns = [
        "E_COORD",                     # 东坐标（米）
        "N_COORD",                     # 北坐标（米）
        "AS" + time_a + "_72",         # 时间 A 的 72 类土地利用
        "AS" + time_b + "_72",         # 时间 B 的 72 类土地利用
        "FJ" + time_a,                 # 时间 A 的实际调查年份
        "FJ" + time_b,                 # 时间 B 的实际调查年份
    ]
    print("  读取点数据：", gpkg_path.name)
    df = pyogrio.read_dataframe(gpkg_path, layer=LULC_LAYER, columns=columns, read_geometry=False)
    print("  点数：", len(df))
    return df
def build_grid_meta(df):
    
    x_min_points = int(df["E_COORD"].min())
    x_max_points = int(df["E_COORD"].max())
    y_min_points = int(df["N_COORD"].min())
    y_max_points = int(df["N_COORD"].max())

    pad = 1
    half = RES // 2                    # 半个像元 = 50 m

    # 点在格子中心，所以：
    # 左边界 = 最西的点再往西半个像元，再往外让出 pad 个格子
    xmin = x_min_points - half - pad * RES
    # 上边界 = 最北的点再往北半个像元，再往外让出 pad 个格子
    ymax = y_max_points + half + pad * RES

    # 列数 = 东西方向点的个数 + 两侧 pad
    ncol = (x_max_points - x_min_points) // RES + 1 + 2 * pad
    # 行数 = 南北方向点的个数 + 两侧 pad
    nrow = (y_max_points - y_min_points) // RES + 1 + 2 * pad

    meta = {
        "xmin": xmin,
        "ymax": ymax,
        "res": RES,
        "nrow": nrow,
        "ncol": ncol,
        "crs": CRS,
    }
    print("  网格：", nrow, "行 ×", ncol, "列")
    return meta


def point_indices(df, meta):
    """
    算出每个点落在网格的第几行、第几列。

    点在格子中心，所以：
    列号：从西往东数，格子的西边是 E - res/2，  col = (E - res/2 - xmin) // res
    行号：从北往南数，格子的北边是 N + res/2，  row = (ymax - (N + res/2)) // res
    """
    east = df["E_COORD"].to_numpy()
    north = df["N_COORD"].to_numpy()
    half = meta["res"] // 2

    cols = (east - half - meta["xmin"]) // meta["res"]
    rows = (meta["ymax"] - (north + half)) // meta["res"]

    rows = rows.astype(np.int64)
    cols = cols.astype(np.int64)

    # 检查 1：所有点都必须落在网格内部
    if rows.min() < 0 or rows.max() >= meta["nrow"]:
        raise ValueError("有点的行号超出网格范围，检查 build_grid_meta")
    if cols.min() < 0 or cols.max() >= meta["ncol"]:
        raise ValueError("有点的列号超出网格范围，检查 build_grid_meta")

    # 检查 2：不能有两个点落进同一个格子
    # 把 (row, col) 编码成一个整数，再看有没有重复
    cell_id = rows * meta["ncol"] + cols
    n_unique = len(np.unique(cell_id))
    if n_unique != len(cell_id):
        raise ValueError("有 " + str(len(cell_id) - n_unique) + " 个点落在同一格子里")

    return rows, cols


def to_grid(values, rows, cols, meta, fill, dtype):
    """
    把每个点的一个数值写进网格。

    先建一张全是 fill 的数组，再按 (rows, cols) 把 values 逐个填进去。
    没有点的格子（瑞士境外）保持 fill，作为无数据。
    """
    grid = np.full((meta["nrow"], meta["ncol"]), fill, dtype=dtype)
    grid[rows, cols] = values
    return grid


# ---------------------------------------------------------------------------
# 3. 森林重分类
# ---------------------------------------------------------------------------
def make_forest(lulc, forest_codes, nodata):
    """
    由 72 类栅格生成森林栅格：1 = 森林，0 = 非森林，nodata = 无数据。

    lulc 中 0 表示没有点（境外），有效类别是 1–72。
    """
    forest = np.full(lulc.shape, nodata, dtype=np.uint8)

    has_data = lulc > 0
    is_forest = np.isin(lulc, forest_codes)   # 每个像元的类别是否属于 forest_codes

    forest[has_data] = 0                       # 先把所有有数据的像元设成非森林
    forest[is_forest] = 1                      # 再把森林像元改成 1
    return forest


# ---------------------------------------------------------------------------
# 4. 研究范围 mask
# ---------------------------------------------------------------------------
def polygons_to_mask(gdf, meta):
    """
    把面要素栅格化成 0/1 mask。

    swissBOUNDARIES3D 的几何带 Z 值（高程），先用 force_2d() 去掉，
    rasterize 只需要平面坐标。
    默认规则：像元中心落在面内的像元 = 1。
    """
    geometries = gdf.geometry.force_2d()

    # rasterize 需要 (几何, 写入值) 这样的配对列表
    shapes = []
    for geom in geometries:
        shapes.append((geom, 1))

    mask = rasterize(
        shapes,
        out_shape=(meta["nrow"], meta["ncol"]),
        transform=utils.creatematrix(meta),
        fill=0,
        dtype="uint8",
    )
    return mask


def unit_masks(cfg, meta):
    """
    生成三个研究单元的 mask，返回字典 {"canton": ..., "urban": ..., "rural": ...}。

    两个时间点都用"当前"的市镇边界：1985 年以来伯尔尼合并过很多市镇，
    用同一套边界才能保证 A 和 B 比较的是同一块地。
    """
    gdb_path = cfg["paths"]["raw"] / cfg["files"]["boundaries"]
    masks = {}

    # --- 州 ---
    cantons = gpd.read_file(gdb_path, layer=CANTON_LAYER)
    canton = cantons[cantons["KANTONSNUMMER"] == cfg["units"]["canton"]]
    if len(canton) == 0:
        raise ValueError("找不到 KANTONSNUMMER = " + str(cfg["units"]["canton"]))
    print("  州：", canton["NAME"].iloc[0])
    masks["canton"] = polygons_to_mask(canton, meta)

    # --- 市镇 ---
    municipalities = gpd.read_file(gdb_path, layer=MUNICIPALITY_LAYER)
    for key in ["urban", "rural"]:
        bfs = cfg["units"][key]
        selected = municipalities[municipalities["BFS_NUMMER"] == bfs]
        if len(selected) == 0:
            raise ValueError("找不到 BFS_NUMMER = " + str(bfs))
        print("  市镇（" + key + "）：", selected["NAME"].iloc[0], "BFS", bfs)
        masks[key] = polygons_to_mask(selected, meta)

    return masks


# ---------------------------------------------------------------------------
# 5. 检查用的表格和图
# ---------------------------------------------------------------------------
def transition_table(df, rows, cols, mask_canton, col_a, col_b, out_path):
    """
    伯尔尼州内 A -> B 的 72 类转移矩阵：行 = 时间 A 的类别，列 = 时间 B 的类别。
    用来判断森林定义是否合理（例如 50 -> 55/56/57 的转移有多少）。
    """
    # 每个点所在格子是否在州内
    in_canton = mask_canton[rows, cols] == 1
    df_canton = df[in_canton]

    table = pd.crosstab(df_canton[col_a], df_canton[col_b])
    table.to_csv(out_path)

    # 在终端里打印森林相关的 50–60 类，方便快速查看
    print("  森林相关类别的转移（行 = A，列 = B）：")
    print(table.loc[50:60, 50:60].to_string())
    return table


def forest_summary(forest_a, forest_b, masks, df, rows, cols, cfg, out_path):
    """
    统计每个研究单元的森林面积（1 像元 = 1 ha）、占比和平均调查年份。
    平均调查年份之差就是后面 Task 3 外推时用的 Δt。
    """
    nodata = cfg["nodata"]
    year_a = df["FJ" + cfg["time_A"]].to_numpy()
    year_b = df["FJ" + cfg["time_B"]].to_numpy()

    records = []
    for unit_name in masks:
        mask = masks[unit_name] == 1

        # 单元内有数据的像元（排除 nodata）
        valid = mask & (forest_a != nodata)
        area_ha = int(valid.sum())

        forest_a_ha = int((mask & (forest_a == 1)).sum())
        forest_b_ha = int((mask & (forest_b == 1)).sum())

        # 平均调查年份：取落在该单元内的点
        point_in_unit = mask[rows, cols]
        mean_year_a = float(year_a[point_in_unit].mean())
        mean_year_b = float(year_b[point_in_unit].mean())

        record = {
            "unit": unit_name,
            "area_ha": area_ha,
            "forest_A_ha": forest_a_ha,
            "forest_B_ha": forest_b_ha,
            "forest_A_pct": round(100.0 * forest_a_ha / area_ha, 2),
            "forest_B_pct": round(100.0 * forest_b_ha / area_ha, 2),
            "change_ha": forest_b_ha - forest_a_ha,
            "mean_year_A": round(mean_year_a, 1),
            "mean_year_B": round(mean_year_b, 1),
            "delta_t": round(mean_year_b - mean_year_a, 1),
        }
        records.append(record)

    summary = pd.DataFrame(records)
    summary.to_csv(out_path, index=False)
    print(summary.to_string(index=False))
    return summary


def quicklook(forest_a, forest_b, mask_canton, out_path):
    """
    伯尔尼州森林变化的快速预览：不变的森林 / 森林减少 / 森林增加。
    只用于检查，不是 Task 5 的正式地图。
    """
    # 裁到州的外接矩形，图会大很多
    rows_in = np.where(mask_canton.any(axis=1))[0]
    cols_in = np.where(mask_canton.any(axis=0))[0]
    r0 = rows_in.min()
    r1 = rows_in.max() + 1
    c0 = cols_in.min()
    c1 = cols_in.max() + 1

    fa = forest_a[r0:r1, c0:c1]
    fb = forest_b[r0:r1, c0:c1]
    inside = mask_canton[r0:r1, c0:c1] == 1

    # 变化类别：NaN = 州外，0 = 非森林不变，1 = 森林不变，2 = 减少，3 = 增加
    change = np.full(fa.shape, np.nan)
    change[inside & (fa == 0) & (fb == 0)] = 0
    change[inside & (fa == 1) & (fb == 1)] = 1
    change[inside & (fa == 1) & (fb == 0)] = 2
    change[inside & (fa == 0) & (fb == 1)] = 3

    colours = ListedColormap(["#eeeeee", "#9fbf8f", "#d7301f", "#1a9850"])
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.imshow(change, cmap=colours, vmin=-0.5, vmax=3.5, interpolation="nearest")
    ax.set_title("Quicklook: forest A→B (grey non-forest, light green stable, red loss, dark green gain)")
    ax.set_axis_off()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 入口：main 只调用这个函数
# ---------------------------------------------------------------------------
def run(cfg):
    processed = cfg["paths"]["processed"]
    tables = cfg["paths"]["tables"]
    figures = cfg["paths"]["figures"]
    nodata = cfg["nodata"]
    col_a = "AS" + cfg["time_A"] + "_72"
    col_b = "AS" + cfg["time_B"] + "_72"

    # 1. 读点
    df = loadptgrid(cfg)


    # 2. 建网格，算每个点的行列号
    meta = build_grid_meta(df)
    rows, cols = point_indices(df, meta)
    utils.save_meta(meta, processed / "grid_meta.json")

    # 3. 72 类栅格（0 = 无数据）
    #    fillna(0)：万一某个点没有类别，当作无数据处理
    values_a = df[col_a].fillna(0).to_numpy().astype(np.uint8)
    values_b = df[col_b].fillna(0).to_numpy().astype(np.uint8)
    lulc_a = to_grid(values_a, rows, cols, meta, fill=0, dtype=np.uint8)
    lulc_b = to_grid(values_b, rows, cols, meta, fill=0, dtype=np.uint8)
    utils.save_tif(lulc_a, processed / "lulc_A.tif", meta, nodata=0)
    utils.save_tif(lulc_b, processed / "lulc_B.tif", meta, nodata=0)

    # 4. 森林栅格
    forest_a = make_forest(lulc_a, cfg["forest_codes"], nodata)
    forest_b = make_forest(lulc_b, cfg["forest_codes"], nodata)
    utils.save_tif(forest_a, processed / "forest_A.tif", meta, nodata=nodata)
    utils.save_tif(forest_b, processed / "forest_B.tif", meta, nodata=nodata)

    # 5. 研究范围 mask
    masks = unit_masks(cfg, meta)
    for unit_name in masks:
        utils.save_tif(masks[unit_name], processed / ("mask_" + unit_name + ".tif"), meta, nodata=None)

    # 6. 检查用的表和图
    transition_table(df, rows, cols, masks["canton"], col_a, col_b, tables / "T1_transition_72.csv")
    forest_summary(forest_a, forest_b, masks, df, rows, cols, cfg, tables / "T1_forest_summary.csv")
    quicklook(forest_a, forest_b, masks["canton"], figures / "T1_quicklook.png")

    print("  Task 1 完成。")
    