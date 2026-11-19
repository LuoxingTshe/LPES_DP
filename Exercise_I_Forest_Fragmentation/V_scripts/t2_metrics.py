"""
t2_metrics.py – Task 2：破碎化几何 (FG1)、三类重分类、meff (CBC)、ED、PD、LPI。

思路：重活全部交给公开库，本文件只负责"把库函数按顺序串起来"。
    geopandas.read_file(where=...)    读矢量时直接用 SQL 筛选，不在 pandas 里筛
    geopandas.dissolve / clip          合并亚区、裁到伯尔尼州
    rasterio.features.rasterize        道路/铁路线 -> 栅格
    scipy.ndimage.label / sum_labels   斑块识别 + 每个斑块的面积（用于 CBC meff）
    rasterio.mask.mask                 按单元边界裁剪栅格
    pylandstats.Landscape              ED、PD、LPI、斑块数、平均斑块面积、meff(CUT)

读取（Task 1 的输出）：IV_data_processed/lulc_A.tif, lulc_B.tif（同时作为参考栅格提供网格信息）
读取（原始数据）    ：swissTLMRegio 道路和铁路、swissBOUNDARIES3D、生物地理分区
写出：
    IV_data_processed/barrier_lines.tif        道路/铁路线栅格（0/1）
    IV_data_processed/landscape_A.tif, _B.tif  0 = 其他, 1 = 森林, 2 = 障碍, 255 = 无数据
    IV_data_processed/units.geojson            所有统计单元的边界（Task 3/4 还会用）
    VI_outputs/tables/T2_metrics.csv           长表：每个单元 × 每个时间一行
    VI_outputs/tables/T2_change.csv            宽表：A、B 并排 + 变化百分比
    VI_outputs/figures/T2_landscape_quicklook.png
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
from rasterio.features import rasterize
from scipy import ndimage
import pylandstats as pls
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from V_scripts import utils, t1_grid

# 三类景观的编码
OTHER = 0      # 非障碍的其他用地（农田、草地、水体、岩石……）
FOREST = 1
BARRIER = 2

ROAD_LAYER = "TLMRegio_Road"
RAIL_LAYER = "TLMRegio_Railway"
REGION_LAYER = "N2020_Revision_BiogeoRegion"
REGION_NAME_TABLE = "RegionName_Catalogue"

# 输出表里要比较 A/B 的指标
METRICS = [
    "forest_ha",
    "meff_forest_cbc_km2",
    "meff_forest_cut_km2",
    "meff_open_cbc_km2",
    "ED_m_per_ha",
    "PD_per_100ha",
    "LPI_pct",
    "n_patches",
    "AREA_MN_ha",
]


# ---------------------------------------------------------------------------
# 1. 统计单元：州、两个市镇、州内的生物地理区
# ---------------------------------------------------------------------------
def sql_in(codes):
    """把 [1, 2, 3] 变成字符串 "1, 2, 3"，用在 SQL 的 IN (...) 里。"""
    text = ""
    for code in codes:
        if text != "":
            text = text + ", "
        text = text + str(code)
    return text


def load_units(cfg):
    """
    返回一个 GeoDataFrame，每行一个统计单元，列：unit（名称）、scale（尺度）、geometry。

    where= 参数会把 SQL 条件交给 GDAL，在读文件时就筛掉不要的要素，
    不用先把整张表读进来再用 pandas 筛选。
    """
    gdb = cfg["paths"]["raw"] / cfg["files"]["boundaries"]
    biogeo = cfg["paths"]["raw"] / cfg["files"]["biogeo"]

    canton = gpd.read_file(gdb, layer=t1_grid.CANTON_LAYER,
                           where="KANTONSNUMMER = " + str(cfg["units"]["canton"]))
    urban = gpd.read_file(gdb, layer=t1_grid.MUNICIPALITY_LAYER,
                          where="BFS_NUMMER = " + str(cfg["units"]["urban"]))
    rural = gpd.read_file(gdb, layer=t1_grid.MUNICIPALITY_LAYER,
                          where="BFS_NUMMER = " + str(cfg["units"]["rural"]))

    # 生物地理区：Mittelland 等区由几个亚区组成 -> dissolve 按 RegionName 合并成一个面
    regions = gpd.read_file(biogeo, layer=REGION_LAYER)
    regions = regions.dissolve(by="RegionName", as_index=False)
    regions = regions[regions["RegionName"].isin(cfg["regions"])]

    # 区名表没有几何，读出来就是普通 DataFrame；merge 相当于 GIS 里的"属性连接"
    names = gpd.read_file(biogeo, layer=REGION_NAME_TABLE)
    regions = regions.merge(names, left_on="RegionName", right_on="CODERegionName")

    # swissBOUNDARIES3D 带 Z 值，CRS 是"LV95 + LN02 高程"的复合坐标系。
    # 去掉 Z 之后平面坐标就是 LV95，用 set_crs 把 CRS 标成纯 EPSG:2056，两个图层才能 clip。
    canton["geometry"] = canton.geometry.force_2d()
    canton = canton.set_crs(t1_grid.CRS, allow_override=True)
    regions["geometry"] = regions.geometry.force_2d()
    regions = gpd.clip(regions, canton)

    unit_names = []
    scales = []
    geometries = []

    unit_names.append("Canton of Bern")
    scales.append("canton")
    geometries.append(canton.geometry.union_all())

    unit_names.append(urban["NAME"].iloc[0] + " (urban)")
    scales.append("municipality")
    geometries.append(urban.geometry.force_2d().union_all())

    unit_names.append(rural["NAME"].iloc[0] + " (rural)")
    scales.append("municipality")
    geometries.append(rural.geometry.force_2d().union_all())

    for i in range(len(regions)):
        unit_names.append(regions["DERegionName"].iloc[i] + " (BE part)")
        scales.append("region")
        geometries.append(regions.geometry.iloc[i])

    units = gpd.GeoDataFrame({"unit": unit_names, "scale": scales},
                             geometry=geometries, crs=t1_grid.CRS)
    for i in range(len(units)):
        print("  ", units["unit"].iloc[i], round(units.geometry.iloc[i].area / 1e6, 1), "km²")
    return units


# ---------------------------------------------------------------------------
# 2. FG1 障碍：Arealstatistik 的建成区/交通用地 + swissTLMRegio 主要道路和铁路
# ---------------------------------------------------------------------------
def barrier_lines(cfg, profile):
    """
    把主要道路和铁路栅格化成 0/1 数组（1 = 线经过的像元）。

    为什么还要矢量线：Arealstatistik 是每公顷一个抽样点，一条 25 m 宽的高速公路
    只有一部分像元的点恰好落在路面上，栅格里的道路是断断续续的，挡不住斑块连通。
    矢量线栅格化后是连续的。

    all_touched：
        False（默认）= 只烧线"主要经过"的像元，得到 8 邻接的细线。
                       8 邻接的线足以切断 4 邻域的斑块。
        True         = 线碰到的像元全部烧成 1，线变粗，变成 4 邻接，
                       用 8 邻域识别斑块时才需要（否则斑块会从对角"漏"过去）。
    """
    gdb = cfg["paths"]["raw"] / cfg["files"]["tlmregio"]

    # CONSTRUCT = 2 是隧道：隧道上方的地面是连通的，不算障碍
    road_where = "OBJVAL IN (" + sql_in(cfg["road_codes"]) + ") AND CONSTRUCT <> 2"
    rail_where = "OBJVAL IN (" + sql_in(cfg["rail_codes"]) + ") AND CONSTRUCT <> 2"
    roads = gpd.read_file(gdb, layer=ROAD_LAYER, where=road_where)
    rails = gpd.read_file(gdb, layer=RAIL_LAYER, where=rail_where)
    print("  道路线段：", len(roads), "  铁路线段：", len(rails))

    geometries = list(roads.geometry.force_2d()) + list(rails.geometry.force_2d())
    all_touched = cfg["neighbourhood"] == 8

    lines = rasterize(
        geometries,                                 # 只给几何时，每条线写入 default_value
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        default_value=1,
        all_touched=all_touched,
        dtype="uint8",
    )
    return lines


def make_landscape(lulc, lines, cfg):
    """
    72 类 + 障碍线 -> 三类景观。后写的覆盖先写的，所以障碍优先于森林：
    一条主路穿过森林，那一格就是障碍，森林在这里被切开。
    """
    nodata = cfg["nodata"]
    has_data = lulc > 0

    land = np.full(lulc.shape, nodata, dtype=np.uint8)
    land[has_data] = OTHER
    land[np.isin(lulc, cfg["forest_codes"])] = FOREST
    land[np.isin(lulc, cfg["barrier_codes"])] = BARRIER
    land[(lines == 1) & has_data] = BARRIER
    return land


# ---------------------------------------------------------------------------
# 3. meff，CBC 方法 (Moser et al. 2007)
# ---------------------------------------------------------------------------
def label_patches(binary, neighbourhood):
    """
    在全瑞士网格上识别斑块，返回：
        labels    : 和 binary 同形状，每个像元写着它所属斑块的编号（0 = 不是斑块）
        patch_ids : 1, 2, ..., n
        full_size : 每个斑块的完整面积（像元数），不管它有没有伸出统计单元

    在全瑞士而不是单元内部识别斑块，就是 CBC 的关键：
    跨越单元边界的斑块保留它的完整面积 A_i_cmpl。
    """
    if neighbourhood == 4:
        structure = ndimage.generate_binary_structure(2, 1)   # 十字形：上下左右
    else:
        structure = ndimage.generate_binary_structure(2, 2)   # 3×3：再加四个对角
    labels, n = ndimage.label(binary, structure=structure)
    patch_ids = np.arange(1, n + 1)
    full_size = ndimage.sum_labels(binary, labels, patch_ids)
    return labels, patch_ids, full_size


def meff_cbc(labels, patch_ids, full_size, unit_mask):
    """
    CBC 有效网格尺寸：meff = (1 / A_total) × Σ A_i × A_i_cmpl

        A_i       : 斑块 i 落在单元内的面积  -> 对单元 mask 按斑块编号求和
        A_i_cmpl  : 斑块 i 的完整面积        -> label_patches 已经算好
        A_total   : 单元总面积（有数据的像元，含障碍）

    1 像元 = 1 ha，结果从 ha 换算成 km²（÷ 100）。
    """
    inside_size = ndimage.sum_labels(unit_mask, labels, patch_ids)
    total_area = unit_mask.sum()
    meff_ha = np.sum(inside_size * full_size) / total_area
    return meff_ha / 100.0


# ---------------------------------------------------------------------------
# 4. 其他指标交给 pylandstats
# ---------------------------------------------------------------------------
def clip_to_unit(tif_path, geometry, nodata):
    """
    按单元边界裁剪栅格。crop=True 会把结果裁到单元的外接矩形，
    边界外的像元填 nodata。规则和 rasterize 一样：像元中心在面内才算在内。
    """
    with rasterio.open(tif_path) as src:
        clipped, clipped_transform = rasterio.mask.mask(src, [geometry], crop=True, nodata=nodata)
    return clipped[0], clipped_transform     # mask 返回 (波段, 行, 列)，取第 1 个波段


def pylandstats_metrics(array, transform, cfg):
    """
    用 pylandstats 计算森林这一类 (class_val=FOREST) 的类别级指标。
    nodata 像元不计入景观面积；count_boundary=False 表示和 nodata 相邻的边不算边缘。
    """
    landscape = pls.Landscape(
        array,
        res=(transform.a, -transform.e),            # 像元宽、高直接从 transform 读（e 是负数）
        nodata=cfg["nodata"],
        transform=transform,
        neighborhood_rule=str(cfg["neighbourhood"]),
    )
    result = {
        "forest_ha": landscape.total_area(class_val=FOREST),
        "ED_m_per_ha": landscape.edge_density(class_val=FOREST),
        "PD_per_100ha": landscape.patch_density(class_val=FOREST),
        "LPI_pct": landscape.largest_patch_index(class_val=FOREST),
        "n_patches": landscape.number_of_patches(class_val=FOREST),
        "AREA_MN_ha": landscape.area_mn(class_val=FOREST),
        # pylandstats 的 meff 是"切割法"(CUT)：伸出单元的斑块被截断，用来和 CBC 对比
        "meff_forest_cut_km2": landscape.effective_mesh_size(class_val=FOREST) / 100.0,
    }
    return result


# ---------------------------------------------------------------------------
# 5. 结果表和检查图
# ---------------------------------------------------------------------------
def change_table(table, out_path):
    """长表 -> 宽表：每个单元一行，每个指标给出 A、B 和变化百分比。"""
    rows = []
    for unit_name in table["unit"].unique():
        a = table[(table["unit"] == unit_name) & (table["time"] == "A")].iloc[0]
        b = table[(table["unit"] == unit_name) & (table["time"] == "B")].iloc[0]
        row = {"unit": unit_name, "scale": a["scale"]}
        for m in METRICS:
            row[m + "_A"] = a[m]
            row[m + "_B"] = b[m]
            row[m + "_chg_pct"] = round(100.0 * (b[m] - a[m]) / a[m], 1)
        rows.append(row)
    wide = pd.DataFrame(rows)
    wide.to_csv(out_path, index=False)
    return wide


def quicklook(paths, labels, canton_geometry, nodata, out_path):
    """伯尔尼州 A、B 两期三类景观并排显示，用来目视检查障碍线是否连续。"""
    colours = ListedColormap(["#e8e4d8", "#4c8c4a", "#c0392b"])   # 其他 / 森林 / 障碍
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    for i in range(2):
        array, _ = clip_to_unit(paths[i], canton_geometry, nodata)
        shown = np.ma.masked_equal(array, nodata)                   # nodata 不上色
        axes[i].imshow(shown, cmap=colours, vmin=-0.5, vmax=2.5, interpolation="nearest")
        axes[i].set_title(labels[i] + ": beige other, green forest, red barrier (FG1)")
        axes[i].set_axis_off()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def run(cfg):
    processed = cfg["paths"]["processed"]
    nodata = cfg["nodata"]
    nb = cfg["neighbourhood"]
    # 参考栅格：lulc_A.tif 的 profile 提供 transform、crs、行列数，后面所有输出都照它写
    _, profile = utils.read_tif(processed / "lulc_A.tif")

    # 1. 统计单元
    units = load_units(cfg)
    units.to_file(processed / "units.geojson", driver="GeoJSON")

    # 2. 障碍线（两期共用同一套路网，见局限性说明）
    lines = barrier_lines(cfg, profile)
    utils.save_tif(lines, processed / "barrier_lines.tif", profile, nodata=None)

    # 每个单元的 mask 只算一次，两期都用
    unit_masks = []
    for i in range(len(units)):
        unit_masks.append(t1_grid.polygons_to_mask(units.iloc[[i]], profile) == 1)

    records = []
    landscape_paths = []
    for time in ["A", "B"]:
        survey = "AS" + cfg["time_" + time]
        print("  ---", time, survey)

        # 3. 三类景观
        lulc, _ = utils.read_tif(processed / ("lulc_" + time + ".tif"))   # 只要数组，profile 已经有了
        land = make_landscape(lulc, lines, cfg)
        path = processed / ("landscape_" + time + ".tif")
        utils.save_tif(land, path, profile, nodata=nodata)
        landscape_paths.append(path)

        # 4. 全瑞士识别两种斑块
        #    森林斑块：被非森林和障碍切开 -> 森林本身的破碎化
        #    开放斑块：森林 + 其他，只被障碍切开 -> Jaeger et al. (2008) 意义上的景观破碎化
        forest_labels, forest_ids, forest_size = label_patches(land == FOREST, nb)
        open_labels, open_ids, open_size = label_patches((land == FOREST) | (land == OTHER), nb)
        print("  森林斑块数（全瑞士）：", len(forest_ids), "  开放斑块数：", len(open_ids))

        for i in range(len(units)):
            unit_mask = unit_masks[i] & (land != nodata)
            record = {
                "unit": units["unit"].iloc[i],
                "scale": units["scale"].iloc[i],
                "time": time,
                "survey": survey,
                "area_ha": int(unit_mask.sum()),
                "meff_forest_cbc_km2": meff_cbc(forest_labels, forest_ids, forest_size, unit_mask),
                "meff_open_cbc_km2": meff_cbc(open_labels, open_ids, open_size, unit_mask),
            }
            array, transform = clip_to_unit(path, units.geometry.iloc[i], nodata)
            record.update(pylandstats_metrics(array, transform, cfg))
            records.append(record)
            print("   ", record["unit"], "完成")

    # 5. 输出
    table = pd.DataFrame(records).round(3)
    table.to_csv(cfg["paths"]["tables"] / "T2_metrics.csv", index=False)
    wide = change_table(table, cfg["paths"]["tables"] / "T2_change.csv")
    quicklook(landscape_paths, ["AS" + cfg["time_A"], "AS" + cfg["time_B"]],
              units.geometry.iloc[0], nodata, cfg["paths"]["figures"] / "T2_landscape_quicklook.png")

    show = ["unit", "time", "forest_ha", "meff_forest_cbc_km2", "meff_forest_cut_km2",
            "meff_open_cbc_km2", "ED_m_per_ha", "PD_per_100ha", "LPI_pct"]
    print(table[show].to_string(index=False))
    print("  Task 2 完成。")
