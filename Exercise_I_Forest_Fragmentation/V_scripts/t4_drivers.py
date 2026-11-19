"""
t4_drivers.py – Task 4：两个生物地理区里森林变化的驱动因子（logistic 回归）。

步骤：
    1. 量化两个区的森林变化（总增加、总减少、净变化），由净变化决定响应变量：
           净减少 -> "loss" 模型：A 期是森林的像元，1 = B 期变成非森林，0 = 仍是森林
           净增加 -> "gain" 模型：A 期是非森林的像元，1 = B 期变成森林，0 = 仍是非森林
    2. 在 100 m 网格上生成候选预测变量（环境、社会经济、土地利用三组）
    3. 按固定间隔系统抽样，减轻空间自相关
    4. 筛选：单变量 AIC 排序 -> 相关系数 |r| > r_max 的变量对去掉一个 -> VIF -> 向后 AIC，最多留 max_predictors 个
    5. statsmodels.Logit 拟合，报告系数、95% CI、odds ratio、AUC，残差 Moran's I

用到的公开库：
    rasterio.warp.reproject             25 m DEM（LV03）-> 100 m 网格（LV95），取平均值
    rasterio.features.rasterize         河流、湖泊、建设区、可达性分区 -> 栅格
    rasterio.transform.rowcol           STATPOP 百米格坐标 -> 行列号
    scipy.ndimage.distance_transform_edt  到最近目标像元的距离
    scipy.ndimage.uniform_filter        5×5 窗口平均（森林比例、人口、居住区比例）
    statsmodels                         Logit、VIF
    sklearn.metrics.roc_auc_score       AUC
    libpysal + esda                     残差的 Moran's I

读取：IV_data_processed/lulc_A.tif, lulc_B.tif, forest_A.tif, forest_B.tif, barrier_lines.tif, units.geojson
      原始数据：DHM25、swissTLMRegio 水系、STATPOP 2017、ARE 可达性、Bauzonen 2017
写出：
      IV_data_processed/change_AB.tif     0 稳定非森林, 1 稳定森林, 2 增加, 3 减少, 255 无数据（Task 5 制图用）
      IV_data_processed/dem_*.tif         DEM 派生变量的缓存（第一次运行约 1 分钟，以后直接读）
      VI_outputs/tables/T4_forest_change.csv
      VI_outputs/tables/T4_correlation_<区>.csv, T4_selection_<区>.csv
      VI_outputs/tables/T4_coefficients.csv, T4_model_summary.csv
      VI_outputs/figures/T4_odds_ratios.png
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import pyogrio
import rasterio
from rasterio.features import rasterize
from rasterio.transform import rowcol
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from scipy import ndimage
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_auc_score
import libpysal
import esda
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from V_scripts import utils, t1_grid, t2_metrics

DEM_CRS = "EPSG:21781"     # DHM25 的 ASCII 网格是 LV03，文件里没写坐标系，要自己指定

# change_AB.tif 的编码
STABLE_NON = 0
STABLE_FOREST = 1
GAIN = 2
LOSS = 3

# 每个变量的分组和单位，写进结果表方便写报告
VARIABLES = {
    "elev":         ("environmental", "m a.s.l."),
    "slope":        ("environmental", "degrees"),
    "tri":          ("environmental", "m (Riley TRI, 25 m)"),
    "tpi":          ("environmental", "m (500 m window)"),
    "dist_water":   ("environmental", "m"),
    "pop_500m":     ("socio-economic", "log(1 + persons in 500 m window), 2017"),
    "access_road":  ("socio-economic", "log(1 + road accessibility index)"),
    "dist_bauzone": ("socio-economic", "m to building zone 2017"),
    "dist_mainroad": ("socio-economic", "m to main road / railway"),
    "settl_chg":    ("socio-economic", "settlement share change A->B, %-points"),
    "forest_share": ("land use", "% forest in 500 m window at A"),
    "dist_edge":    ("land use", "m to forest/non-forest boundary at A"),
    "patch_size":   ("land use", "log10(ha) of forest patch at A"),
}


# ---------------------------------------------------------------------------
# 1. 森林变化和响应变量
# ---------------------------------------------------------------------------
def change_raster(forest_a, forest_b, nodata):
    """两期森林 -> 四类变化栅格。"""
    change = np.full(forest_a.shape, nodata, dtype=np.uint8)
    valid = (forest_a != nodata) & (forest_b != nodata)
    change[valid & (forest_a == 0) & (forest_b == 0)] = STABLE_NON
    change[valid & (forest_a == 1) & (forest_b == 1)] = STABLE_FOREST
    change[valid & (forest_a == 0) & (forest_b == 1)] = GAIN
    change[valid & (forest_a == 1) & (forest_b == 0)] = LOSS
    return change


def forest_change_table(change, units, unit_masks, cfg):
    """每个单元的总增加、总减少、净变化（ha）。净变化决定这个区用哪种模型。"""
    rows = []
    for i in range(len(units)):
        inside = unit_masks[i]
        gain = int(np.sum(inside & (change == GAIN)))
        loss = int(np.sum(inside & (change == LOSS)))
        forest_a = int(np.sum(inside & ((change == STABLE_FOREST) | (change == LOSS))))
        model = "gain"
        if gain - loss < 0:
            model = "loss"
        rows.append({
            "unit": units["unit"].iloc[i],
            "forest_A_ha": forest_a,
            "forest_B_ha": forest_a + gain - loss,
            "gross_gain_ha": gain,
            "gross_loss_ha": loss,
            "net_ha": gain - loss,
            "net_pct": round(100.0 * (gain - loss) / forest_a, 2),
            "model": model,
        })
    table = pd.DataFrame(rows)
    table.to_csv(cfg["paths"]["tables"] / "T4_forest_change.csv", index=False)
    return table


# ---------------------------------------------------------------------------
# 2. 预测变量
# ---------------------------------------------------------------------------
def dem_predictors(cfg, profile, canton):
    """
    高程、坡度、TRI：在 25 m DEM 上计算，再用"平均值"重采样到 100 m 网格。
    先算再聚合，而不是先聚合再算：100 m 的平均高程会把小尺度的陡坡抹平。

    只读伯尔尼州外接矩形（外扩 2 km）这一块，不读全瑞士。
    结果缓存在 dem_*.tif，以后直接读，ASCII 网格解析一次要将近一分钟。
    """
    processed = cfg["paths"]["processed"]
    names = ["elev", "slope", "tri"]
    cached = True
    for name in names:
        if not (processed / ("dem_" + name + ".tif")).exists():
            cached = False
    if cached:
        print("  DEM 变量：读缓存")
        result = {}
        for name in names:
            result[name], _ = utils.read_tif(processed / ("dem_" + name + ".tif"))
        return result

    print("  DEM 变量：从 DHM25 计算（约 1 分钟）")
    path = "/vsizip/" + str(cfg["paths"]["raw"] / cfg["files"]["dem"])
    xmin, ymin, xmax, ymax = canton.total_bounds
    buffer = 2000
    bounds_lv03 = transform_bounds(t1_grid.CRS, DEM_CRS, xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer)

    with rasterio.open(path) as src:
        window = from_bounds(*bounds_lv03, transform=src.transform).round_offsets().round_lengths()
        z = src.read(1, window=window).astype("float32")
        z_transform = src.window_transform(window)
        z[z == src.nodata] = np.nan
    cell = z_transform.a     # 25 m

    # 坡度：两个方向的高差梯度 -> 最大坡度角
    dz_dy, dz_dx = np.gradient(z, cell)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))

    # TRI (Riley et al. 1999)：中心像元和 8 个邻居高差平方和的平方根
    nrow, ncol = z.shape
    padded = np.pad(z, 1, mode="edge")
    total = np.zeros(z.shape, dtype="float32")
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            neighbour = padded[1 + dr:1 + dr + nrow, 1 + dc:1 + dc + ncol]
            total = total + (neighbour - z) ** 2
    tri = np.sqrt(total)

    result = {}
    sources = {"elev": z, "slope": slope, "tri": tri}
    for name in names:
        out = np.full((profile["height"], profile["width"]), np.nan, dtype="float32")
        reproject(
            source=sources[name], destination=out,
            src_transform=z_transform, src_crs=DEM_CRS, src_nodata=np.nan,
            dst_transform=profile["transform"], dst_crs=profile["crs"], dst_nodata=np.nan,
            resampling=Resampling.average,          # 16 个 25 m 像元取平均 -> 1 个 100 m 像元
        )
        utils.save_tif(out, processed / ("dem_" + name + ".tif"), profile, nodata=np.nan)
        result[name] = out
    return result


def window_mean(array, size):
    """size × size 窗口平均。窗口超出网格的部分按 0 计。"""
    return ndimage.uniform_filter(array.astype("float32"), size=size, mode="constant", cval=0.0)


def window_nanmean(array, size):
    """
    忽略 NaN 的窗口平均：分子 = 把 NaN 当 0 的窗口和，分母 = 窗口里有效像元的个数。
    uniform_filter 是按行、按列分两次累加的，直接用会让一个 NaN 沿整行整列扩散。
    """
    valid = np.logical_not(np.isnan(array))
    filled = np.where(valid, array, 0.0)
    total = window_mean(filled, size)
    count = window_mean(valid, size)
    result = np.full(array.shape, np.nan, dtype="float32")
    np.divide(total, count, out=result, where=count > 0)      # 只在 count > 0 的地方做除法
    return result


def fill_nearest(array):
    """
    NaN 像元取最近一个有值像元的值。
    distance_transform_edt 加上 return_indices=True，会同时返回"最近的非目标像元"的行列号。
    """
    missing = np.isnan(array)
    _, (near_rows, near_cols) = ndimage.distance_transform_edt(missing, return_indices=True)
    return array[near_rows, near_cols]


def distance_to(target, cell=100.0):
    """每个像元到最近 target 像元的距离（m）。target 本身距离为 0。"""
    return ndimage.distance_transform_edt(np.logical_not(target)) * cell


def burn(geometries, profile):
    """几何 -> 0/1 栅格（像元中心落在面内，或被线经过）。"""
    return rasterize(geometries, out_shape=(profile["height"], profile["width"]),
                     transform=profile["transform"], fill=0, default_value=1, dtype="uint8")


def build_predictors(cfg, profile, canton, lulc_a, lulc_b, forest_a):
    """返回 {变量名: 100 m 网格上的数组}。"""
    raw = cfg["paths"]["raw"]
    processed = cfg["paths"]["processed"]
    size = cfg["window"]
    nodata = cfg["nodata"]
    bbox = tuple(canton.total_bounds)
    pred = {}

    # --- 环境 ---
    pred.update(dem_predictors(cfg, profile, canton))
    # TPI：像元平均高程 - 周围 500 m 窗口平均高程。正 = 山脊、凸地，负 = 谷底、洼地
    pred["tpi"] = pred["elev"] - window_nanmean(pred["elev"], size)

    tlm = raw / cfg["files"]["tlmregio"]
    rivers = gpd.read_file(tlm, layer="TLMRegio_FlowingWater", bbox=bbox)
    lakes = gpd.read_file(tlm, layer="TLMRegio_Lake", bbox=bbox)
    water = burn(list(rivers.geometry.force_2d()) + list(lakes.geometry.force_2d()), profile)
    pred["dist_water"] = distance_to(water == 1)

    # --- 社会经济 ---
    # STATPOP 2017：每个有人居住的百米格一行，坐标是格子西南角，+50 m 取格子中心
    pop = pyogrio.read_dataframe(raw / cfg["files"]["statpop"], layer=cfg["statpop_layer"],
                                 columns=["E_KOORD", "N_KOORD", "BBTOT"], read_geometry=False)
    rows, cols = rowcol(profile["transform"], pop["E_KOORD"].to_numpy() + 50, pop["N_KOORD"].to_numpy() + 50)
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    inside = (rows >= 0) & (rows < profile["height"]) & (cols >= 0) & (cols < profile["width"])
    pop_grid = np.zeros((profile["height"], profile["width"]), dtype="float32")
    pop_grid[rows[inside], cols[inside]] = pop["BBTOT"].to_numpy()[inside]
    pred["pop_500m"] = np.log1p(window_mean(pop_grid, size) * size * size)   # 窗口内总人数

    # 可达性：交通分区面，每个区一个值 -> rasterize 时直接写这个值
    access = gpd.read_file(raw / cfg["files"]["access"], bbox=bbox)
    shapes = []
    for i in range(len(access)):
        shapes.append((access.geometry.iloc[i], float(access[cfg["access_field"]].iloc[i])))
    access_grid = rasterize(shapes, out_shape=(profile["height"], profile["width"]),
                            transform=profile["transform"], fill=np.nan, dtype="float32")
    # 交通分区只覆盖有人居住的地方（伯尔尼州约 68%），山区和大片森林里没有分区。
    # 这些像元取最近一个分区的值：可达性在空间上是连续变化的，最近分区是合理的近似。
    pred["access_road"] = np.log1p(fill_nearest(access_grid))

    zones = gpd.read_file(raw / cfg["files"]["bauzonen"], bbox=bbox)
    pred["dist_bauzone"] = distance_to(burn(list(zones.geometry), profile) == 1)

    lines, _ = utils.read_tif(processed / "barrier_lines.tif")     # Task 2 的主要道路和铁路
    pred["dist_mainroad"] = distance_to(lines == 1)

    settl_a = window_mean(np.isin(lulc_a, cfg["settlement_codes"]), size)
    settl_b = window_mean(np.isin(lulc_b, cfg["settlement_codes"]), size)
    pred["settl_chg"] = 100.0 * (settl_b - settl_a)

    # --- 土地利用（A 期） ---
    is_forest = forest_a == 1
    pred["forest_share"] = 100.0 * window_mean(is_forest, size)

    # 到森林边界的距离：森林像元量到最近的非森林，非森林像元量到最近的森林。
    # 国界外（nodata）和 Task 3 一样，不当作边界。
    dist_in_forest = ndimage.distance_transform_edt(forest_a != 0) * 100.0
    dist_outside = ndimage.distance_transform_edt(forest_a != 1) * 100.0
    pred["dist_edge"] = np.where(is_forest, dist_in_forest, dist_outside)

    # 所在森林斑块的面积（只对森林像元有意义）
    labels, _, full_size = t2_metrics.label_patches(is_forest, cfg["neighbourhood"])
    size_by_label = np.concatenate([[np.nan], full_size])     # 编号 0 = 不是森林 -> NaN
    pred["patch_size"] = np.log10(size_by_label[labels])

    return pred


# ---------------------------------------------------------------------------
# 3. 抽样
# ---------------------------------------------------------------------------
def sample_region(region_mask, change, lulc_a, model, pred, cfg):
    """
    在一个区里按 sample_step 的间隔取像元（行号、列号都是 step 的倍数），组成回归用的表。
        loss 模型：A 期是森林的像元
        gain 模型：A 期是非森林、并且属于 gain_eligible_codes 的像元
    """
    step = cfg["sample_step"]
    row_index, col_index = np.indices(change.shape)
    on_grid = (row_index % step == 0) & (col_index % step == 0)

    if model == "loss":
        population = (change == STABLE_FOREST) | (change == LOSS)
        event = change == LOSS
    else:
        population = ((change == STABLE_NON) | (change == GAIN)) & np.isin(lulc_a, cfg["gain_eligible_codes"])
        event = change == GAIN

    take = region_mask & population & on_grid
    rows, cols = np.nonzero(take)

    data = {"row": rows, "col": cols, "y": event[rows, cols].astype(int)}
    for name in cfg["candidates"]:
        if model == "gain" and name == "patch_size":
            continue                       # 非森林像元没有"所在森林斑块"
        data[name] = pred[name][rows, cols]
    table = pd.DataFrame(data).dropna()
    return table


# ---------------------------------------------------------------------------
# 4. 变量筛选
# ---------------------------------------------------------------------------
def standardise(table, names):
    """z 分数：减均值、除标准差。系数变成"每增加 1 个标准差"，变量之间才能比较。"""
    z = table[names].copy()
    for name in names:
        z[name] = (z[name] - z[name].mean()) / z[name].std()
    return z


def fit_logit(y, X):
    """加上截距后拟合 logistic 回归，disp=0 不打印迭代过程。"""
    return sm.Logit(y, sm.add_constant(X)).fit(disp=0, maxiter=200)


def vif_table(X):
    """每个变量的方差膨胀因子 VIF（含截距的设计矩阵，截距本身不报告）。"""
    design = sm.add_constant(X)
    values = {}
    for i in range(1, design.shape[1]):
        values[design.columns[i]] = variance_inflation_factor(design.values, i)
    return values


def select_predictors(table, names, cfg, key):
    """
    四步筛选，每一步的结果写进 T4_selection_<区>.csv，报告里可以直接引用：
      1. 单变量模型的 AIC：越小说明这个变量单独解释力越强
      2. 按 AIC 从小到大依次检查，和已保留变量的 |r| > r_max 就去掉（保留解释力更强的那个）
      3. VIF > vif_max 的变量逐个去掉（每次去掉最大的那个）
      4. 向后 AIC：去掉某个变量能让 AIC 变小就去掉；变量数超过 max_predictors 时强制去掉影响最小的
    """
    y = table["y"]
    z = standardise(table, names)
    log = pd.DataFrame(index=names)
    log["selected"] = False

    corr = z.corr()
    corr.round(3).to_csv(cfg["paths"]["tables"] / ("T4_correlation_" + key + ".csv"))

    # 1. 单变量 AIC
    for name in names:
        log.loc[name, "aic_univariate"] = fit_logit(y, z[[name]]).aic
    order = log.sort_values("aic_univariate").index

    # 2. 相关性
    kept = []
    for name in order:
        conflict = ""
        for other in kept:
            if abs(corr.loc[name, other]) > cfg["r_max"]:
                conflict = other
                break
        if conflict == "":
            kept.append(name)
            log.loc[name, "step2_correlation"] = "kept"
        else:
            log.loc[name, "step2_correlation"] = "dropped (|r| > " + str(cfg["r_max"]) + " with " + conflict + ")"

    # 3. VIF
    while True:
        vif = vif_table(z[kept])
        worst = max(vif, key=vif.get)
        if vif[worst] <= cfg["vif_max"]:
            break
        kept.remove(worst)
        log.loc[worst, "step3_vif"] = "dropped (VIF " + str(round(vif[worst], 1)) + ")"
    for name in kept:
        log.loc[name, "vif"] = vif[name]

    # 4. 向后 AIC
    while len(kept) > 1:
        current_aic = fit_logit(y, z[kept]).aic
        best_name = ""
        best_aic = np.inf
        for name in kept:
            reduced = []
            for other in kept:
                if other != name:
                    reduced.append(other)
            aic = fit_logit(y, z[reduced]).aic
            if aic < best_aic:
                best_aic = aic
                best_name = name
        if best_aic < current_aic or len(kept) > cfg["max_predictors"]:
            kept.remove(best_name)
            log.loc[best_name, "step4_aic"] = "dropped (AIC " + str(round(current_aic, 1)) + " -> " + str(round(best_aic, 1)) + ")"
        else:
            break

    for name in kept:
        log.loc[name, "selected"] = True
    log.round(3).to_csv(cfg["paths"]["tables"] / ("T4_selection_" + key + ".csv"), index_label="variable")
    return kept


# ---------------------------------------------------------------------------
# 5. 模型和报告
# ---------------------------------------------------------------------------
def morans_i(table, residuals):
    """
    抽样点之间残差的空间自相关。用 8 个最近邻作为空间权重，行标准化。
    I 接近 0 = 残差没有空间结构；显著为正 = 相邻点的残差相似，标准误被低估。
    """
    coords = np.column_stack([table["col"].to_numpy(), table["row"].to_numpy()])
    weights = libpysal.weights.KNN.from_array(coords, k=8)
    weights.transform = "r"
    moran = esda.Moran(np.asarray(residuals), weights, permutations=499)
    return moran.I, moran.p_sim


def report(result, table, z, region, model):
    """系数表 + 模型摘要。系数是每 1 个标准差的变化；odds ratio = exp(系数)。"""
    ci = result.conf_int(alpha=0.05)
    coef_rows = []
    for name in result.params.index:
        group = ""
        unit = ""
        if name in VARIABLES:
            group = VARIABLES[name][0]
            unit = VARIABLES[name][1]
        coef_rows.append({
            "region": region, "model": model, "variable": name, "group": group, "unit": unit,
            "mean": table[name].mean() if name in table else np.nan,
            "sd": table[name].std() if name in table else np.nan,
            "coef_per_sd": result.params[name],
            "ci_low": ci.loc[name, 0], "ci_high": ci.loc[name, 1],
            "odds_ratio": np.exp(result.params[name]),
            "or_low": np.exp(ci.loc[name, 0]), "or_high": np.exp(ci.loc[name, 1]),
            "p_value": result.pvalues[name],
        })

    probability = result.predict(sm.add_constant(z))
    moran_i, moran_p = morans_i(table, result.resid_pearson)
    summary = {
        "region": region, "model": model,
        "n": len(table), "events": int(table["y"].sum()),
        "event_rate_pct": 100.0 * table["y"].mean(),
        "aic": result.aic, "mcfadden_r2": result.prsquared,
        "auc": roc_auc_score(table["y"], probability),
        "moran_I_resid": moran_i, "moran_p": moran_p,
    }
    return coef_rows, summary


def odds_ratio_plot(coefficients, out_path):
    """每个区一列，横轴 odds ratio（对数刻度），横线是 95% CI，竖线 OR = 1 表示没有影响。"""
    regions = list(coefficients["region"].unique())
    fig, axes = plt.subplots(1, len(regions), figsize=(6 * len(regions), 4), squeeze=False)
    for i in range(len(regions)):
        ax = axes[0, i]
        part = coefficients[(coefficients["region"] == regions[i]) & (coefficients["variable"] != "const")]
        part = part.sort_values("odds_ratio")
        positions = np.arange(len(part))
        ax.errorbar(part["odds_ratio"], positions,
                    xerr=[part["odds_ratio"] - part["or_low"], part["or_high"] - part["odds_ratio"]],
                    fmt="o", color="#2f6b34", capsize=3)
        ax.axvline(1.0, color="grey", linestyle="--", linewidth=1)
        ax.set_xscale("log")
        ax.set_yticks(positions)
        ax.set_yticklabels(part["variable"])
        ax.set_xlabel("odds ratio per 1 SD (95% CI)")
        ax.set_title(regions[i] + " – forest " + part["model"].iloc[0])
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def run(cfg):
    processed = cfg["paths"]["processed"]
    tables = cfg["paths"]["tables"]
    nodata = cfg["nodata"]

    lulc_a, profile = utils.read_tif(processed / "lulc_A.tif")
    lulc_b, _ = utils.read_tif(processed / "lulc_B.tif")
    forest_a, _ = utils.read_tif(processed / "forest_A.tif")
    forest_b, _ = utils.read_tif(processed / "forest_B.tif")
    units = gpd.read_file(processed / "units.geojson")
    canton = units[units["scale"] == "canton"]

    unit_masks = []
    for i in range(len(units)):
        unit_masks.append(t1_grid.polygons_to_mask(units.iloc[[i]], profile) == 1)

    # 1. 森林变化
    change = change_raster(forest_a, forest_b, nodata)
    utils.save_tif(change, processed / "change_AB.tif", profile, nodata=nodata)
    change_table = forest_change_table(change, units, unit_masks, cfg)
    print(change_table.to_string(index=False))

    # 2. 预测变量
    pred = build_predictors(cfg, profile, canton, lulc_a, lulc_b, forest_a)

    all_coefficients = []
    summaries = []
    for region in cfg["model_regions"]:
        i = list(units["unit"]).index(region)
        key = region.split(" ")[0]
        model = change_table.loc[change_table["unit"] == region, "model"].iloc[0]

        # 3. 抽样
        table = sample_region(unit_masks[i], change, lulc_a, model, pred, cfg)
        names = []
        for name in cfg["candidates"]:
            if name in table.columns:
                names.append(name)
        print("  ---", region, "| 模型：forest", model, "| 样本", len(table), "| 事件", int(table["y"].sum()))

        # 4. 筛选
        selected = select_predictors(table, names, cfg, key)
        print("  选中的变量：", selected)

        # 5. 最终模型
        z = standardise(table, selected)
        result = fit_logit(table["y"], z)
        coef_rows, summary = report(result, table, z, key, model)
        all_coefficients = all_coefficients + coef_rows
        summaries.append(summary)

    coefficients = pd.DataFrame(all_coefficients)
    coefficients.round(4).to_csv(tables / "T4_coefficients.csv", index=False)
    summary_table = pd.DataFrame(summaries).round(4)
    summary_table.to_csv(tables / "T4_model_summary.csv", index=False)
    odds_ratio_plot(coefficients, cfg["paths"]["figures"] / "T4_odds_ratios.png")

    show = ["region", "variable", "coef_per_sd", "ci_low", "ci_high", "odds_ratio", "p_value"]
    print(coefficients[show].round(3).to_string(index=False))
    print(summary_table.to_string(index=False))
    print("  Task 4 完成。")
