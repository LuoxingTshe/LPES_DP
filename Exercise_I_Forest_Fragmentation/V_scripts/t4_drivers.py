"""Task 4 – drivers of forest change in two biogeographical regions (logistic regression)."""
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

DEM_CRS = "EPSG:21781"     # DHM25 ASCII grid is in LV03; the file carries no CRS

# classes of change_AB.tif
STABLE_NON = 0
STABLE_FOREST = 1
GAIN = 2
LOSS = 3


def change_raster(forest_a, forest_b, nodata):
    """Two forest rasters -> stable non-forest / stable forest / gain / loss."""
    change = np.full(forest_a.shape, nodata, dtype=np.uint8)
    valid = (forest_a != nodata) & (forest_b != nodata)
    change[valid & (forest_a == 0) & (forest_b == 0)] = STABLE_NON
    change[valid & (forest_a == 1) & (forest_b == 1)] = STABLE_FOREST
    change[valid & (forest_a == 0) & (forest_b == 1)] = GAIN
    change[valid & (forest_a == 1) & (forest_b == 0)] = LOSS
    return change


def forest_change_table(change, units, unit_masks, cfg):
    """Gross gain, gross loss and net change (ha) per unit; the sign of the net change selects the model."""
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


def dem_predictors(cfg, profile, canton):
    """
    Elevation, slope and TRI computed on the 25 m DEM, then averaged to the 100 m grid
    (aggregating first would smooth away steep terrain). Only the canton's bounding box (+ 2 km) is read.
    Results are cached as dem_*.tif because parsing the ASCII grid takes about a minute.
    """
    processed = cfg["paths"]["processed"]
    names = ["elev", "slope", "tri"]
    cached = True
    for name in names:
        if not (processed / ("dem_" + name + ".tif")).exists():
            cached = False
    if cached:
        print("  DEM predictors: read from cache")
        result = {}
        for name in names:
            result[name], _ = utils.read_tif(processed / ("dem_" + name + ".tif"))
        return result

    print("  DEM predictors: computing from DHM25 (about 1 minute)")
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

    dz_dy, dz_dx = np.gradient(z, cell)
    slope = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))

    # Terrain Ruggedness Index (Riley et al. 1999): sqrt of the summed squared differences to the 8 neighbours
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
            resampling=Resampling.average,
        )
        utils.save_tif(out, processed / ("dem_" + name + ".tif"), profile, nodata=np.nan)
        result[name] = out
    return result


def window_mean(array, size):
    """Mean in a size x size window; cells beyond the grid count as 0."""
    return ndimage.uniform_filter(array.astype("float32"), size=size, mode="constant", cval=0.0)


def window_nanmean(array, size):
    """
    Window mean ignoring NaN: sum with NaN set to 0, divided by the number of valid cells.
    uniform_filter works row by row and column by column, so a single NaN would spread along both.
    """
    valid = np.logical_not(np.isnan(array))
    filled = np.where(valid, array, 0.0)
    total = window_mean(filled, size)
    count = window_mean(valid, size)
    result = np.full(array.shape, np.nan, dtype="float32")
    np.divide(total, count, out=result, where=count > 0)
    return result


def fill_nearest(array):
    """Replace NaN with the value of the nearest valid cell (indices from distance_transform_edt)."""
    missing = np.isnan(array)
    _, (near_rows, near_cols) = ndimage.distance_transform_edt(missing, return_indices=True)
    return array[near_rows, near_cols]


def distance_to(target, cell=100.0):
    """Distance (m) from every cell to the nearest target cell."""
    return ndimage.distance_transform_edt(np.logical_not(target)) * cell


def burn(geometries, profile):
    """Geometries -> 0/1 raster."""
    return rasterize(geometries, out_shape=(profile["height"], profile["width"]),
                     transform=profile["transform"], fill=0, default_value=1, dtype="uint8")


def build_predictors(cfg, profile, canton, lulc_a, lulc_b, forest_a):
    """Candidate predictors on the 100 m grid, returned as {name: array}."""
    raw = cfg["paths"]["raw"]
    processed = cfg["paths"]["processed"]
    size = cfg["window"]
    bbox = tuple(canton.total_bounds)
    pred = {}

    # environmental
    pred.update(dem_predictors(cfg, profile, canton))
    # TPI: positive = ridges and convex terrain, negative = valleys and hollows
    pred["tpi"] = pred["elev"] - window_nanmean(pred["elev"], size)

    tlm = raw / cfg["files"]["tlmregio"]
    rivers = gpd.read_file(tlm, layer="TLMRegio_FlowingWater", bbox=bbox)
    lakes = gpd.read_file(tlm, layer="TLMRegio_Lake", bbox=bbox)
    water = burn(list(rivers.geometry.force_2d()) + list(lakes.geometry.force_2d()), profile)
    pred["dist_water"] = distance_to(water == 1)

    # socio-economic
    # STATPOP 2017 uses the same hectare coordinates as the Arealstatistik, so the same (E, N) -> same cell
    pop = pyogrio.read_dataframe(raw / cfg["files"]["statpop"], layer=cfg["statpop_layer"],
                                 columns=["E_KOORD", "N_KOORD", "BBTOT"], read_geometry=False)
    rows, cols = rowcol(profile["transform"], pop["E_KOORD"].to_numpy(), pop["N_KOORD"].to_numpy())
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    inside = (rows >= 0) & (rows < profile["height"]) & (cols >= 0) & (cols < profile["width"])
    pop_grid = np.zeros((profile["height"], profile["width"]), dtype="float32")
    pop_grid[rows[inside], cols[inside]] = pop["BBTOT"].to_numpy()[inside]
    pred["pop_500m"] = np.log1p(window_mean(pop_grid, size) * size * size)   # persons in the window

    # traffic zones cover only populated areas (≈68 % of Bern); cells outside take the nearest zone's value
    access = gpd.read_file(raw / cfg["files"]["access"], bbox=bbox)
    shapes = []
    for i in range(len(access)):
        shapes.append((access.geometry.iloc[i], float(access[cfg["access_field"]].iloc[i])))
    access_grid = rasterize(shapes, out_shape=(profile["height"], profile["width"]),
                            transform=profile["transform"], fill=np.nan, dtype="float32")
    pred["access_road"] = np.log1p(fill_nearest(access_grid))

    zones = gpd.read_file(raw / cfg["files"]["bauzonen"], bbox=bbox)
    pred["dist_bauzone"] = distance_to(burn(list(zones.geometry), profile) == 1)

    lines, _ = utils.read_tif(processed / "barrier_lines.tif")     # main roads and railways (Task 2)
    pred["dist_mainroad"] = distance_to(lines == 1)

    settl_a = window_mean(np.isin(lulc_a, cfg["settlement_codes"]), size)
    settl_b = window_mean(np.isin(lulc_b, cfg["settlement_codes"]), size)
    pred["settl_chg"] = 100.0 * (settl_b - settl_a)

    # land use at time A
    is_forest = forest_a == 1
    pred["forest_share"] = 100.0 * window_mean(is_forest, size)

    # forest cells: distance to non-forest; non-forest cells: distance to forest (nodata is not a boundary)
    dist_in_forest = ndimage.distance_transform_edt(forest_a != 0) * 100.0
    dist_outside = ndimage.distance_transform_edt(forest_a != 1) * 100.0
    pred["dist_edge"] = np.where(is_forest, dist_in_forest, dist_outside)

    labels, _, full_size = t2_metrics.label_patches(is_forest, cfg["neighbourhood"])
    size_by_label = np.concatenate([[np.nan], full_size])     # label 0 = not forest -> NaN
    pred["patch_size"] = np.log10(size_by_label[labels])

    return pred


def sample_region(region_mask, change, lulc_a, model, pred, cfg):
    """
    Systematic sample every sample_step cells. Loss model: forest at A. Gain model: non-forest at A
    in gain_eligible_codes.
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
            continue                       # non-forest cells have no forest patch
        data[name] = pred[name][rows, cols]
    table = pd.DataFrame(data).dropna()
    return table


def standardise(table, names):
    """z-scores, so that coefficients are per 1 SD and comparable between predictors."""
    z = table[names].copy()
    for name in names:
        z[name] = (z[name] - z[name].mean()) / z[name].std()
    return z


def fit_logit(y, X):
    return sm.Logit(y, sm.add_constant(X)).fit(disp=0, maxiter=200)


def vif_table(X):
    """Variance inflation factor of each predictor (design matrix with intercept)."""
    design = sm.add_constant(X)
    values = {}
    for i in range(1, design.shape[1]):
        values[design.columns[i]] = variance_inflation_factor(design.values, i)
    return values


def select_predictors(table, names, cfg, key):
    """
    1. univariate AIC; 2. in order of AIC, drop a predictor with |r| > r_max to one already kept;
    3. drop the largest VIF until all <= vif_max; 4. backward AIC, forced down to max_predictors.
    Each step is logged in T4_selection_<region>.csv.
    """
    y = table["y"]
    z = standardise(table, names)
    log = pd.DataFrame(index=names)
    log["selected"] = False

    corr = z.corr()
    corr.round(3).to_csv(cfg["paths"]["tables"] / ("T4_correlation_" + key + ".csv"))

    for name in names:
        log.loc[name, "aic_univariate"] = fit_logit(y, z[[name]]).aic
    order = log.sort_values("aic_univariate").index

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

    while True:
        vif = vif_table(z[kept])
        worst = max(vif, key=vif.get)
        if vif[worst] <= cfg["vif_max"]:
            break
        kept.remove(worst)
        log.loc[worst, "step3_vif"] = "dropped (VIF " + str(round(vif[worst], 1)) + ")"
    for name in kept:
        log.loc[name, "vif"] = vif[name]

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


def morans_i(table, residuals, seed):
    """Moran's I of the residuals with 8 nearest neighbours (row-standardised) and a permutation test."""
    coords = np.column_stack([table["col"].to_numpy(), table["row"].to_numpy()])
    weights = libpysal.weights.KNN.from_array(coords, k=8)
    weights.transform = "r"
    np.random.seed(seed)        # fixed seed: the permutation test gives the same p-value on every run
    moran = esda.Moran(np.asarray(residuals), weights, permutations=499)
    return moran.I, moran.p_sim


def report(result, table, z, region, model, seed):
    """Coefficient table (per 1 SD, odds ratio = exp(coef)) and model summary."""
    ci = result.conf_int(alpha=0.05)
    coef_rows = []
    for name in result.params.index:
        mean = np.nan
        sd = np.nan
        if name in table:
            mean = table[name].mean()
            sd = table[name].std()
        coef_rows.append({
            "region": region, "model": model, "variable": name,
            "mean": mean, "sd": sd,
            "coef_per_sd": result.params[name],
            "ci_low": ci.loc[name, 0], "ci_high": ci.loc[name, 1],
            "odds_ratio": np.exp(result.params[name]),
            "or_low": np.exp(ci.loc[name, 0]), "or_high": np.exp(ci.loc[name, 1]),
            "p_value": result.pvalues[name],
        })

    probability = result.predict(sm.add_constant(z))
    moran_i, moran_p = morans_i(table, result.resid_pearson, seed)
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
    """Odds ratios with 95% CI per region (log scale); OR = 1 means no effect."""
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

    change = change_raster(forest_a, forest_b, nodata)
    utils.save_tif(change, processed / "change_AB.tif", profile, nodata=nodata)
    change_table = forest_change_table(change, units, unit_masks, cfg)
    print(change_table.to_string(index=False))

    pred = build_predictors(cfg, profile, canton, lulc_a, lulc_b, forest_a)

    all_coefficients = []
    summaries = []
    for region in cfg["model_regions"]:
        i = list(units["unit"]).index(region)
        key = region.split(" ")[0]
        model = change_table.loc[change_table["unit"] == region, "model"].iloc[0]

        table = sample_region(unit_masks[i], change, lulc_a, model, pred, cfg)
        names = []
        for name in cfg["candidates"]:
            if name in table.columns:
                names.append(name)
        print("  ---", region, "| model: forest", model, "| n", len(table), "| events", int(table["y"].sum()))

        selected = select_predictors(table, names, cfg, key)
        print("  selected:", selected)

        z = standardise(table, selected)
        result = fit_logit(table["y"], z)
        coef_rows, summary = report(result, table, z, key, model, cfg["seed"])
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
    print("  Task 4 done.")
