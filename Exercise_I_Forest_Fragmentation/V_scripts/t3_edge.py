"""Task 3 – forest edge (0–100 m) and interior (> 100 m), edge/interior ratio, transitions and 20-year outlook."""
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy import ndimage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from V_scripts import utils, t1_grid, t2_metrics

NONFOREST = 0
EDGE = 1
INTERIOR = 2
CLASS_NAMES = {NONFOREST: "non-forest", EDGE: "edge", INTERIOR: "interior"}


def classify_edge(land, cfg):
    """
    Three-class landscape (Task 2) -> 0 non-forest / 1 edge / 2 interior by binary erosion.
    A forest cell is interior if all neighbours under the chosen rule are forest. On the 100 m grid,
    orthogonal neighbours are 100 m away and diagonal ones 141 m, so the 4-neighbour rule matches
    "within 100 m of non-forest". Using the landscape raster means roads through forest also create edge.
    Nodata (outside Switzerland) is treated as forest so the national border is not an artificial edge.
    """
    nodata = cfg["nodata"]
    forest = land == t2_metrics.FOREST
    forest_or_nodata = forest | (land == nodata)

    if cfg["neighbourhood"] == 4:
        structure = ndimage.generate_binary_structure(2, 1)
    else:
        structure = ndimage.generate_binary_structure(2, 2)

    iterations = int(cfg["edge_width_m"] / t1_grid.RES)
    core = ndimage.binary_erosion(forest_or_nodata, structure=structure,
                                  iterations=iterations, border_value=1)

    interior = forest & core
    edge = forest & np.logical_not(core)

    classes = np.full(land.shape, nodata, dtype=np.uint8)
    classes[land != nodata] = NONFOREST
    classes[edge] = EDGE
    classes[interior] = INTERIOR
    return classes


def summarise(classes, unit_mask):
    """Edge and interior area (ha), edge share and edge/interior ratio within one unit."""
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
        "edge_share_pct": edge_share,
        "edge_interior_ratio": ratio,
    }
    return result


def summary_key(unit_name):
    """Row in T1_forest_summary.csv; regions use the canton's mean survey years."""
    if unit_name.endswith("(urban)"):
        return "urban"
    if unit_name.endswith("(rural)"):
        return "rural"
    return "canton"


def outlook(table, cfg):
    """
    Linear extrapolation: annual rate = (ratio_B - ratio_A) / (year_B - year_A), continued for outlook_years
    after B. Mean survey years from Task 1 are used (≈1981 and ≈2014 in Bern), not the nominal 1985 / 2018.
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


def transition_matrix(classes_a, classes_b, unit_mask, nodata, out_path):
    """
    Area (ha) by class at A (rows) and B (columns). For example, interior -> edge means the interior was
    opened up; non-forest -> edge is new forest, which usually appears as edge first.
    """
    inside = unit_mask & (classes_a != nodata) & (classes_b != nodata)
    a = pd.Series(classes_a[inside]).map(CLASS_NAMES)
    b = pd.Series(classes_b[inside]).map(CLASS_NAMES)
    matrix = pd.crosstab(a, b, rownames=["A"], colnames=["B"])
    matrix.to_csv(out_path)
    return matrix


def quicklook(paths, labels, units, nodata, out_path):
    """Canton, urban and rural municipality (rows) at A and B (columns): non-forest / edge / interior."""
    colours = ListedColormap(["#ece8dc", "#e08a2c", "#2f6b34"])
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


def run(cfg):
    processed = cfg["paths"]["processed"]
    nodata = cfg["nodata"]
    print("  neighbourhood rule:", cfg["neighbourhood"], "  edge width:", cfg["edge_width_m"], "m")

    units = gpd.read_file(processed / "units.geojson")
    _, profile = utils.read_tif(processed / "lulc_A.tif")     # reference raster

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
    print("  canton transitions A -> B (ha):")
    print(matrix)
    print("  Task 3 done.")
