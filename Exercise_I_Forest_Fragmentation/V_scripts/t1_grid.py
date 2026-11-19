"""Task 1 – rasterise the Arealstatistik points, reclassify forest and build the study-unit masks."""
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
    """Read coordinates, the two 72-class columns and the survey years from the Arealstatistik GeoPackage."""
    gpkg_path = cfg["paths"]["raw"] / cfg["files"]["lulc"]
    time_a = cfg["time_A"]
    time_b = cfg["time_B"]
    columns = [
        "E_COORD",
        "N_COORD",
        "AS" + time_a + "_72",
        "AS" + time_b + "_72",
        "FJ" + time_a,                 # actual survey year
        "FJ" + time_b,
    ]
    print("  reading points:", gpkg_path.name)
    df = pyogrio.read_dataframe(gpkg_path, layer=LULC_LAYER, columns=columns, read_geometry=False)
    print("  points:", len(df))
    return df


def build_grid_meta(df):
    """Grid parameters with the points at cell centres and a one-cell margin around the data."""
    x_min_points = int(df["E_COORD"].min())
    x_max_points = int(df["E_COORD"].max())
    y_min_points = int(df["N_COORD"].min())
    y_max_points = int(df["N_COORD"].max())

    pad = 1
    half = RES // 2

    xmin = x_min_points - half - pad * RES
    ymax = y_max_points + half + pad * RES
    ncol = (x_max_points - x_min_points) // RES + 1 + 2 * pad
    nrow = (y_max_points - y_min_points) // RES + 1 + 2 * pad

    meta = {
        "xmin": xmin,
        "ymax": ymax,
        "res": RES,
        "nrow": nrow,
        "ncol": ncol,
        "crs": CRS,
    }
    print("  grid:", nrow, "rows x", ncol, "cols")
    return meta


def point_indices(df, meta):
    """Row and column of each point: col = (E - res/2 - xmin) // res, row = (ymax - (N + res/2)) // res."""
    east = df["E_COORD"].to_numpy()
    north = df["N_COORD"].to_numpy()
    half = meta["res"] // 2

    cols = (east - half - meta["xmin"]) // meta["res"]
    rows = (meta["ymax"] - (north + half)) // meta["res"]

    rows = rows.astype(np.int64)
    cols = cols.astype(np.int64)

    if rows.min() < 0 or rows.max() >= meta["nrow"]:
        raise ValueError("row index outside the grid, check build_grid_meta")
    if cols.min() < 0 or cols.max() >= meta["ncol"]:
        raise ValueError("column index outside the grid, check build_grid_meta")

    # every cell may hold at most one point
    cell_id = rows * meta["ncol"] + cols
    n_unique = len(np.unique(cell_id))
    if n_unique != len(cell_id):
        raise ValueError(str(len(cell_id) - n_unique) + " points fall into an occupied cell")

    return rows, cols


def to_grid(values, rows, cols, meta, fill, dtype):
    """Write one value per point into the grid; cells without a point (outside Switzerland) keep fill."""
    grid = np.full((meta["nrow"], meta["ncol"]), fill, dtype=dtype)
    grid[rows, cols] = values
    return grid


def make_forest(lulc, forest_codes, nodata):
    """72-class raster -> 1 forest, 0 non-forest, nodata (lulc = 0 means no point)."""
    forest = np.full(lulc.shape, nodata, dtype=np.uint8)
    has_data = lulc > 0
    is_forest = np.isin(lulc, forest_codes)
    forest[has_data] = 0
    forest[is_forest] = 1
    return forest


def polygons_to_mask(gdf, profile):
    """Polygons -> 0/1 mask; a cell is 1 if its centre lies inside (Z values are dropped first)."""
    geometries = gdf.geometry.force_2d()
    shapes = []
    for geom in geometries:
        shapes.append((geom, 1))

    mask = rasterize(
        shapes,
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        dtype="uint8",
    )
    return mask


def unit_masks(cfg, profile):
    """
    Masks for the canton and the two municipalities.
    Current boundaries are used for both time steps, so A and B always cover the same area
    despite the municipal mergers since 1985.
    """
    gdb_path = cfg["paths"]["raw"] / cfg["files"]["boundaries"]
    masks = {}

    cantons = gpd.read_file(gdb_path, layer=CANTON_LAYER)
    canton = cantons[cantons["KANTONSNUMMER"] == cfg["units"]["canton"]]
    if len(canton) == 0:
        raise ValueError("no canton with KANTONSNUMMER = " + str(cfg["units"]["canton"]))
    print("  canton:", canton["NAME"].iloc[0])
    masks["canton"] = polygons_to_mask(canton, profile)

    municipalities = gpd.read_file(gdb_path, layer=MUNICIPALITY_LAYER)
    for key in ["urban", "rural"]:
        bfs = cfg["units"][key]
        selected = municipalities[municipalities["BFS_NUMMER"] == bfs]
        if len(selected) == 0:
            raise ValueError("no municipality with BFS_NUMMER = " + str(bfs))
        print("  municipality (" + key + "):", selected["NAME"].iloc[0], "BFS", bfs)
        masks[key] = polygons_to_mask(selected, profile)

    return masks


def transition_table(df, rows, cols, mask_canton, col_a, col_b, out_path):
    """72-class transition matrix A -> B inside the canton (rows = A, columns = B), used to check the forest definition."""
    in_canton = mask_canton[rows, cols] == 1
    df_canton = df[in_canton]

    table = pd.crosstab(df_canton[col_a], df_canton[col_b])
    table.to_csv(out_path)

    print("  transitions among classes 50-60 (rows = A, columns = B):")
    print(table.loc[50:60, 50:60].to_string())
    return table


def forest_summary(forest_a, forest_b, masks, df, rows, cols, cfg, out_path):
    """Forest area (1 cell = 1 ha), share and mean survey year per unit; delta_t is used in Task 3."""
    nodata = cfg["nodata"]
    year_a = df["FJ" + cfg["time_A"]].to_numpy()
    year_b = df["FJ" + cfg["time_B"]].to_numpy()

    records = []
    for unit_name in masks:
        mask = masks[unit_name] == 1
        valid = mask & (forest_a != nodata)
        area_ha = int(valid.sum())

        forest_a_ha = int((mask & (forest_a == 1)).sum())
        forest_b_ha = int((mask & (forest_b == 1)).sum())

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
    """Check plot of forest change in the canton (not the final Task 5 map)."""
    rows_in = np.where(mask_canton.any(axis=1))[0]
    cols_in = np.where(mask_canton.any(axis=0))[0]
    r0 = rows_in.min()
    r1 = rows_in.max() + 1
    c0 = cols_in.min()
    c1 = cols_in.max() + 1

    fa = forest_a[r0:r1, c0:c1]
    fb = forest_b[r0:r1, c0:c1]
    inside = mask_canton[r0:r1, c0:c1] == 1

    # NaN outside, 0 stable non-forest, 1 stable forest, 2 loss, 3 gain
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


def run(cfg):
    processed = cfg["paths"]["processed"]
    tables = cfg["paths"]["tables"]
    figures = cfg["paths"]["figures"]
    nodata = cfg["nodata"]
    col_a = "AS" + cfg["time_A"] + "_72"
    col_b = "AS" + cfg["time_B"] + "_72"

    df = loadptgrid(cfg)

    meta = build_grid_meta(df)
    rows, cols = point_indices(df, meta)
    # meta is only needed here; from now on the grid is read from the profile of lulc_A.tif
    profile = utils.new_profile(meta)

    # 72-class rasters, 0 = no data
    values_a = df[col_a].fillna(0).to_numpy().astype(np.uint8)
    values_b = df[col_b].fillna(0).to_numpy().astype(np.uint8)
    lulc_a = to_grid(values_a, rows, cols, meta, fill=0, dtype=np.uint8)
    lulc_b = to_grid(values_b, rows, cols, meta, fill=0, dtype=np.uint8)
    utils.save_tif(lulc_a, processed / "lulc_A.tif", profile, nodata=0)
    utils.save_tif(lulc_b, processed / "lulc_B.tif", profile, nodata=0)

    forest_a = make_forest(lulc_a, cfg["forest_codes"], nodata)
    forest_b = make_forest(lulc_b, cfg["forest_codes"], nodata)
    utils.save_tif(forest_a, processed / "forest_A.tif", profile, nodata=nodata)
    utils.save_tif(forest_b, processed / "forest_B.tif", profile, nodata=nodata)

    masks = unit_masks(cfg, profile)
    for unit_name in masks:
        utils.save_tif(masks[unit_name], processed / ("mask_" + unit_name + ".tif"), profile, nodata=None)

    transition_table(df, rows, cols, masks["canton"], col_a, col_b, tables / "T1_transition_72.csv")
    forest_summary(forest_a, forest_b, masks, df, rows, cols, cfg, tables / "T1_forest_summary.csv")
    quicklook(forest_a, forest_b, masks["canton"], figures / "T1_quicklook.png")

    print("  Task 1 done.")
