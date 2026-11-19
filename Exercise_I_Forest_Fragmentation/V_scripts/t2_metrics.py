"""Task 2 – FG1 fragmentation geometry, three-class landscape, meff (CBC and CUT), ED, PD, LPI at several scales."""
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

# landscape classes
OTHER = 0      # non-barrier land use (agriculture, grassland, water, rock, ...)
FOREST = 1
BARRIER = 2

ROAD_LAYER = "TLMRegio_Road"
RAIL_LAYER = "TLMRegio_Railway"
REGION_LAYER = "N2020_Revision_BiogeoRegion"
REGION_NAME_TABLE = "RegionName_Catalogue"

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


def sql_in(codes):
    """[1, 2, 3] -> "1, 2, 3" for an SQL IN (...) clause."""
    text = ""
    for code in codes:
        if text != "":
            text = text + ", "
        text = text + str(code)
    return text


def load_units(cfg):
    """
    One row per reporting unit (columns unit, scale, geometry): canton, two municipalities and the
    biogeographical regions clipped to the canton. where= filters features while reading.
    """
    gdb = cfg["paths"]["raw"] / cfg["files"]["boundaries"]
    biogeo = cfg["paths"]["raw"] / cfg["files"]["biogeo"]

    canton = gpd.read_file(gdb, layer=t1_grid.CANTON_LAYER,
                           where="KANTONSNUMMER = " + str(cfg["units"]["canton"]))
    urban = gpd.read_file(gdb, layer=t1_grid.MUNICIPALITY_LAYER,
                          where="BFS_NUMMER = " + str(cfg["units"]["urban"]))
    rural = gpd.read_file(gdb, layer=t1_grid.MUNICIPALITY_LAYER,
                          where="BFS_NUMMER = " + str(cfg["units"]["rural"]))

    # regions consist of several sub-regions -> dissolve by RegionName
    regions = gpd.read_file(biogeo, layer=REGION_LAYER)
    regions = regions.dissolve(by="RegionName", as_index=False)
    regions = regions[regions["RegionName"].isin(cfg["regions"])]

    names = gpd.read_file(biogeo, layer=REGION_NAME_TABLE)        # attribute table without geometry
    regions = regions.merge(names, left_on="RegionName", right_on="CODERegionName")

    # swissBOUNDARIES3D has a compound CRS (LV95 + LN02 heights); after dropping Z it is plain EPSG:2056
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


def barrier_lines(cfg, profile):
    """
    Main roads and railways rasterised to 0/1. Needed because the 100 m sample points capture roads
    only intermittently, so roads in the Arealstatistik raster do not cut patches.
    all_touched=False gives 8-connected lines (enough to cut 4-neighbour patches);
    all_touched=True gives thicker, 4-connected lines, required for the 8-neighbour rule.
    """
    gdb = cfg["paths"]["raw"] / cfg["files"]["tlmregio"]

    # CONSTRUCT = 2 are tunnels: the ground above stays connected
    road_where = "OBJVAL IN (" + sql_in(cfg["road_codes"]) + ") AND CONSTRUCT <> 2"
    rail_where = "OBJVAL IN (" + sql_in(cfg["rail_codes"]) + ") AND CONSTRUCT <> 2"
    roads = gpd.read_file(gdb, layer=ROAD_LAYER, where=road_where)
    rails = gpd.read_file(gdb, layer=RAIL_LAYER, where=rail_where)
    print("  road segments:", len(roads), "  rail segments:", len(rails))

    geometries = list(roads.geometry.force_2d()) + list(rails.geometry.force_2d())
    all_touched = cfg["neighbourhood"] == 8

    lines = rasterize(
        geometries,
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        default_value=1,
        all_touched=all_touched,
        dtype="uint8",
    )
    return lines


def make_landscape(lulc, lines, cfg):
    """72 classes + barrier lines -> other / forest / barrier. Barriers are written last, so a road cuts the forest."""
    nodata = cfg["nodata"]
    has_data = lulc > 0

    land = np.full(lulc.shape, nodata, dtype=np.uint8)
    land[has_data] = OTHER
    land[np.isin(lulc, cfg["forest_codes"])] = FOREST
    land[np.isin(lulc, cfg["barrier_codes"])] = BARRIER
    land[(lines == 1) & has_data] = BARRIER
    return land


def label_patches(binary, neighbourhood):
    """
    Label patches on the whole Swiss grid; return labels, patch ids and the complete size of each patch.
    Labelling on the whole grid (not inside a unit) keeps the full area of cross-boundary patches (CBC).
    """
    if neighbourhood == 4:
        structure = ndimage.generate_binary_structure(2, 1)
    else:
        structure = ndimage.generate_binary_structure(2, 2)
    labels, n = ndimage.label(binary, structure=structure)
    patch_ids = np.arange(1, n + 1)
    full_size = ndimage.sum_labels(binary, labels, patch_ids)
    return labels, patch_ids, full_size


def meff_cbc(labels, patch_ids, full_size, unit_mask):
    """
    Effective mesh size, CBC method (Moser et al. 2007): meff = (1 / A_total) * sum(A_i * A_i_cmpl),
    with A_i the patch area inside the unit and A_i_cmpl its complete area. Result in km² (1 cell = 1 ha).
    """
    inside_size = ndimage.sum_labels(unit_mask, labels, patch_ids)
    total_area = unit_mask.sum()
    meff_ha = np.sum(inside_size * full_size) / total_area
    return meff_ha / 100.0


def clip_to_unit(tif_path, geometry, nodata):
    """Clip a raster to a unit (crop to its bounding box, cells outside = nodata); return (array, transform)."""
    with rasterio.open(tif_path) as src:
        clipped, clipped_transform = rasterio.mask.mask(src, [geometry], crop=True, nodata=nodata)
    return clipped[0], clipped_transform


def pylandstats_metrics(array, transform, cfg):
    """Class-level metrics for forest; nodata cells are not part of the landscape."""
    landscape = pls.Landscape(
        array,
        res=(transform.a, -transform.e),
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
        # pylandstats uses the cutting-out method (CUT), reported for comparison with CBC
        "meff_forest_cut_km2": landscape.effective_mesh_size(class_val=FOREST) / 100.0,
    }
    return result


def change_table(table, out_path):
    """Long table -> one row per unit with A, B and the change in percent for each metric."""
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
    """Three-class landscape of A and B side by side, to check that barrier lines are continuous."""
    colours = ListedColormap(["#e8e4d8", "#4c8c4a", "#c0392b"])   # other / forest / barrier
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    for i in range(2):
        array, _ = clip_to_unit(paths[i], canton_geometry, nodata)
        shown = np.ma.masked_equal(array, nodata)
        axes[i].imshow(shown, cmap=colours, vmin=-0.5, vmax=2.5, interpolation="nearest")
        axes[i].set_title(labels[i] + ": beige other, green forest, red barrier (FG1)")
        axes[i].set_axis_off()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run(cfg):
    processed = cfg["paths"]["processed"]
    nodata = cfg["nodata"]
    nb = cfg["neighbourhood"]
    _, profile = utils.read_tif(processed / "lulc_A.tif")      # reference raster

    units = load_units(cfg)
    units.to_file(processed / "units.geojson", driver="GeoJSON")

    # the same road network is used for both time steps (see limitations)
    lines = barrier_lines(cfg, profile)
    utils.save_tif(lines, processed / "barrier_lines.tif", profile, nodata=None)

    unit_masks = []
    for i in range(len(units)):
        unit_masks.append(t1_grid.polygons_to_mask(units.iloc[[i]], profile) == 1)

    records = []
    landscape_paths = []
    for time in ["A", "B"]:
        survey = "AS" + cfg["time_" + time]
        print("  ---", time, survey)

        lulc, _ = utils.read_tif(processed / ("lulc_" + time + ".tif"))
        land = make_landscape(lulc, lines, cfg)
        path = processed / ("landscape_" + time + ".tif")
        utils.save_tif(land, path, profile, nodata=nodata)
        landscape_paths.append(path)

        # forest patches: cut by non-forest and barriers (forest fragmentation)
        # open patches: forest + other, cut only by barriers (landscape fragmentation, Jaeger et al. 2008)
        forest_labels, forest_ids, forest_size = label_patches(land == FOREST, nb)
        open_labels, open_ids, open_size = label_patches((land == FOREST) | (land == OTHER), nb)
        print("  forest patches (CH):", len(forest_ids), "  open patches:", len(open_ids))

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
            print("   ", record["unit"], "done")

    table = pd.DataFrame(records).round(3)
    table.to_csv(cfg["paths"]["tables"] / "T2_metrics.csv", index=False)
    change_table(table, cfg["paths"]["tables"] / "T2_change.csv")
    quicklook(landscape_paths, ["AS" + cfg["time_A"], "AS" + cfg["time_B"]],
              units.geometry.iloc[0], nodata, cfg["paths"]["figures"] / "T2_landscape_quicklook.png")

    show = ["unit", "time", "forest_ha", "meff_forest_cbc_km2", "meff_forest_cut_km2",
            "meff_open_cbc_km2", "ED_m_per_ha", "PD_per_100ha", "LPI_pct"]
    print(table[show].to_string(index=False))
    print("  Task 2 done.")
