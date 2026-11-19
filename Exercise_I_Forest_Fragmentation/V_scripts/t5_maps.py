"""Task 5 – forest change map of the canton and the numbers for the infographic chain."""
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

from V_scripts import t2_metrics

# change_AB.tif: stable non-forest / stable forest / gain / loss
COLOURS = ["#f3f1ea", "#bcd3ab", "#2c7bb6", "#d7191c"]
REGION_COLOUR = "#7b3294"

SOURCES = ("Data: FSO Arealstatistik (AS85, AS18, 100 m); FOEN biogeographical regions; "
           "swisstopo swissBOUNDARIES3D. CRS: CH1903+/LV95 (EPSG:2056).")


def raster_extent(array, transform):
    """imshow extent (left, right, bottom, top) in map coordinates, so vector layers overlay directly."""
    nrow, ncol = array.shape
    left = transform.c
    top = transform.f
    right = left + ncol * transform.a
    bottom = top + nrow * transform.e          # e is negative
    return (left, right, bottom, top)


def north_arrow(ax):
    """North arrow in the upper right corner (LV95 grid north ≈ true north in Switzerland)."""
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
    # solid coloured lines: dense boundary vertices make dashed lines look solid
    regions.boundary.plot(ax=ax, color=REGION_COLOUR, linewidth=1.3)
    canton.boundary.plot(ax=ax, color="black", linewidth=1.6)

    for i in range(len(regions)):
        point = regions.geometry.iloc[i].representative_point()     # always inside the polygon
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
    """meff change (CBC) -> edge share and edge/interior ratio -> strongest predictor, one row per unit."""
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
        key = unit_name.split(" ")[0]          # only the two modelled regions have a driver
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
    print("  Task 5 done. Map in VI_outputs/maps/, infographic numbers in T5_infographic_numbers.csv")
