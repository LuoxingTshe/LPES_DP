"""Raster I/O helpers shared by all stages (reference-raster pattern: the grid is read from the GeoTIFF profile)."""
import rasterio
from rasterio.transform import from_origin


def grid_transform(meta):
    """Affine transform from grid parameters: x = xmin + col * res, y = ymax - row * res."""
    return from_origin(meta["xmin"], meta["ymax"], meta["res"], meta["res"])


def new_profile(meta):
    """Blank GeoTIFF profile for the first rasters written in Task 1 (dtype and nodata are set when saving)."""
    profile = {
        "driver": "GTiff",
        "height": meta["nrow"],
        "width": meta["ncol"],
        "count": 1,
        "crs": meta["crs"],
        "transform": grid_transform(meta),
        "compress": "lzw",
    }
    return profile


def read_tif(path):
    """Read band 1 of a GeoTIFF; return (array, profile)."""
    with rasterio.open(path) as src:
        array = src.read(1)
        profile = src.profile
    return array, profile


def save_tif(array, path, profile, nodata=None):
    """Write a 2-D array as a single-band GeoTIFF on the grid described by profile."""
    expected_shape = (profile["height"], profile["width"])
    if array.shape != expected_shape:
        raise ValueError("array shape " + str(array.shape) + " does not match grid " + str(expected_shape))

    out_profile = profile.copy()       # do not modify the caller's profile
    out_profile.update(dtype=array.dtype, nodata=nodata, count=1, driver="GTiff", compress="lzw")

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array, 1)
