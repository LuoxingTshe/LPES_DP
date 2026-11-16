import json
import numpy as np
import rasterio
from rasterio.transform import from_origin

def creatematrix(pointgrid):
    matrix = from_origin(pointgrid["xmin"], pointgrid["ymax"], pointgrid["res"], pointgrid["res"])
    return matrix

def save_meta(meta, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def load_meta(path):
    """读取 save_meta 写出的 JSON 文件，返回字典。"""
    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta


def save_tif(array, path, meta, nodata=None):
    """
    把一个二维 numpy 数组写成单波段 GeoTIFF。

    参数
    ----
    array  : 二维数组，形状必须是 (meta["nrow"], meta["ncol"])
    path   : 输出文件路径
    meta   : 网格元数据字典（提供 transform 和 crs）
    nodata : 表示"无数据"的数值；写进文件后 QGIS 会自动把它显示为透明
    """
    # 先检查形状，防止把错尺寸的数组写进去（这种错误在 QGIS 里很难发现）
    expected_shape = (meta["nrow"], meta["ncol"])
    if array.shape != expected_shape:
        raise ValueError("数组形状 " + str(array.shape) + " 和网格 " + str(expected_shape) + " 不一致")

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,                       # 波段数
        dtype=array.dtype,
        crs=meta["crs"],
        transform=creatematrix(meta),
        nodata=nodata,
        compress="lzw",                # 无损压缩，文件小很多
    ) as dst:
        dst.write(array, 1)            # 写入第 1 个波段


def read_tif(path):
    """读取单波段 GeoTIFF，只返回数组本身（网格信息统一用 grid_meta.json）。"""
    with rasterio.open(path) as src:
        array = src.read(1)
    return array