"""
utils.py – 各个 Task 共用的栅格读写函数。

网格信息的唯一来源是栅格文件本身：
    每个 GeoTIFF 的文件头里都写着 transform、crs、行列数、nodata、数据类型，
    rasterio 把这些打包成一个字典，叫 profile。
    Task 1 从点数据第一次建出网格并写出 lulc_A.tif 之后，后续所有阶段都从
    这张 tif 读 profile，不再单独保存网格参数。这就是 GDAL/rasterio 的"参考栅格"做法。
"""
import rasterio
from rasterio.transform import from_origin


def grid_transform(meta):
    """
    由网格参数生成仿射变换 (Affine transform)：行列号 <-> 地图坐标。
        x = xmin + col * res
        y = ymax - row * res
    只在 Task 1 建网格时用一次；之后 transform 都从 tif 的 profile 里读。
    """
    transform = from_origin(meta["xmin"], meta["ymax"], meta["res"], meta["res"])
    return transform


def new_profile(meta):
    """
    由网格参数生成一个"空白" profile，供 Task 1 写出第一批 tif。
    dtype 和 nodata 不在这里定，由 save_tif 根据要写的数组决定。
    """
    profile = {
        "driver": "GTiff",
        "height": meta["nrow"],
        "width": meta["ncol"],
        "count": 1,                    # 波段数
        "crs": meta["crs"],
        "transform": grid_transform(meta),
        "compress": "lzw",             # 无损压缩，文件小很多
    }
    return profile


def read_tif(path):
    """
    读取单波段 GeoTIFF，返回 (数组, profile)。
    profile 是一个字典，里面有 transform、crs、width、height、dtype、nodata 等。
    """
    with rasterio.open(path) as src:
        array = src.read(1)            # 读第 1 个波段
        profile = src.profile
    return array, profile


def save_tif(array, path, profile, nodata=None):
    """
    以 profile 为模板，把二维数组写成单波段 GeoTIFF。

    profile 只提供"网格说明"（transform、crs、行列数）；
    dtype 和 nodata 每次按要写的数组重新设置，所以先 copy 一份再 update，
    不会改动调用者手里的 profile。
    nodata : 表示"无数据"的数值；None 表示这张图没有无数据值
    """
    # 先检查形状，防止把错尺寸的数组写进去（这种错误在 QGIS 里很难发现）
    expected_shape = (profile["height"], profile["width"])
    if array.shape != expected_shape:
        raise ValueError("数组形状 " + str(array.shape) + " 和网格 " + str(expected_shape) + " 不一致")

    out_profile = profile.copy()
    out_profile.update(dtype=array.dtype, nodata=nodata, count=1, driver="GTiff", compress="lzw")

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array, 1)            # 写入第 1 个波段
