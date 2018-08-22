import cv2
import json
import logging
import subprocess

import numpy as np
import matplotlib.pyplot as plt

from osgeo import gdal
from skimage import filters
from shapely import geometry
from subprocess import DEVNULL
logger = logging.getLogger()


def grd2water(
        file_path: str,
        thr_value: int = 1,
        cnt_value: float = 40.0,
        wid: tuple = (15, 15),
        step: int = 4,
        water_range: tuple = (-12, -16)
)-> (bytearray, int):
    """
    Function:
        covert to img to bit by threshold and contour extract
    input:
        img: image array
        thr_value: the max value of binary image
        cnt_value: the min area of contours of image that should be keep
        wid: the window size of morphology
        step: the iteration of morphology
        water_range: the range of water threshold
    output:
        (img, int): ..
        img: binary image from input
        int: threshold
    """
    src_ds = gdal.Open(file_path)
    src_geo_trans = src_ds.GetGeoTransform()
    cols = src_ds.RasterXSize
    rows = src_ds.RasterYSize
    band = src_ds.GetRasterBand(1)
    water_data = band.ReadAsArray(0, 0, cols, rows)
    data_mask = np.logical_and(
        np.greater(water_data, 0.0), np.less_equal(water_data, 1.0)
    )
    data_orig = np.multiply(data_mask, water_data)
    data_orig[data_orig == 0.0] = 1.0
    data_orig = 10 * np.log10(data_orig)
    # get threshold
    blur = cv2.bilateralFilter(data_orig, 5, 25 * 2, 25 / 2)
    blur[np.isnan(blur)] = 0.0
    blur[np.isinf(blur)] = 0.0
    # figure after filter
    plt.figure(0)
    plt.imshow(blur, 'gray')
    plt.title('image after bilateral')
    # binary image
    data_mask2 = blur[blur != 0]
    threshold = filters.threshold_otsu(data_mask2)
    if (threshold > water_range[0]) | (threshold < water_range[1]):
        logger.warning("Use -13 as threshold here")
        threshold = -13

    ret1, bin_img = cv2.threshold(blur, threshold, thr_value, cv2.THRESH_BINARY_INV)
    bin_img = bin_img.astype(np.uint8)
    plt.figure(1)
    plt.imshow(bin_img, 'gray')
    plt.title('binary image')
    data_arr2 = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel=wid, iterations=step)

    # contours extract
    _, contour, hierarchy = cv2.findContours(image=data_arr2.copy(), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contour]
    contour_seq = sorted(contour_sizes, key=lambda x: x[0], reverse=True)
    size = [size[0] for size in contour_seq]
    size = np.array(size)
    last = np.where(size < cnt_value)[0][0]
    draw_cnts = [p[1] for p in contour_seq[0:last + 1]]
    res_img = np.zeros(data_arr2.shape, np.uint8)
    cv2.drawContours(res_img, np.array(draw_cnts), -1, 1, -1)
    res_img = res_img * 255
    plt.figure(2)
    plt.imshow(res_img, 'gray')
    plt.title('final image')
    # write into tiff
    water_tif = file_path.replace('_TF_TC.data/Gamma0_VV.img', '_water.tif')
    des_ds = gdal.GetDriverByName("GTiff").Create(
        water_tif, cols, rows, 1
    )
    des_ds.SetProjection(src_ds.GetProjection())
    des_ds.SetGeoTransform(src_geo_trans)
    des_ds.GetRasterBand(1).WriteArray(res_img, 0, 0)


def flood_mask(file_path_s1, file_path_s2, geojson_dict, json_path):

    src1 = gdal.Open(file_path_s1)
    src1_geo_trans = src1.GetGeoTransform()
    print(src1_geo_trans)
    cols1 = src1.RasterXSize
    rows1 = src1.RasterYSize
    temp1 = src1_geo_trans[0] + src1_geo_trans[1] * cols1
    temp2 = src1_geo_trans[3] + src1_geo_trans[5] * rows1
    src1_p = [[src1_geo_trans[0], src1_geo_trans[3]], [temp1, src1_geo_trans[3]], [temp1, temp2],
              [src1_geo_trans[0], temp2], [src1_geo_trans[0], src1_geo_trans[3]]]
    poly1 = geometry.Polygon([p[0], p[1]] for p in src1_p)
    print(poly1, poly1.is_valid, poly1.exterior.type)

    src2 = gdal.Open(file_path_s2)
    src2_geo_trans = src2.GetGeoTransform()
    print(src2_geo_trans)
    cols2 = src2.RasterXSize  # 列
    rows2 = src2.RasterYSize  # 行
    temp3 = src2_geo_trans[0] + src2_geo_trans[1] * cols2
    temp4 = src2_geo_trans[3] + src2_geo_trans[5] * rows2
    src2_p = [[src2_geo_trans[0], src2_geo_trans[3]], [temp3, src2_geo_trans[3]], [temp3, temp4],
              [src2_geo_trans[0], temp4], [src2_geo_trans[0], src2_geo_trans[3]]]
    poly2 = geometry.Polygon([p[0], p[1]] for p in src2_p)
    print(poly2, poly2.is_valid, poly2.exterior.type)

    # 求交叠区域
    poly_union = poly1.intersection(poly2)
    print(poly_union)
    creat_json(poly_union, geojson_dict, json_path)
    new_src1, arr1 = clip_tif(file_path_s1, json_path)
    new_src2, arr2 = clip_tif(file_path_s2, json_path)
    print(np.shape(arr1), np.shape(arr2))
    if np. shape(arr1) != np.shape(arr2):
        arr1 = arr_extract(arr1, arr2)
    mask = arr1-arr2  # src1矩阵比src2大
    mask[mask == -255] = 0
    ds_path = '/home/zy/data_pool/Flood/Argentina/water mask/final_mask.tif'
    rows, cols = np.shape(mask)
    try:
        des_ds = gdal.GetDriverByName("GTiff").Create(ds_path, cols, rows, 1)
        des_ds.SetProjection(new_src2.GetProjection())
        des_ds.SetGeoTransform(new_src2.GetGeoTransform())
        des_ds.GetRasterBand(1).WriteArray(mask, 0, 0)
        logger.info("Writing-TIFF Done! >.< %s", ds_path)
    except Exception as e:
        logger.debug("Writing failed! T-T %s %s", ds_path, e)
    print('write-tif done!')
    return mask


def creat_json(polygon, geojson_dict, json_path):
    wkt = polygon.wkt
    str_list = wkt.split('(')[-1].split(')')[0].split(',')
    poly_list = []

    for i in range(len(str_list)):
        tmp1 = str_list[i].split()
        tmp3 = [float(x) for x in tmp1]
        poly_list.append(tmp3)
    print(poly_list)

    geojson_dict["features"][0]['geometry']["coordinates"] = [poly_list]
    print(geojson_dict)
    with open(json_path, "w") as fp:
        print(json.dumps(geojson_dict), file=fp)


def clip_tif(src_path, json_path, band=1):
    clip_path = src_path.replace(".tif", "_clip.tif")
    subprocess.run(
        ['gdalwarp', '--config', 'GDALWARP_IGNORE_BAD_CUTLINE', 'YES', '-overwrite', '-of', 'GTiff', '-cutline',
         json_path, src_path,
         '-crop_to_cutline', clip_path,
         ])
    tmp = gdal.Open(clip_path)
    if tmp is None:
        logger.debug('Clip-tif failed! T-T')
        return False
    cols = tmp.RasterXSize
    rows = tmp.RasterYSize
    band = tmp.GetRasterBand(band)
    band_arr = band.ReadAsArray(0, 0, cols, rows)
    return tmp, band_arr


def arr_extract(arr1, arr2):
    x1, y1 = np.shape(arr1)
    x2, y2 = np.shape(arr2)
    dis_x = int(abs(x1-x2)/2)
    dis_y = int(abs(y1-y2)/2)
    new_arr1 = arr1[dis_x:(dis_x+x2), dis_y:(dis_y+y2)]
    return new_arr1


def water_generate(file_path, wid, step):
    src_ds = gdal.Open(file_path)
    src_geo_trans = src_ds.GetGeoTransform()
    cols = src_ds.RasterXSize
    rows = src_ds.RasterYSize
    band = src_ds.GetRasterBand(1)
    band_data = band.ReadAsArray(0, 0, cols, rows)
    print(rows, cols)
    # downsampling
    water_data = band_data[::5, ::5]
    val = 15
    water_data = cv2.bilateralFilter(water_data, val, val * 2, val / 2)
    print(np.shape(water_data))
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, wid)
    new_img = cv2.morphologyEx(water_data, cv2.MORPH_DILATE, kernel=ker, iterations=step)

    # resampling
    row, col = np.shape(new_img)
    data_arr = new_img.repeat(5)
    data_arr = data_arr.reshape(row, col*5).T
    data_arr2 = data_arr.repeat(5)
    data_arr2 = data_arr2.reshape(col*5, row*5).T

    plt.figure(0)
    plt.subplot(121)
    plt.imshow(water_data, 'gray')
    plt.title('downsample+dilate image')
    plt.subplot(122)
    plt.imshow(data_arr2, 'gray')
    plt.title('upsample image')

    # contours extract
    # data_arr2 = data_arr2/255
    data_arr2 = data_arr2.astype(np.uint8)
    _, contour, hierarchy = cv2.findContours(image=data_arr2.copy(), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contour]
    contour_seq = sorted(contour_sizes, key=lambda x: x[0], reverse=True)
    size = [size[0] for size in contour_seq]
    size = np.array(size)
    print(size)
    last = np.where(size < 7000)[0][0]
    draw_cnts = [p[1] for p in contour_seq[0:last + 1]]
    res_img = np.zeros(data_arr2.shape, np.uint8)
    cv2.drawContours(res_img, np.array(draw_cnts), -1, 1, -1)
    res_img = res_img * 255

    plt.figure(1)
    plt.subplot(121)
    plt.imshow(band_data, 'gray')
    plt.title('original data')
    plt.subplot(122)
    plt.imshow(res_img, 'gray')
    plt.title('final image')

    # write into tif
    mask = res_img[:rows, :cols]
    ds_path = "/home/zy/data_pool/Flood/Argentina/water mask/processed_flooded_soy2.tif"
    des_ds = gdal.GetDriverByName("GTiff").Create(ds_path, cols, rows, 1)
    des_ds.SetProjection(src_ds.GetProjection())
    des_ds.SetGeoTransform(src_geo_trans)
    des_ds.GetRasterBand(1).WriteArray(mask, 0, 0)

if __name__ == "__main__":
    process_list = [
        '/home/zy/data_pool/Flood/Argentina/water mask/S1A_IW_GRDH_1SDV_20160419T092222_20160419T092242_'
        '010888_0104F5_7DD1_GRD_water.tif',
        '/home/zy/data_pool/Flood/Argentina/water mask/S1A_IW_GRDH_1SDV_20160207T092213_20160207T092238_'
        '009838_00E67B_87B6_open_water.tif',
        '/home/zy/data_pool/Flood/Argentina/water mask/flooded_soy.tif'
    ]
    from searover.AG_water import *
    water_generate(process_list[2], (7, 7), 2)
    json_path = '/tmp/geo.json'
    geojson_dict = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[]]
                }
            }
        ]
    }

    # for process_file in process_list:
    # grd2water(process_file)
    # flood_mask(process_list[0], process_list[1], geojson_dict, json_path)
