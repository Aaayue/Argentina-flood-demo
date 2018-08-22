import pytest
# import os
# import glob
from searover.WaterExtract import WaterExtract
from searover.script import settings
import numpy as np
# from osgeo import gdal


file_path = '/home/zy/data_pool/Flood/tempRes/'
water_range = settings.water_range
we = WaterExtract(file_path, water_range)


def test_seg_td():
    arr = np.random.rand(20, 20)
    new_arr, ret, flag = we.seg_td(img=arr, value=255)
    assert np.shape(new_arr) == (20, 20)


def test_waterextract():
    we.water_extract()
