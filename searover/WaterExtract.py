import glob
import cv2
import os
import math
import subprocess
import logging
import numpy as np
import xml.etree.ElementTree as ET
import script.settings
from osgeo import gdal
from skimage import filters
import time


class WaterExtract:
    my_logger = logging.getLogger(__qualname__)

    def __init__(
        self,
        process_file: str,
        result_path: str,
        aux_path_dict: dict,
        process_dict: dict,
    ):
        self.home_dir = aux_path_dict["home_dir"]
        self.process_dict = process_dict
        self.local_path = os.path.join(self.home_dir, process_file)
        self.SLC_aux_dir = os.path.join(self.home_dir, process_dict["aux_dir"])

        # create path
        self.my_logger.info("water creat process path!")
        try:
            self.input_file = glob.glob(os.path.join(self.local_path, "*_GRD.dim"))[0]
            # /home/tq/tq-data01/sentinel_GRD/S1*_TIME/S1*_GRD.dim
            self.result_file = self.input_file.replace(
                "_GRD.dim", self.process_dict["part_file_format"][0]
            )  # /home/tq/tq-data01/sentinel_GRD/S1*_TIME/S1*_GRD_TF_TC.dim
            self.process_file = self.result_file.replace(".dim", ".data")
            # /home/tq/tq-data01/sentinel_GRD/S1*_TIME/S1*_GRD_TF_TC.data

            self.open_water_tif = self.input_file.replace("_GRD.dim", "_open_water.tif")
            self.close_water_tif = self.input_file.replace(
                "_GRD.dim", "_close_water.tif"
            )
            self.water_tif = self.input_file.replace("_GRD.dim", "_water.tif")
        except Exception as e:
            self.my_logger.debug("create path failed: %s", e)

    def check_status(self) -> (bool, int):
        """
        Function:
            check water.tif
        """
        self.my_logger.info("check status!")
        tif_all = glob.glob(os.path.join(self.local_path, "*_water.tif"))
        open_tif = glob.glob(os.path.join(self.local_path, "*_open_water.tif"))
        close_tif = glob.glob(os.path.join(self.local_path, "*_close_water.tif"))
        prep_path = len(glob.glob(self.process_file))
        file_num = len(os.listdir(self.process_file))
        prep_done = np.logical_and(
            np.equal(prep_path, 1),
            np.equal(file_num, 7)
        )
        if len(tif_all) == 3:
            self.my_logger.info(" %s has been process", self.local_path)
            return 0
        elif not prep_done:
            self.my_logger.info("Do water pre-process next. %s", self.process_file)
            return 1
        else:
            if len(open_tif) == 0:
                self.my_logger.info("Do open water extract next. %s", self.process_file)
                return 2
            elif len(close_tif) == 0:
                self.my_logger.info(
                    "Do close water extract next. %s", self.process_file
                )
                return 3

    def water_preprocess(self) -> bool:
        """
        Function:
            preprocess the data include TF and TC
        """
        # create xml and run
        self.my_logger.info("create xml!")
        part_xml = os.path.join(self.local_path, self.process_dict["xml_file_names"][0])
        base_part_xml = os.path.join(
            self.SLC_aux_dir, self.process_dict["base_xml_files"][0]
        )
        tree = ET.ElementTree(file=base_part_xml)
        root = tree.getroot()
        for child in root.iter(tag="file"):  # set new xml for process
            if child.text == "process_file":
                child.text = self.input_file
                continue
            elif child.text == "result_file":
                child.text = self.result_file
                continue
            else:
                raise ValueError("Cannot find node, please check the xml file")
                return False
        try:
            tree.write(part_xml)
        except Exception as e:
            self.my_logger.debug("write xml failed: %s", e)
            return False

        # run gpt
        self.my_logger.info("Process xml")
        try:
            process_flag = subprocess.run(["gpt", part_xml])
            if process_flag.returncode == 0:
                self.my_logger.info(
                    "GPT run finish! process success! %s", self.process_file
                )
                return True
            else:
                self.my_logger.warning(
                    "GPT run finish! process failed! %s", self.process_file
                )
                return False
        except Exception:
            self.my_logger.warning(
                "Try GPT error! process failed! %s", self.process_file
            )
            return False

    def seg_td(
        self,
        img: bytearray,
        thr_value: int = 1,
        cnt_value: float = 35.0,
        wid: tuple = (15, 15),
        step: int = 4,
        water_range: tuple = (-12, -16),
    ) -> (bytearray, float):
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
        # get threshold
        blur = cv2.bilateralFilter(img, 5, 25 * 2, 25 / 2)
        blur[np.isnan(blur)] = 0.0
        blur[np.isinf(blur)] = 0.0
        data_mask2 = blur[blur != 0]
        threshold = filters.threshold_otsu(data_mask2)
        if (threshold > water_range[0]) | (threshold < water_range[1]):
            self.my_logger.warning("Use -13 as threshold here")
            threshold = -13

        ret1, bin_img = cv2.threshold(blur, threshold, thr_value, cv2.THRESH_BINARY_INV)
        bin_img = bin_img.astype(np.uint8)
        mor_img = cv2.morphologyEx(bin_img, cv2.MORPH_DILATE, kernel=wid, iterations=step)

        # contours extract
        _, contour, hierarchy = cv2.findContours(image=mor_img.copy(), mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_NONE)
        contour_sizes = [(cv2.contourArea(contour), contour) for contour in contour]
        contour_seq = sorted(contour_sizes, key=lambda x: x[0], reverse=True)
        size = [size[0] for size in contour_seq]
        size = np.array(size)
        last = np.where(size < cnt_value)[0][0]
        draw_cnts = [p[1] for p in contour_seq[0:last + 1]]
        res_img = np.zeros(mor_img.shape, np.uint8)
        cv2.drawContours(res_img, np.array(draw_cnts), -1, 1, -1)
        return res_img, ret1

    def water_extract(self) -> bool:
        """
        Function:
            post-process tif image
        :param
            self: input .tif file path
        :return:
            src_ds: geo information of input image
            new_img: an nd-array of the processed image
        """

        # find raw tif and create water tif
        try:
            raw_tif = os.path.join(self.process_file, "Gamma0_VV.img")
        except Exception as e:
            self.my_logger.debug("Please recheck the file path! %s", e)
            return False

        # get information from raw_tif
        try:
            src_ds = gdal.Open(raw_tif)
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
            self.my_logger.info("Open tiff succeed!")
        except Exception as e:
            self.my_logger.debug("Open failed! T-T %s", e)
            return False

        # get threshold
        new_img, ret1 = self.seg_td(data_orig, 1)
        self.my_logger.info("Threshold is %.5f", ret1)
        self.my_logger.info("Finish segmentation.")

        # write tif to local path
        try:
            des_ds = gdal.GetDriverByName("GTiff").Create(
                self.open_water_tif, cols, rows, 1
            )
            des_ds.SetProjection(src_ds.GetProjection())
            des_ds.SetGeoTransform(src_geo_trans)
            des_ds.GetRasterBand(1).WriteArray(new_img, 0, 0)
            self.my_logger.info("Writing-TIFF Done! %s", self.local_path)
            return True
        except Exception as e:
            self.my_logger.debug("Writing failed! T-T %s %s", self.local_path, e)
            return False

    def water_mask(self) -> bool:
        """
        Function:
            overlap close water mask and open water mask together
        input:
            open water tiff path
            close water tiff path
        output:
            water mask tiff
        """
        # open water
        try:
            src_water1 = gdal.Open(self.open_water_tif)
            geo_trans1 = src_water1.GetGeoTransform()
            self.my_logger.info("open water geo: ", geo_trans1)

            cols1 = src_water1.RasterXSize
            rows1 = src_water1.RasterYSize
            self.my_logger.info("open water size: ", (cols1, rows1))

            band1 = src_water1.GetRasterBand(1)
            arr1 = band1.ReadAsArray(0, 0, cols1, rows1)
            arr1 = arr1 / 255
        except Exception as e:
            self.my_logger.debug("read open_water failed! %s", e)
            return False

        # close water
        try:
            src_water2 = gdal.Open(self.close_water_tif)
            geo_trans2 = src_water2.GetGeoTransform()
            self.my_logger.info("close water geo: ", geo_trans2)

            cols2 = src_water2.RasterXSize
            rows2 = src_water2.RasterYSize
            self.my_logger.info("close water size: ", (cols2, rows2))

            band2 = src_water2.GetRasterBand(1)
            arr2 = band2.ReadAsArray(0, 0, cols2, rows2)
        except Exception as e:
            self.my_logger.debug("read open_water failed! %s", e)
            return False

        # result water
        try:
            mask = arr1 + arr2
            mask[mask != 0] = 1
        except Exception as e:
            self.my_logger.debug("Overlay failed! T-T %s", e)
            return False

        #  output result
        try:
            des_ds = gdal.GetDriverByName("GTiff").Create(
                self.water_tif, cols1, rows1, 1
            )
            des_ds.SetProjection(src_water1.GetProjection())
            des_ds.SetGeoTransform(geo_trans1)
            des_ds.GetRasterBand(1).WriteArray(mask, 0, 0)
            self.my_logger.info("Writing-TIFF Done! %s", self.water_tif)
            return True
        except Exception as e:
            self.my_logger.debug("Writing failed! T-T %s %s", self.water_tif, e)
            return False

    def close_water_run(self, zscale: int = 20) -> bool:
        """
        zscale: scale coef when zooming is needed
        """
        self.my_logger.info("close water running")
        try:
            dem_file_path = os.path.join(self.process_file, "elevation.img")
        except Exception as e:
            self.my_logger.debug("File not found! %s", e)

        try:
            dem = gdal.Open(dem_file_path)
            dem_geo_trans = dem.GetGeoTransform()
            cols = dem.RasterXSize
            rows = dem.RasterYSize
            band = dem.GetRasterBand(1)
            mx_dem = band.ReadAsArray(0, 0, cols, rows)
        except Exception as e:
            self.my_logger.debug("Open file failed! %s", e)
            return False

        seedlist = []
        seedlistz = seedlist.copy()

        if len(seedlist) == 0:
            self.my_logger.info("No seed for close water!")
            return False

        # downscaling img, z for zoomed
        demshape = list(mx_dem.shape)
        zshape = [int(i / zscale) for i in demshape]
        zshape = [zshape[1], zshape[0]]
        mxz_dem = cv2.resize(mx_dem, tuple(zshape))

        seedlistz = [
            [int(seedlistz[i][j] / zscale) for j in range(len(seedlistz[i]))]
            for i in range(0, len(seedlistz))
        ]
        maskz = self.gen_buff_resistance(mxz_dem, 500, seedlistz, 800.0, 1.3)

        mxz_dem = mxz_dem + maskz * 540
        fldz = self.region_grow(mxz_dem, 67, seedlistz)
        demshape = [demshape[1], demshape[0]]
        fld = cv2.resize(fldz, tuple(demshape))
        fld = fld.astype(np.int8)

        try:
            outds = gdal.GetDriverByName("GTiff").Create(
                self.close_water_tif, cols, rows, 1
            )
            outds.SetProjection(dem.GetProjection())
            outds.SetGeoTransform(dem_geo_trans)
            outds.GetRasterBand(1).WriteArray(fld, 0, 0)
            self.my_logger.info("Writing TIFF Done! %s", self.local_path)
            return True
        except Exception as e:
            self.my_logger.debug("Writing file failed! %s %s", self.local_path, e)
            return False

    def gen_buff_resistance(
        self, img: np.ndarray, radius: int, slist: list, scale0: float, scale1: float
    ) -> bytearray:
        """
        Function:
            generate a resistance buffer, which describes the resistance of
            water flowing to next pixel. the resistance at point p will increase
            when the distance increases between p and nearest seed points.
            the returned resistance buffer will add to dem to simply rise the altitude
            as the resistance. then region-grow algorithm will be restrained by a higher
            altitude when it's far away from seed points.
        :param
            self:
            img: input dem array
            radius: buffer radius, useless in this func
            slist: a list contains several seeds' coordinate
            scale0: scale factor controls the resistance radius, 100 for recommand
                    default value
            scale1: scale factor controls the resistance increasing rate
        :return:
            out: an nd-array of the resistance buffer img, usually between 0 and 1
        """
        self.my_logger.info("Making resist-buffer...")
        shape = img.shape
        xmax = shape[0]
        ymax = shape[1]
        out = np.zeros(shape, dtype=float)
        dmax0 = (xmax * 1.0) * (xmax * 1.0) + (ymax * 1.0) * (ymax * 1.0)
        dmax0 = dmax0 / scale0
        slist0 = slist.copy()
        disarr = [0] * len(slist0)
        for i in range(0, xmax - 1):
            for j in range(0, ymax - 1):
                for s in range(0, len(disarr)):
                    cor = slist0[s]
                    x = cor[0] * 1.0
                    y = cor[1] * 1.0
                    disarr[s] = (i - x) * (i - x) + (j - y) * (j - y)
                dmax = min(disarr)
                dd = dmax / dmax0
                # use logistic func to normalize the distance weight
                dd1 = 1 / (1 + math.exp(dd * (-1)))
                dd1 = dd1 * 2 - scale1
                if dd1 < 0:
                    dd1 = 0
                out[i, j] = dd1
        return out

    def region_grow(self, img: np.ndarray, thre: float, slist: list) -> bytearray:
        """
        Function:
            fill a region by growing from some seed points in slist,
            grow direction is 4-connection by default
        :param
            self:
            img: input image array
            thre: threshold when growing
            slist: a list contains several seeds' coordinate
        :return:
            out: an nd-array of the processed image
        """
        self.my_logger.info("Growing...")
        slist0 = slist.copy()
        shape = img.shape
        xmax = shape[0]
        ymax = shape[1]
        out = np.zeros(shape, dtype=np.float)
        dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        ndir = 4
        num = 0
        while len(slist0) > 0:
            co = slist0.pop()
            x0 = co[0]
            y0 = co[1]
            out[x0, y0] = 1
            for i in range(0, ndir):
                ii = dirs[i]
                cor = [co[i] + ii[i] for i in range(len(ii))]  # list add each other
                # is out of edge
                if (
                    (cor[0] >= xmax)
                    or (cor[0] < 0)
                    or (cor[1] >= ymax)
                    or (cor[1] < 0)
                    or (img[cor[0], cor[1]] <= -1000)
                ):
                    continue
                # spread to neighbor pixels
                if img[cor[0], cor[1]] <= thre:
                    if out[cor[0], cor[1]] == 0:
                        out[cor[0], cor[1]] = 1
                        num = num + 1
                        slist0.append(cor)
        return out


if __name__ == "__main__":

    st = time.time()
    # json_file = "/home/tq/data_pool/Flood/Sentinel1-List-full.json"
    # with open(json_file, "r") as fp:
    #     process_list = json.load(fp)
    # process_list = list(set(process_list))
    # hostname = "tq-data03"
    # process_list = [f for f in process_list if hostname in f]
    process_list = [
            "data_pool/Flood/Argentina/process data/S1A_IW_GRDH_1SSV_20160121T225423_20160121T225452_009598_00DF90_08F6",
            "data_pool/Flood/Argentina/process data/S1A_IW_GRDH_1SSV_20160426T225424_20160426T225454_010998_01086F_FF7A",
            "data_pool/Flood/Argentina/process data/S1A_IW_GRDH_1SDV_20160207T092213_20160207T092238_009838_00E67B_87B6",
            "data_pool/Flood/Argentina/process data/S1A_IW_GRDH_1SDV_20160419T092222_20160419T092242_010888_0104F5_7DD1",
            "data_pool/Flood/Argentina/process data/S1A_IW_GRDH_1SSV_20160402T225518_20160402T225543_010648_00FDC7_16B4",
            "data_pool/Flood/Argentina/process data/S1A_IW_GRDH_1SSV_20160121T225452_20160121T225517_009598_00DF90_D1F7",
            "data_pool/Flood/Argentina/process data/S1A_IW_GRDH_1SSV_20160426T225454_20160426T225519_010998_01086F_EBF3",
            "data_pool/Flood/Argentina/process data/S1A_IW_GRDH_1SSV_20160426T225519_20160426T225544_010998_01086F_D6D8"
            ]
    for process_file in process_list:
        WE = WaterExtract(
            process_file, "", settings.aux_path_dict, settings.process_dict_water
        )
        if WE is None:
            WE.my_logger.debug("init failed")

        # check status
        flag = WE.check_status()
        # print(flag)

        # process
        if flag == 0:
            WE.my_logger.info("File has been processed before!")
        else:
            if flag == 1:
                res = WE.water_preprocess()
                WE.my_logger.info("water pre-process %s", flag)
            # extract water
            if flag in [1, 2]:
                WE.my_logger.info("Do open water extract next...")
                res = WE.water_extract()
            if flag in [1, 3]:
                WE.my_logger.info("Do close water extract next...")
                res = WE.close_water_run()
                if res:
                    WE.my_logger.info("Will do water mask")
                    flag = WE.water_mask()
        WE.my_logger.info("water extract done")
        WE.my_logger.info(time.time() - st)


