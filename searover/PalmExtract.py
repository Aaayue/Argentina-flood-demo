import os
import glob
import time
import gdal
import cv2
import logging
import subprocess
import scipy.ndimage
import numpy as np
import scipy.signal as signal
import xml.etree.ElementTree as ET


class PalmExtract:
    my_logger = logging.getLogger(__qualname__)

    def __init__(
        self,
        process_file: str,
        result_root: str,  # will be use in future
        aux_path_dict: str,
        process_dict: dict,
    ):
        self.home_dir = aux_path_dict["home_dir"]
        self.SLC_aux_dir = os.path.join(self.home_dir, process_dict["aux_dir"])
        self.local_path = os.path.join(self.home_dir, process_file)
        self.process_dict = process_dict

        # creat path
        self.my_logger.info("creat process path!")
        try:
            self.input_file = glob.glob(os.path.join(self.local_path, "*_GRD.dim"))[
                0
            ]  # /home/tq/tq-data01/sentinel_GRD/S1*_TIME/S1*_GRD.dim

            self.result_file = self.input_file.replace(
                "_GRD.dim", self.process_dict["part_file_format"][0]
            )  # /home/tq/tq-data01/sentinel_GRD/S1*_TIME/S1*_GRD_TC_DB.dim

            self.process_file = self.result_file.replace(".dim", ".data")
            # /home/tq/tq-data01/sentinel_GRD/S1*_TIME/S1*_GRD_TC_DB.data
        except Exception as e:
            self.my_logger.debug("creat path failed: %s", e)
            return None

    def check_status(self) -> int:
        """
        Function:
            check palm
        output:
            1: palm tif is ok
            2: preprocess is ok
        """
        self.my_logger.info("check status!")
        tif_path = glob.glob(os.path.join(self.local_path, "*_palm_*.tif"))
        img_path = glob.glob(
            os.path.join(self.local_path, "*_GRD_TC_DB_*", "Sigma*.img")
        )
        if len(tif_path) >= 1:
            self.my_logger.info(" %s has been process", self.local_path)
            return 0
        elif len(img_path) >= 2:
            return 1
        else:
            return 2

    def palm_preprocess(self) -> bool:
        """
        Function:
            preprocess the data include TC and DB
        """
        # creat xml and run
        self.my_logger.info("creat xml!")
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
                    "GPT run finish! process sucess! %s", self.process_file
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

    def extract_palm(self, palm_value: (float, float, float) = (6.7, -9, -5)) -> bool:
        """
        Function:
            extract palm, (Sigma0_VV_db - Sigma0_VH_db) > 6.7 and  Sigma0_VV_db > -10
        Input:
            palm_value[0]: more than 6.7
            palm_value[1-2]: between -9 and -5
        """
        self.my_logger.info("extract palm!")
        process_key = ["Sigma0_VV_db", "Sigma0_VH_db"]
        File_Path = {
            pk: os.path.join(self.process_file, pk + ".img") for pk in process_key
        }

        # get data
        bandfiles = {}
        bandraster = {}
        try:
            for band_type in process_key:
                bandfiles[band_type] = gdal.Open(File_Path[band_type])
                bandraster[band_type] = (
                    bandfiles[band_type].GetRasterBand(1).ReadAsArray()
                )
        except Exception as e:
            self.my_logger.warning("Unable to open %s", e)
            return False

        # get palm  mask
        Palm_mask1 = np.subtract(
            bandraster["Sigma0_VV_db"].astype(float),
            bandraster["Sigma0_VH_db"].astype(float),
        )
        Palm_mask1 = np.greater(Palm_mask1, palm_value[0])
        Palm_mask2 = np.logical_and(
            np.greater(bandraster["Sigma0_VV_db"], palm_value[1]),
            np.less(bandraster["Sigma0_VV_db"], palm_value[2]),
        )
        Palm_mask = np.multiply(Palm_mask1, Palm_mask2)

        # post preprocess
        (cols, rows) = Palm_mask.shape
        Palm_mask = scipy.ndimage.zoom(Palm_mask, 0.125, order=0)
        Palm_mask = signal.medfilt(Palm_mask, (5, 5))

        # erode and
        Palm_mask = cv2.morphologyEx(
            Palm_mask, cv2.MORPH_CLOSE, kernel=(3, 3), iterations=1
        )

        Palm_mask = cv2.resize(Palm_mask, (rows, cols), interpolation=cv2.INTER_AREA)
        # Palm_mask = scipy.ndimage.zoom(Palm_mask, 8, order=0) size change by zhou
        # get information for geo
        geotransform = bandfiles["Sigma0_VV_db"].GetGeoTransform()
        dst_proj = bandfiles["Sigma0_VV_db"].GetProjection()
        [cols, rows] = Palm_mask.shape

        # creat output
        self.my_logger.info("output result data!")
        try:
            driver = gdal.GetDriverByName("GTiff")
        except Exception:
            self.my_logger.debug("Creat driver failed!")
            return False

        file_name = "_palm_" + time.strftime("%Y%m%dT%H%M%S") + ".tif"
        result_file = self.process_file.replace(".data", file_name)
        print(result_file)
        try:
            palm_out = driver.Create(result_file, rows, cols, 1, gdal.GDT_Byte)
        except Exception as e:
            print("Creat output failed! %s", e)
            self.my_logger.debug("Creat output failed! %s", e)
            return False

        palm_out.SetGeoTransform(geotransform)
        palm_out.SetProjection(dst_proj)
        palm_out.GetRasterBand(1).WriteArray(Palm_mask)
        palm_out.FlushCache()
        palm_out = None
        return True


if __name__ == "__main__":

    process_list = glob.glob("/home/tq/tq-data03/sentinel_GRD/*/*")
    # process_list = glob.glob("/home/tq/tq-data04/sentinel_GRD/*/*")
    count = 0
    process_file = [f.replace("/home/tq/", "") for f in process_list]
    for tmp_file in process_file:
        st = time.time()
        count += 1
        logger.info("%s, %s", count, tmp_file)
        PE = PalmExtract(
            tmp_file, "", settings.aux_path_dict, settings.process_dict_palm
        )
        if PE is None:
            continue
        else:
            # check status
            flag = PE.check_status()

            # process
            if flag:
                continue
            else:
                flag = PE.palm_preprocess()
                print("palm preprocess ", flag)

                # extract palm
                flag = PE.extract_palm()
                print(time.time() - st, flag)
