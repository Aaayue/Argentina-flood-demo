import time
import os
import settings
import shutil
from common import logger

import searover.SentinelSLC as FMSLC
import searover.PostLocal as FMPL
import searover.PalmExtract as FMPE
import searover.WaterExtract as FMWE

import glob


def go_slc2grd(process_file: str, result_root: str) -> (str, int):
    """
    Function:
        ****************slc2grd****************
        Step 1: Sentinel-1 preprocess
    ouput:
        process_status
        0: sucess
        1: data has been processed
        2: unzip data has problem
        3: creat xml failed
        4: gpt falied
        5: post process failed
    """
    # logger.info("{} will be process slc to grd".format(process_file))

    # preprocess
    SLC = FMSLC.SentinelSLC(
        process_file, result_root, settings.aux_path_dict, settings.process_dict_slc2grd
    )
    local_path, status = SLC.preprocess()
    if status != 0 and status != 1:
        logger.warning("preprocess failed!")
        print('local_path_delet:', local_path)
        shutil.rmtree(local_path)
        # os.remove(local_path)
        return None, status
    elif status == 1:
        logger.info("data has been processed!")
        return local_path, status
    else:
        logger.info("preprocess success!")

    # Post process
    PL = FMPL.PostLocal(
        local_path, settings.aux_path_dict, settings.process_dict_slc2grd
    )
    res_path, flag = PL.post_local()
    if not flag:
        logger.warning("copy and clean local path failed!")
        return None, 5
    else:
        logger.info("copy and clean local path success!")

    return res_path, 0


def go_water(process_file: str, result_root: str) -> bool:
    """
    Function:
        ****************Water_Extract****************
        Step 1: Sentinel-1 water preprocess
        Step 2: open water_extract
        Step 3: close water_extract
        Step 4: water_mask
    input:
        local_path: "tq-data01/sentinel_GRD/*/*"
    ouput:
        process_status
        0: sucess
        1: preprocess failed
        2: open water_extract failed
        3: close water_extract failed
        4: water_mask failed
    """
    # logger.info("{} will be process water extract.".format(process_file))

    WE = FMWE.WaterExtract(
        process_file, result_root, settings.aux_path_dict, settings.process_dict_water
    )
    if WE is None:
        logger.debug("init failed")
        return None

    # check status
    check_flag = WE.check_status()

    # process
    if check_flag == 0:
        logger.info("File has been processed before!")
    else:
        if check_flag == 1:
            logger.info("water pre-process {}".format(check_flag))
            preprocess_flag = WE.water_preprocess()
            if not preprocess_flag:
                return 1
        # extract water
        if check_flag in [1, 2]:
            logger.info("Do open water extract next...")
            open_flag = WE.water_extract()
            if not open_flag:
                return 2
        if check_flag in [1, 3]:
            logger.info("Do close water extract next...")
            close_flag = WE.close_water_run()
            if not close_flag:
                return 3
            if close_flag:
                logger.info("Will do water mask")
                water_flag = WE.water_mask()
                if not water_flag:
                    return 4
    logger.info("water extract done")
    return 0


def go_palm(process_file: str, result_root: str) -> bool:
    """
    Function:
        ****************Palm_Extract****************
        Step 1: Sentinel-1 palm preprocess
        Step 2: palm water_extract
    input:
        local_path: "tq-data01/sentinel_GRD/*/*"
    ouput:
        process_status
        0: sucess
        1: preprocess failed
        2: palm failed
    """
    # logger.info("{} will be process palm extract.".format(process_file))

    PE = FMPE.PalmExtract(
        process_file, result_root, settings.aux_path_dict, settings.process_dict_palm
    )
    if PE is None:
        logger.debug("init failed")
        return None
    else:
        # check status
        check_flag = PE.check_status()

        # process
        if check_flag == 0:
            return check_flag
        elif check_flag == 1:
            logger.info("will be run palm_extract.")
            palm_flag = PE.extract_palm()
            if not palm_flag:
                return 2
        elif check_flag == 2:
            logger.info("will be run palm_preprocess.")
            preprocess_flag = PE.palm_preprocess()
            if not preprocess_flag:
                return 1
            else:
                palm_flag = PE.extract_palm()
                if not palm_flag:
                    return 2
                else:
                    return 0


def go_main(process_file: str, result_root: str, process_flag: int) -> (str, int):
    """
    Function:
        ****************main_palm_water****************
        Step 1: Sentinel-1 preprocess [0, 1, 2, 3, 4]
        Step 2: water extract [0, 5]
        Step 3: Palm extract [0, 6]
        step 4: postprocess [0, 7]
    input:
        process_tile: tq-data01/sentinel/n/S1*.zip
        result_root: put the result to which server
        process_flag:
                0 : slc2grd
                1 : slc2grd -> water
                2 : slcgrd -> palm
                3 : slc -> water -> palm
    ouput:
        process_status
        0: sucess
        1: data has been processed
        2: unzip data has problem
        3: creat xml failed
        4: gpt falied
        5: palm extract failed
        6: water extract failed
        7: water and palm failed
    """
    if process_flag == 0:
        local_path, slc2grd_flag = go_slc2grd(process_file, result_root)
        logger.info(
            "{} process status {}, {}".format(process_file, local_path, slc2grd_flag)
        )
        if slc2grd_flag == 0:
            return 0
        else:
            return slc2grd_flag
    elif process_flag == 1:
        local_path, slc2grd_flag = go_slc2grd(process_file, result_root)
        logger.info(
            "{} process status {}, {}".format(process_file, local_path, slc2grd_flag)
        )
        if slc2grd_flag == 0:
            water_flag = go_water(local_path, result_root)
            logger.info(
                "{} water process status {}, {}".format(
                    process_file, local_path, water_flag
                )
            )
            if water_flag == 0:
                return 0
            else:
                return 6
        else:
            return slc2grd_flag
    elif process_flag == 2:
        local_path, slc2grd_flag = go_slc2grd(process_file, result_root)
        logger.info(
            "{} process status {}, {}".format(process_file, local_path, slc2grd_flag)
        )
        if slc2grd_flag == 0:
            palm_flag = go_palm(local_path, result_root)
            logger.info(
                "{} palm process status {}, {}".format(
                    process_file, local_path, palm_flag
                )
            )
            if palm_flag == 0:
                return 0
            else:
                return 5
        else:
            return slc2grd_flag
    elif process_flag == 3:
        local_path, slc2grd_flag = go_slc2grd(process_file, result_root)
        logger.info(
            "{} process status {}, {}".format(process_file, local_path, slc2grd_flag)
        )
        if slc2grd_flag in [1, 0]:
            water_flag = go_water(local_path, result_root)
            logger.info(
                "{} water process status {}, {}".format(
                    process_file, local_path, water_flag
                )
            )
            print("water_flag:", water_flag)
            palm_flag = go_palm(local_path, result_root)
            logger.info(
                "{} palm process status {}, {}".format(
                    process_file, local_path, palm_flag
                )
            )
            print("palm_flag:", palm_flag)
            if water_flag == 0 and palm_flag == 0:
                return 0
            elif water_flag == 0 and palm_flag != 0:
                return 5
            elif water_flag != 0 and palm_flag == 0:
                return 6
            else:
                return 7
        else:
            print(slc2grd_flag)
            return slc2grd_flag
    else:
        logger.info("check the input process_flag")


if __name__ == "__main__":

    # process_list = glob.glob("/home/tq/tq-data01/sentinel1/256/S1A_IW_SLC__1SDV_2015*")
    process_list = [
            "tq-data01/sentinel1/256/S1A_IW_SLC__1SDV_20151130T224758_20151130T224826_008840_00CA00_5DB4.zip"
            ]
    result_root = "tq-data01"
    print(process_list)
    for process_file in process_list:
        start = time.time()
        flag = go_main(process_file, result_root, 3)
        logger.debug("%s", flag)
        end = time.time()
        logger.debug("Task runs %0.2f seconds" % (end - start))

