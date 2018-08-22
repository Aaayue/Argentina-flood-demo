import time
import json

import settings
from common import logger

import searover.SentinelSLC as FMSLC
import searover.PostLocal as FMPL


def slc2grd(process_tile: str, result_root: str) -> (str, int):
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

    # preprocess
    SLC = FMSLC.SentinelSLC(
        process_file, result_root, settings.aux_path_dict, settings.process_dict_slc2grd
    )
    local_path, status = SLC.preprocess()
    if status != 0 and status != 1:
        logger.warning("preprocess failed!")
        return None, status
    elif status == 1:
        logger.info("data has been processed!")
        return local_path, status
    else:
        logger.info("preprocess success!")

    # Post process
    PL = FMPL.PostLocal(local_path, settings.process_dict_slc2grd)
    res_path, flag = PL.post_local()
    if not flag:
        logger.warning("copy and clean local path failed!")
        return None, 5
    else:
        logger.info("copy and clean local path success!")

    return res_path, 0


if __name__ == "__main__":

    start = time.time()
    file_name = "/home/tq/data_pool/Flood/repalce_two.json"
    with open(file_name, "r") as fp:
        process_list = json.load(fp)
    print(process_list)
    hostname = "tq-data03"
    process_list = [f for f in process_list if hostname in f]
    result_root = "tq-data03"
    print(process_list)
    for process_file in process_list:
        res_path, flag = slc2grd(process_file, result_root)
        logger.debug("%s, %s", res_path, flag)
        end = time.time()
        logger.debug("Task runs %0.2f seconds" % (end - start))
