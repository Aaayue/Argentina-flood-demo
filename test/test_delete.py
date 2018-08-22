import searover.copy_clean_local as CCL
import logging

logger = logging.getLogger(__name__)


def test_cpoy_clean():
    """
    test copy and clean function
    """
    home_dir = "/home/xyz"
    result_root = "tq-data01"
    root_folder = "sentinel1_GRD"
    local_path = "/home/xyz/data_pool/test_data/sentinel_GRD/77"
    suffix_pattern = suffix_pattern = ["/*.data", "/*.dim", "/*.SAFE"]

    logger.info("Delete test")
    res_path, flag = CCL.copy_clean_local(
        home_dir, result_root, root_folder, local_path, suffix_pattern
    )
    logger.info("%s, %s", res_path, flag)
