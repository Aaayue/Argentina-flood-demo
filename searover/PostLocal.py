import glob
import os
from os.path import join
import shutil
import logging


class PostLocal:
    my_logger = logging.getLogger(__qualname__)

    def __init__(self, local_path: str, aux_path_dict: dict, process_dict: dict):
        self.local_path = join(aux_path_dict["home_dir"], local_path)
        self.deleted_dict = process_dict["deleted_file"]

    def post_local(self) -> (str, bool):
        """
        Function:
            deleted the file, then copy file to result path
        output:
            None, False: process failue str
            True: return the path
        """

        os.chdir(self.local_path)

        # deleted temporary file
        deleted_list = []
        self.my_logger.info("Find the deleted file.")
        for tmp in self.deleted_dict:
            tmpList = glob.glob(self.local_path + tmp)
            if tmpList:
                deleted_list.extend(tmpList)

        if len(deleted_list) is 0:
            self.my_logger.warning("No file deleted! %s", self.local_path)
            return None, False

        try:
            for tmp in deleted_list:
                if os.path.isfile(tmp):
                    os.remove(tmp)
                else:
                    shutil.rmtree(tmp)
        except Exception as e:
            self.my_logger.warning("Deleted file failed! %s", e)
            return None, False
        return self.local_path, True
