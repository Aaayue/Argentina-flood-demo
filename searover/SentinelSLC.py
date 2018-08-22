import os
import glob
import subprocess
import logging
import time
import xml.etree.ElementTree as ET
from pathlib import Path


class SentinelSLC:
    my_logger = logging.getLogger(__qualname__)

    def __init__(
        self,
        process_file: str,
        result_root: str,
        aux_path_dict: dict,
        process_dict: dict,
    ):
        self.result_root = result_root
        self.root_folder = aux_path_dict["root_folder"]
        self.home_dir = aux_path_dict["home_dir"]
        self.process_dict = process_dict
        self.process_file = os.path.join(
            self.home_dir, process_file
        )  # /home/tq/tq-data0*/sentinel1/*/*.zip

        self.local_path = os.path.join(
            self.home_dir,
            process_file.split("/S1")[0].replace("sentinel1", self.root_folder),
            os.path.split(process_file)[1].replace(
                ".zip", "_" + time.strftime("%Y%m%dT%H%M%S")
            ),
        )  # /home/tq/tq-data0*/root_folder/*
        self.orbital_number = process_file.split("/")[2]

        self.result_file_name = (
            Path(self.process_file)
            .name.replace("SLC", "GRDH")
            .replace(".zip", process_dict["part_file_format"][-1])
        )  # *._GRD.dim

        self.server_list = aux_path_dict["server_list"]
        self.SLC_aux_dir = os.path.join(self.home_dir, process_dict["aux_dir"])

    def check_process_status(self) -> (str, bool):
        """
        Function:
            Check whether the data has been processed
        output:
            str, True: return result path
            None, False: return None and false
        """
        # check the result
        for tmp_server in self.server_list:
            tmp_path = os.path.join(
                self.home_dir,
                tmp_server,
                self.root_folder,
                self.orbital_number,
                "*",
                self.result_file_name.replace("dim", "data"),
            )
            # find the result path
            img_list = glob.glob(os.path.join(tmp_path, "*.img"))
            if len(img_list) >= 4:
                return str(Path(img_list[0]).parent.parent).replace(self.home_dir, "").strip('/'), True
            else:
                continue
        return None, False

    def unzip_data_local(self) -> bool:
        """
        Function:
            unzip the data to local path
        input:
            process data path
            tq-data04/sentinel1/3/S1A_IW_SLC__1SDV_20180517T220619_20180517T220646_021950_025ECB_DA3B.zip
        output:
            (False): unzip failue
            (Ture): return local path:/home/tq/tq-tmp/tq-data*/sentinel1/3
        """
        #  check the raw data
        if not os.path.exists(self.process_file):
            self.my_logger.warning("%s process file does not exist!", self.process_file)
            return False
        else:
            if not os.path.exists(self.local_path):
                os.makedirs(self.local_path)

            # creat local path (overwrite) to process the data
            process_flag = subprocess.run(
                ["unzip", "-o", self.process_file, "-d", self.local_path]
            )
            if process_flag.returncode == 0:
                self.my_logger.info("Raw data unzip sucess. %s", self.process_file)
                return True
            else:
                self.my_logger.warning("Raw data unzip! %s", self.process_file)
                return False

    def creat_process_xml(self) -> (str, bool):
        """
        Function:
            create process xml (i.e. SLC2GRD.xml / Terrian.xml)
        input:
            local_path i.e. /home/tq/tq-tmp/sentinel1/2
        output:
            False: creat xml failue
            True: creat xml sucess
        """
        os.chdir(self.local_path)
        part_xml = [
            os.path.join(self.local_path, self.process_dict["xml_file_names"][index])
            for index in range(self.process_dict["parts_num"])
        ]

        part_result_path = [
            os.path.join(
                self.local_path,
                self.result_file_name.replace(
                    self.process_dict["part_file_format"][-1],
                    self.process_dict["part_file_format"][index],
                ),
            )
            for index in range(self.process_dict["parts_num"])
        ]

        # Add origin *safe file
        origin_file = os.path.join(
            self.local_path,
            self.result_file_name.replace("GRDH", "SLC").replace(
                self.process_dict["part_file_format"][-1], ".SAFE/manifest.safe"
            ),
        )
        part_result_path.insert(0, origin_file)

        for index in range(self.process_dict["parts_num"]):
            base_part_xml = os.path.join(
                self.SLC_aux_dir, self.process_dict["base_xml_files"][index]
            )

            tree = ET.ElementTree(file=base_part_xml)
            root = tree.getroot()

            for child in root.iter(tag="file"):  # set new xml for process
                if child.text == "process_file":
                    child.text = part_result_path[index]
                    continue
                elif child.text == "result_file":
                    child.text = part_result_path[index + 1]
                    continue
                else:
                    raise ValueError("Cannot find node, please check the xml file")
                    return None, False
            try:
                tree.write(part_xml[index])
            except Exception as e:
                self.my_logger.warning("write xml failed:", part_xml[index])
                return None, False

        return part_xml, True

    def gpt_process(self, process_xml) -> bool:
        """
        Function:
            use gpt to process sentinel 1
        input:
            process-xml is based on config_xml /data_pool/SLC_preprocess_aux/*.xml
        output:
            None, False: process failue
            str, True: return the path
        """
        for index in range(self.process_dict["parts_num"]):
            xml = process_xml[index]
            self.my_logger.info("Process xml %s", xml)

            if (self.process_dict["xml_file_names"][index]) not in xml:
                self.my_logger.info(
                    "Xml file name error! process failed! %s part %d",
                    self.process_file,
                    (index + 1),
                )
                return False

            try:
                process_flag = subprocess.run(["gpt", xml])
                if process_flag.returncode == 0:
                    self.my_logger.info(
                        "GPT run finish! process sucess! %s part %d",
                        self.process_file,
                        (index + 1),
                    )
                    continue
                else:
                    self.my_logger.warning(
                        "GPT run finish! process failed! %s part %d",
                        self.process_file,
                        (index + 1),
                    )
                    return False
            except Exception:
                self.my_logger.warning(
                    "Try GPT error! process failed! %s part %d",
                    self.process_file,
                    (index + 1),
                )
                return False

        return True

    def preprocess(self) -> (str, int):
        """
        Function:
            ****************sentinel1 SLC preprocess****************
            Step 1: check process data status
            Step 2: unzip raw  data
            Step 3: creat process xml
            Step 4: gpt process
        ouput:
            process_status
            0: sucess
            1: data has been processed
            2: unzip data has problem
            3: creat xml failed
            4: gpt falied
        """
        # Step 1
        self.my_logger.info("Step 1: will be check process data status!")
        result_path, flag = self.check_process_status()
        if flag:
            process_status = 1
            return result_path, process_status
        else:
            self.my_logger.info("Step 2: will be unzip raw data")

        # Step 2
        flag = self.unzip_data_local()
        if not flag:
            process_status = 2
            return self.local_path, process_status
        else:
            self.my_logger.info("Step 3: will be creat process xml!")

        # Step 3

        process_xml, flag = self.creat_process_xml()
        if not flag:
            process_status = 3
            return self.local_path, process_status
        else:
            self.my_logger.info("Step 4: gpt process.")

        # Step 4
        flag = self.gpt_process(process_xml)
        if not flag:
            process_status = 4
            return self.local_path, process_status
        else:
            process_status = 0
            self.my_logger.info("All finished.")
            return self.local_path.replace(self.home_dir, "").strip('/'), process_status
