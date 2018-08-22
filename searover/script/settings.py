"""'
A. all auxdata: /home/tq/data_pool/snap_aux
B. Preprocess DEM using local path
    Config file: /usr/local/snap/etc/snap.auxdata.properties
    DEM http: http://192.168.50.172:8000/auxdata/dem/SRTMGL1
    DEM local path:the local data in /homq/tq/data_pool/SRTML1/auxdata/dem/SRTMGL1
"""

# Preprocess arguments
aux_path_dict = {
    "home_dir": "/home/tq",  # docker must be processed by user tq
    "root_folder": "sentinel_GRD",
    "server_list": ["tq-data04", "tq-data03", "tq-data02", "tq-data01"],
}

process_dict_slc2grd = {
    "parts_num": 2,
    "base_xml_files": ["SLC_to_GRD_part1.xml", "SLC_to_GRD_part2.xml"],
    "part_file_format": ["_part.dim", "_GRD.dim"],
    "xml_file_names": ["slc2grd_part1.xml", "slc2grd_part2.xml"],
    "deleted_file": ["/*_part.data", "/*_part.dim", "/*.SAFE"],
    "aux_dir": "data_pool/snap_aux/process_xml/SLC2GRD",
}

process_dict_water = {
    "parts_num": 1,
    "base_xml_files": ["GRD_TF_TC_water.xml"],
    "part_file_format": ["_GRD_TF_TC.dim"],
    "xml_file_names": ["GRD_TF_TC_water.xml"],
    "deleted_file": ["/*_TF_TC.data", "/*_TF_TC.dim"],
    "aux_dir": "data_pool/snap_aux/process_xml/Water_extract",
}

process_dict_palm = {
    "parts_num": 1,
    "base_xml_files": ["GRD_TC_DB_palm.xml"],
    "part_file_format": ["_GRD_TC_DB.dim"],
    "xml_file_names": ["GRD_TC_DB_palm.xml"],
    "deleted_file": ["/*_TC_DB.data", "/*_TC_DB.dim"],
    "aux_dir": "data_pool/snap_aux/process_xml/Palm_extract",
}
