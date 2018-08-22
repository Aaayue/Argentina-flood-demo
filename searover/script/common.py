import os
import re
import sys
import logging
import colorlog

script_path = os.path.abspath(__file__)
package_path = re.findall(".*/searover", script_path)[0]
dir = os.path.dirname(package_path)
sys.path.append(dir)

from searover.progress_notify import ProgressStream  # noqa: E402

# use root logger below
# format = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
# logging.basicConfig(format=format, level=logging.DEBUG)


handler = logging.StreamHandler(ProgressStream())
handler.setLevel(logging.DEBUG)

# handler.setFormatter(logging.Formatter(format))
handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s | %(levelname)s | %(name)s-%(process)d | %(message)s",
        # datefmt="%Y-%d-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "blue",
            "SUCCESS:": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
)

logger = logging.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
# logger.propagate = False


logging.getLogger("matplotlib").setLevel(logging.INFO)
