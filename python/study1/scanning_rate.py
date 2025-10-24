
import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import logging as log
import pandas as pd
from output_reader import FileReader

log.basicConfig(
    level=log.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)



def scanning_rate(file_path: str):
    log.info(f"Reading tau data from: {file_path}\n")

    


if __name__ == "__main__":
    log.info("Scanning rate module started.")

    tau_data = read_tau_data("data/output_tau.txt")
    log.debug(f"Tau data read: {tau_data}")

    log.info("Scanning rate module finished.")


