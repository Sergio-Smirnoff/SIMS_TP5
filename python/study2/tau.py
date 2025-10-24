import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import logging as log
import pandas as pd
import numpy as np
from output_reader import FileReader

log.basicConfig(
    level=log.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def read_times_data(file_path: str):
    """
    Reads time data from the simulation output file
    Returns a dictionary with particle ids as keys and their corresponding time lists as values
    """
    reader = FileReader(file_path)
    times = reader.read_times()
    reader.close_file()

    times_data = times['t'].tolist()
    tau_data = np.array([])

    for index, time in enumerate(times_data):
        log.debug(f"Time {index}: {time}")
        if index+1 > len(times_data)-1:
            break
        tau = times_data[index+1] - time
        tau_data = np.append(tau_data, tau)

    return tau_data

if __name__ == "__main__":
    taus = read_times_data("data/output_tau.txt")
    log.info("Tau data reading module finished.")
    for tau in taus:
        log.info(f"Tau: {tau}")