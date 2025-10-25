import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import logging as log
import pandas as pd
import numpy as np
import ast
from output_reader import FileReader

log.basicConfig(
    level=log.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

INPUT_FILE = "output/study1/data/phi_times_values.txt"
OUTPUT_FILE_DIR = "output/study2/data/"
OUTPUT_GRAPH_DIR = "output/study2/graphs/"


def read_times_data() -> pd.DataFrame:
    """
    Reads time data from the simulation output file
    Returns a dictionary with particle ids as keys and their corresponding time lists as values
    """
    log.info(f"Reading times data from: {INPUT_FILE}\n")

    df = pd.read_csv(INPUT_FILE, sep=';')

    try:
        df['times'] = df['times'].apply(ast.literal_eval)
        df['times_cumulative'] = df['times_cumulative'].apply(ast.literal_eval)
        log.debug("Columnas 'times' y 'times_cumulative' convertidas a listas correctamente.")
    except Exception as e:
        log.error(f"Error al convertir las columnas de string a lista: {e}")
        return
    values = []
    for index, row in df.iterrows():
        times_data = row['times'].tolist()
        tau_data = np.array([])

        for index, time in enumerate(times_data):
            log.debug(f"Time {index}: {time}")
            if index+1 > len(times_data)-1:
                break
            tau = times_data[index+1] - time
            tau_data = np.append(tau_data, tau)

        value = {"N": df['N'].iloc[0], "phi": df['phi'].iloc[0], "tau_values": tau_data}
        values.append(value)
    log.debug(f"Valores de tau leídos: {values}")
    df = pd.DataFrame(values, columns=["N", "phi", "tau_values"])
    output_file = OUTPUT_FILE_DIR + "tau_values.txt"    
    df.to_csv(output_file, sep=';', index=False)
    log.info(f"Valores de tau guardados en: {output_file}")

    return df

if __name__ == "__main__":
    taus = read_times_data("data/output_tau.txt")
    log.info("Tau data reading module finished.")
    for tau in taus:
        log.info(f"Tau: {tau}")
