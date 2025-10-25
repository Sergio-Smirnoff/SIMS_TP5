
import os
import sys
from pathlib import Path
import ast

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import logging as log
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from output_reader import FileReader

log.basicConfig(
    level=log.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

OUTPUT_GRAPH_DIR = "output/study1/graphs/"

OUTPUT_FILE_DIR = "output/study1/data/"

def calculate_phi(data: pd.DataFrame, L: float, N: int) -> float:
    """
    Calculate the phi value from the given DataFrame.
    phi = (sum of all particle areas) / (total area of the box)
    """
    value = {'N': N, 'phi': 0.0}
    total_area = L * L
    particle_areas = data['r'] ** 2 * np.pi
    phi = particle_areas.sum() / total_area
    value['phi'] = phi
    return value



def calculate_phi_times(file_path: str):
    """
    Suposicion: Todos los archivos dentro de las carpetas sim y time tienen el mismo nombre salvo por N
    por lo cual, puedo suponer que estan en ese orden. Luego cuando tenga las 2 listas emparejo los valores
    bajo esa suposicion.
    """

    log.info(f"Reading tau data from: {file_path}\n")
    sim_files = os.listdir(file_path+"sim")
    time_files = os.listdir(file_path+"times")

    # calculo de phi
    phi_values = []
    out_file = open(OUTPUT_FILE_DIR + "phi_times_values.txt", "w")

    for file in sim_files:
        if file.endswith(".txt"):
            log.info(f"Found tau file: {file}")
            reader = FileReader(os.path.join(file_path, "sim", file))
            N = reader.parameters["N"]
            L = reader.parameters["L"]
            log.info(f"Number of particles: {N}")
            log.info(f"Box size: {L}x{L}\n")
            df = reader.read_next_timestep() # Read first timestep only
            reader.close_file()
            if df is not None:
                phi_value = calculate_phi(df, L, N)
                phi_values.append(phi_value)
                log.info(f"Calculated phi: {phi_value['phi']}\n")

    # Tiempos de contacto
    times_values = []

    for file in time_files:
        if file.endswith(".txt"):
            log.info(f"Found time file: {file}\n")
            reader = FileReader(os.path.join(file_path, "times", file))
            N = reader.parameters["N"]
            L = reader.parameters["L"]
            times = reader.read_times()
            times_cumulative = np.array(range(1,len(times)+1))
            reader.close_file()
            log.info(f"Number of times: {len(times)}\n")
            times_values.append({"times": times, "times_cumulative": times_cumulative})

    rows = []

    # Emparejamiento de valores
    for i in range(len(phi_values)):
        N = phi_values[i]['N']
        phi = phi_values[i]['phi']
        times = times_values[i]['times']
        times_cumulative = times_values[i]['times_cumulative']

        log.debug(f"N: {N}, phi: {phi}, times: {times['t'].tolist()}, times_cumulative: {times_cumulative}\n")

        headers = ["N", "phi", "times", "times_cumulative"]
        row = {
            "N": N,
            "phi": phi,
            "times": times['t'].tolist(),
            "times_cumulative": times_cumulative.tolist()
        }
        rows.append(row)

    df = pd.DataFrame(rows, columns=headers)
    df.to_csv(out_file, sep=';', index=False, header=headers)

    out_file.close()

def graph_cumulative_times():
    """
    Graph cumulative times from the output file.
    """
    log.info(f"Graphing cumulative times from: {OUTPUT_FILE_DIR + "phi_times_values.txt"}\n")
    df = pd.read_csv(OUTPUT_FILE_DIR + "phi_times_values.txt", sep=';')

    try:
        df['times'] = df['times'].apply(ast.literal_eval)
        df['times_cumulative'] = df['times_cumulative'].apply(ast.literal_eval)
        log.debug("Columnas 'times' y 'times_cumulative' convertidas a listas correctamente.")
    except Exception as e:
        log.error(f"Error al convertir las columnas de string a lista: {e}")
        return

    for index, row in df.iterrows():
        N = row['N']
        phi = row['phi']
        times_cumulative = row['times_cumulative']
        times = row['times']

        log.debug(f"Times:\n{times}\n")
        log.debug(f"times[0]: {times[0]}, times[-1]: {times[-1]}\n")
        log.debug(f"Cumulative Times:\n{times_cumulative}\n")

        plt.figure()
        plt.plot(times, times_cumulative, marker='o', label=f'N={N}, phi={phi:.4f}')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Contacts')
        plt.title(f'Cumulative Contacts vs Time')
        plt.legend()
        plt.grid()
        output_path = os.path.join(OUTPUT_GRAPH_DIR, f'cumulative_times_N{N}_phi{phi:.4f}.png')
        plt.savefig(output_path)
        log.info(f"Saved graph to: {output_path}\n")
        plt.close()

def calculate_scanning_rate(file_path: str, output_file: str, from_time: float = 0.0) -> pd.DataFrame:
    """
    Calculate scanning rate from the output file.
    """
    log.info(f"Calculating scanning rate from: {file_path}\n")
    df = pd.read_csv(file_path, sep=';')

    try:
        df['times'] = df['times'].apply(ast.literal_eval)
        df['times_cumulative'] = df['times_cumulative'].apply(ast.literal_eval)
        log.debug("Columnas 'times' y 'times_cumulative' convertidas a listas correctamente.")
    except Exception as e:
        log.error(f"Error al convertir las columnas de string a lista: {e}")
        return

    scanning_rates = []

    for index, row in df.iterrows():
        N = row['N']
        phi = row['phi']
        times_cumulative = row['times_cumulative'][from_time:]
        times = row['times'][from_time:]

        fit = np.polyfit(times, times_cumulative, 1)

        scanning_rate = fit[0]
        scanning_rates.append({'N': N, 'phi': phi, 'scanning_rate': scanning_rate})
        log.info(f"N={N}, phi={phi:.4f}, Scanning Rate={scanning_rate:.4f}\n")

    out_df = pd.DataFrame(scanning_rates)
    out_df.to_csv(output_file, sep=';', index=False)
    log.info(f"Saved scanning rates to: {output_file}\n")
    return out_df

def graph_scanning_rate(df: pd.DataFrame, output_path: str):
    """
    Graph scanning rate from the DataFrame.
    """
    log.info("Graphing scanning rate.\n")

    plt.figure()
    for N in df['N'].unique():
        subset = df[df['N'] == N]
        plt.plot(subset['phi'], subset['scanning_rate'], marker='o', label=f'N={N}')

    plt.xlabel('Phi')
    plt.ylabel('Scanning Rate')
    plt.title('Scanning Rate vs Phi')
    plt.legend()
    plt.grid()
    plt.savefig(output_path)
    log.info(f"Saved scanning rate graph to: {output_path}\n")
    plt.close()

if __name__ == "__main__":
    log.info("Scanning rate module started.")

    if not os.path.exists("data/sim") or not os.path.exists("data/times"):
        log.error("Error: Los directorios 'data/sim' y/o 'data/times' no existen.")
        sys.exit(1) 
    
    if len(sys.argv) < 2:
        log.error("Error: No se especificó una acción. Use --cumulativegraph o --scanningrate.")
        sys.exit(1)

    log.info("Calculando valores de phi y tiempos...")
    calculate_phi_times("data/")

    action = sys.argv[1]
    
    if action == '--cumulativegraph':
        log.info("Acción seleccionada: Graficar tiempos acumulados.")
        graph_cumulative_times()

    elif action == '--scanningrate':
        log.info("Acción seleccionada: Calcular y graficar scanning rate.")
        df = calculate_scanning_rate(OUTPUT_FILE_DIR + "phi_times_values.txt", OUTPUT_FILE_DIR + "scanning_rates.txt", from_time=5)
        log.debug(f"Scanning rates calculados:\n{df}\n")
        graph_scanning_rate(df, OUTPUT_GRAPH_DIR + "scanning_rate_graph.png")
    
    else:
        log.error(f"Error: Acción desconocida '{action}'. Use --cumulativegraph o --scanningrate.")
        sys.exit(1)

    log.info("Scanning rate module finished.")


