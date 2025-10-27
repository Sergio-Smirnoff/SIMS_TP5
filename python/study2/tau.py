import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import logging as log
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast
import os
import powerlaw as pwl 

log.basicConfig(
    level=log.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

INPUT_FILE_DIR = "output/study1/data/"
OUTPUT_FILE_DIR = "output/study2/data/"
OUTPUT_GRAPH_DIR = "output/study2/graphs/"


def process_and_group_taus(input_dir: str) -> pd.DataFrame:
    """
    Lee todos los archivos de datos, calcula los intervalos tau para cada simulación,
    y luego agrupa todos los taus por el valor de 'N'.
    """
    log.info(f"Leyendo y procesando todos los archivos desde: {input_dir}")

    all_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv') or f.endswith('.txt')]
    
    if not all_files:
        log.error(f"No se encontraron archivos de datos en el directorio: {input_dir}")
        return None

    all_tau_data = []
    for file_path in all_files:
        log.info(f"Procesando archivo: {file_path}")
        try:
            df = pd.read_csv(file_path, sep=';')
            df['times'] = df['times'].apply(ast.literal_eval)
        except Exception as e:
            log.error(f"Error al leer o procesar el archivo {file_path}: {e}")
            continue

        for _, row in df.iterrows():
            times_data = np.array(row['times'])
            
            tau_values = np.diff(times_data)
            tau_values = tau_values[tau_values > 0]

            if tau_values.size > 0:
                all_tau_data.append({
                    "N": row['N'],
                    "phi": row['phi'],
                    "taus": tau_values
                })

    if not all_tau_data:
        log.warning("No se pudieron calcular datos de tau válidos de ningún archivo.")
        return None
    
    combined_df = pd.DataFrame(all_tau_data)
    log.info("Agrupando todos los valores de tau por N.")
    grouped_df = combined_df.groupby('N').agg(
        phi=('phi', 'mean'),
        all_taus=('taus', lambda x: np.concatenate(x.tolist()))
    ).reset_index()

    log.info("Agrupación completada con éxito.")
    log.info(f"\nResumen de datos agrupados:\n{grouped_df[['N', 'phi']].to_string(index=False)}")


    return grouped_df


def analyze_tau_distribution(grouped_tau_df: pd.DataFrame, output_graph_dir: str):
    """
    Analiza la distribución de los valores de tau para cada N y ajusta una ley de potencias.
    """
    if grouped_tau_df is None or grouped_tau_df.empty:
        log.error("El DataFrame de taus está vacío. No hay nada que analizar.")
        return

    alpha_results = []
    
    for _, row in grouped_tau_df.iterrows():
        N = row['N']
        phi = row['phi']
        tau_values = row['all_taus']

        if len(tau_values) < 50:
            log.warning(f"Para N={N}, hay solo {len(tau_values)} valores de tau. Saltando análisis por falta de datos.")
            continue
            
        log.info(f"Analizando distribución para N={N} (phi≈{phi:.4f}) con {len(tau_values)} puntos de datos.")
        
        fit = pwl.Fit(tau_values, verbose=False)
        
        alpha = fit.power_law.alpha
        sigma = fit.power_law.sigma 
        
        log.info(f"--> Resultado para N={N}: alpha = {alpha:.3f} ± {sigma:.3f}")
        
        alpha_results.append({'N': N, 'phi': phi, 'alpha': alpha, 'sigma': sigma})
        
        fig, ax = plt.subplots(figsize=(10, 7))
        fit.plot_pdf(ax=ax, color='b', linewidth=2, label='Datos Empíricos')
        fit.power_law.plot_pdf(ax=ax, color='r', linestyle='--', linewidth=2, label=f'Ajuste ($\\alpha$={alpha:.2f})')
        
        ax.set_xlabel('$\\tau$', fontsize=14)
        ax.set_ylabel('PDF', fontsize=14)
        ax.legend(fontsize='large')
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        
        output_path = os.path.join(output_graph_dir, f'distribucion_tau_N{N}.png')
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        log.info(f"Gráfico de distribución guardado en: {output_path}")


    if not alpha_results:
        log.warning("No se generaron resultados de alpha. No se puede crear el gráfico final.")
        return
    
    results_df = pd.DataFrame(alpha_results).sort_values(by='phi')
    
    log.info(f"\nResumen de exponentes calculados:\n{results_df.to_string(index=False)}")
    
    plt.figure(figsize=(12, 8))
    plt.errorbar(results_df['phi'], results_df['alpha'], yerr=results_df['sigma'], 
                 fmt='o-', capsize=5, markerfacecolor='royalblue', markeredgecolor='black', ecolor='darkgray', label='Exponente $\\alpha$ medido')
    
    plt.xlabel('$\\phi$', fontsize=14)
    plt.ylabel('$\\alpha$', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    final_graph_path = os.path.join(output_graph_dir, 'alpha_vs_phi.png')
    plt.savefig(final_graph_path, dpi=300)
    plt.close()
    log.info(f"Gráfico final 'alpha vs. phi' guardado en: {final_graph_path}")


if __name__ == "__main__":
    os.makedirs(OUTPUT_FILE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_GRAPH_DIR, exist_ok=True)
    
    log.info("--- PASO 1: Procesando y agrupando datos de tau ---")
    grouped_dataframe = process_and_group_taus(INPUT_FILE_DIR)
    
    if grouped_dataframe is not None:
        log.info("\n--- PASO 2: Analizando la distribución de tau y ajustando ley de potencias ---")
        analyze_tau_distribution(grouped_dataframe, OUTPUT_GRAPH_DIR)
        log.info("\nAnálisis completado.")
    else:
        log.error("No se generaron datos para analizar. El programa terminará.")