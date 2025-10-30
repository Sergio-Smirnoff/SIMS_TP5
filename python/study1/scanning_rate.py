import datetime
import os
import sys
from pathlib import Path
import ast
from collections import defaultdict
from scipy import stats

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import logging as log
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from output_reader import FileReader

log.basicConfig(
    level=log.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# --- Directorios de Entrada y Salida ---
BASE_DATA_DIR = "data/"
OUTPUT_GRAPH_DIR = "output/study1/graphs/"
OUTPUT_FILE_DIR = "output/study1/data/"

def separate_files(sim_dir: str, times_dir: str):
    """
    
    """
    N_files = {}

    try:
        sim_files = [f for f in os.listdir(sim_dir) if f.startswith('simulation_') and f.endswith('.csv')]
    except FileNotFoundError:
        log.error(f"Directorio de simulación no encontrado: {sim_dir}")
        return {}
    except Exception as e:
        log.error(f"Error listando archivos en {sim_dir}: {e}")
        return {}

    log.info(f"Recorriendo los archivos de simulacion")
    for sim_file in sim_files:
        to_add = {
            'sim': os.path.join(sim_dir, sim_file),
            'times': os.path.join(times_dir, sim_file.replace('simulation_', 'times_'))
        }
        try:
            ## Para saber en que N ponerlo
            N = int(sim_file.split('_')[1].replace('N', ''))

            if N not in N_files:
                N_files[N] = []

            N_files[N].append(to_add)

        except Exception as e:
            log.error(f"Error procesando archivo {sim_file}: {e}")

    return N_files
        
def calculate_times(times_files, max_time=None, transient_fraction=0.3):
    """
    Calcula los tiempos promediados y acumulados de múltiples realizaciones.
    
    Args:
        times_files: Lista de archivos con tiempos de contacto
        max_time: Tiempo máximo común para interpolar (si None, usa el mínimo)
        transient_fraction: Fracción del tiempo total considerada como transitorio
    
    Returns:
        common_times: Array con tiempos comunes
        avg_accumulated: Array con contactos acumulados promediados
        std_accumulated: Desviación estándar de contactos acumulados
        Q: Scanning rate promedio
        Q_error: Error del scanning rate
    """
    all_accumulated_curves = []
    all_Q = []
    
    for times_file in times_files:
        try:
            times_reader = FileReader(times_file)
            times_df = times_reader.read_times()
            times_reader.close_file()
            
            if 't' not in times_df.columns:
                log.error(f"Columna 't' no encontrada en {times_file}. Columnas: {times_df.columns.tolist()}")
                continue
            
            contact_times = times_df['t'].values
            
            # Validaciones
            if len(contact_times) == 0:
                log.warning(f"Archivo vacío: {times_file}")
                continue
            
            if np.any(contact_times < 0):
                log.warning(f"Tiempos negativos en {times_file}")
                contact_times = contact_times[contact_times >= 0]
            
            if len(contact_times) < 2:
                log.warning(f"Muy pocos contactos en {times_file}: {len(contact_times)}")
                continue
            
            contact_times = np.sort(contact_times)
            
            n_contacts = np.arange(1, len(contact_times) + 1)
            
            log.info(f"Archivo {os.path.basename(times_file)}: {len(contact_times)} contactos, "
                    f"rango tiempo: [{contact_times[0]:.2f}, {contact_times[-1]:.2f}]s")
            
            all_accumulated_curves.append((contact_times, n_contacts))
            
        except Exception as e:
            log.error(f"Error leyendo {times_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(all_accumulated_curves) == 0:
        log.error("No se pudieron leer archivos de tiempos")
        return None, None, None, None, None
    
    if max_time is None:
        max_time = min([times[-1] for times, _ in all_accumulated_curves])
    
    log.info(f"Tiempo máximo común para interpolación: {max_time:.2f}s")
    
    if max_time <= 0:
        log.error(f"Tiempo máximo inválido: {max_time}")
        return None, None, None, None, None
    
    common_times = np.linspace(0, max_time, 1000)
    
    interpolated_curves = []
    for contact_times, n_contacts in all_accumulated_curves:

        mask = contact_times <= max_time
        contact_times_filtered = contact_times[mask]
        n_contacts_filtered = n_contacts[mask]
        
        if len(contact_times_filtered) < 2:
            log.warning("Muy pocos puntos después de filtrar por max_time")
            continue
        

        interpolated = np.interp(common_times, contact_times_filtered, n_contacts_filtered)
        interpolated_curves.append(interpolated)
    
    if len(interpolated_curves) == 0:
        log.error("No se pudieron interpolar curvas")
        return None, None, None, None, None
    

    interpolated_curves = np.array(interpolated_curves)
    

    avg_accumulated = np.mean(interpolated_curves, axis=0)
    std_accumulated = np.std(interpolated_curves, axis=0)
    

    transient_time = max_time * transient_fraction
    
    log.info(f"Calculando Q con régimen estacionario desde t={transient_time:.2f}s")
    
    for contact_times, n_contacts in all_accumulated_curves:

        mask = (contact_times > transient_time) & (contact_times <= max_time)
        
        if np.sum(mask) < 10:
            log.warning(f"Pocos puntos en régimen estacionario: {np.sum(mask)}")

            mask = contact_times > transient_time
            if np.sum(mask) < 5:
                log.warning("Aún muy pocos puntos, saltando esta realización")
                continue
        
        times_steady = contact_times[mask]
        contacts_steady = n_contacts[mask]
        

        slope, intercept, r_value, p_value, std_err = stats.linregress(
            times_steady, contacts_steady
        )
        
        log.info(f"Q = {slope:.4f} contactos/s, R² = {r_value**2:.4f}, std_err = {std_err:.4f}")

        if slope > 0:
            all_Q.append(slope)
        else:
            log.warning(f"Pendiente negativa o cero: {slope}")
    
    if len(all_Q) == 0:
        log.error("No se pudo calcular Q para ninguna realización")
        Q, Q_error = 0, 0
    else:
        Q = np.mean(all_Q)
        Q_error = np.std(all_Q)
        log.info(f"Q promedio: {Q:.4f} ± {Q_error:.4f} contactos/s (n={len(all_Q)} realizaciones)")
    
    return common_times, avg_accumulated, std_accumulated, Q, Q_error

def process_all_runs(base_dir: str) -> dict:
    """
    Lee todos los archivos de simulación, los agrupa por N y agrega los datos.
    """
    log.info("Procesando todas las corridas de simulación...")
    sim_dir = os.path.join(base_dir, "sim")
    times_dir = os.path.join(base_dir, "times")
    
    if not os.path.exists(sim_dir) or not os.path.exists(times_dir):
        log.error(f"Error: Los directorios '{sim_dir}' y/o '{times_dir}' no existen.")
        sys.exit(1)

    N_files = separate_files(sim_dir, times_dir)

    final_data = []

    # Ordenar por N
    for N in sorted(N_files.keys()):
        files = N_files[N]
        log.info(f"\n{'='*60}")
        log.info(f"Procesando N={N} con {len(files)} realizaciones")
        log.info(f"{'='*60}")
        
        json_data = {
            "N": N,
            "simulation_times": len(files),
            "times": [],
            "accumulated_contacts": [],
            "std_contacts": [],
            "scanning_rate": 0,
            "scanning_rate_error": 0,
            "avg_phi": 0,
            "error_phi": 0
        }
        
        # Calcular φ
        phis = []
        for file_dict in files:
            try:
                sim_reader = FileReader(file_dict['sim'])
                data = sim_reader.read_next_timestep()
                sim_reader.close_file()
                
                L = sim_reader.parameters['L']
                r = data['r'].to_numpy()
                phi = np.sum(np.pi * r**2) / (L**2)
                phis.append(phi)
            except Exception as e:
                log.error(f"Error procesando {file_dict['sim']}: {e}")
                continue

        if len(phis) > 0:
            json_data['avg_phi'] = np.mean(phis)
            json_data['error_phi'] = (max(phis) - min(phis)) / 2 if len(phis) > 1 else 0
            log.info(f"φ = {json_data['avg_phi']:.4f} ± {json_data['error_phi']:.4f}")
        else:
            log.warning(f"No se pudo calcular φ para N={N}")

        # Calcular tiempos y scanning rate
        times_files_list = [file_dict['times'] for file_dict in files]
        
        result = calculate_times(times_files_list)
        
        if result[0] is not None:
            common_times, avg_accumulated, std_accumulated, Q, Q_error = result
            
            json_data['times'] = common_times.tolist()
            json_data['accumulated_contacts'] = avg_accumulated.tolist()
            json_data['std_contacts'] = std_accumulated.tolist()
            json_data['scanning_rate'] = Q
            json_data['scanning_rate_error'] = Q_error
        else:
            log.error(f"No se pudieron calcular tiempos para N={N}")

        final_data.append(json_data)

    log.info("\n" + "="*60)
    log.info("Finalizando el análisis de datos de las simulaciones...")
    log.info("="*60)

    return final_data

def plot_accumulated_contacts(final_data):
    """Gráfico 1: Contactos acumulados promedio vs tiempo"""
    plt.figure(figsize=(12, 8))
    
    for data in final_data:
        N = data['N']
        phi = data['avg_phi']
        times = np.array(data['times'])
        accumulated = np.array(data['accumulated_contacts'])
        
        plt.plot(times, accumulated, label=f"N={N} (φ={phi:.3f})")
    
    plt.xlabel('Tiempo (s)', fontsize=12)
    plt.ylabel('Contactos acumulados promedio', fontsize=12)
    plt.title('Promedio de contactos acumulados por segundo (varios N)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_GRAPH_DIR + 'accumulated_contacts.png', dpi=300, bbox_inches='tight')
    #plt.show()

def plot_scanning_rate(final_data):
    """Gráfico 2: Q vs φ con barras de error"""
    plt.figure(figsize=(10, 6))
    
    phis = [data['avg_phi'] for data in final_data]
    Qs = [data['scanning_rate'] for data in final_data]
    Q_errors = [data['scanning_rate_error'] for data in final_data]
    Ns = [data['N'] for data in final_data]
    
    plt.errorbar(phis, Qs, yerr=Q_errors, fmt='o-', capsize=5, markersize=8)
    
    # Agregar etiquetas de N en cada punto
    for phi, Q, N in zip(phis, Qs, Ns):
        plt.text(phi, Q, f'N{N}', fontsize=8, ha='right', va='bottom')
    
    plt.xlabel('Fracción de área ocupada φ', fontsize=12)
    plt.ylabel('Scanning rate Q (contactos/s)', fontsize=12)
    plt.title('Q vs φ (promedio de 5 realizaciones)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_GRAPH_DIR + 'scanning_rate_vs_phi.png', dpi=300, bbox_inches='tight')
    #plt.show()

if __name__ == "__main__":
    log.info("Módulo de análisis de Scanning Rate iniciado.")

    os.makedirs(OUTPUT_GRAPH_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FILE_DIR, exist_ok=True)
    
    log.info("--- PASO 1: Procesando y agregando datos de todas las corridas ---")

    final_data = process_all_runs(BASE_DATA_DIR)
    plot_accumulated_contacts(final_data)
    plot_scanning_rate(final_data)

    log.info("Módulo de análisis finalizado.")