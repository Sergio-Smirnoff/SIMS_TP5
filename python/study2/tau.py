import sys
from pathlib import Path

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import powerlaw
from tabulate import tabulate
from output_reader import FileReader

# Configuración de matplotlib
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

def load_contact_times(data_dir, times_dir, file_reader_class):
    """
    Carga los tiempos de contactos únicos desde los archivos de datos.
    Usa el JSON ya procesado para obtener phi y N.
    Usa FileReader para leer los archivos tal como lo hace calculate_times.
    
    Args:
        data_dir: Directorio donde está scanning_rate_data.json
        times_dir: Directorio donde están los archivos times_N*_L*_TT*_*.csv
        file_reader_class: Clase FileReader para leer los archivos
    
    Returns:
        dict: Diccionario con N como clave y datos de configuración
    """
    import glob
    contact_times_by_N = {}
    
    # Leer el archivo scanning_rate_data.json (ya procesado)
    summary_file = os.path.join(data_dir, "scanning_rate_data.json")
    
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"No se encontró el archivo: {summary_file}")
    
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    
    print(f"   ✓ Cargado {summary_file}")
    print(f"   ✓ Encontradas {len(summary_data)} configuraciones")
    
    # Para cada configuración, cargar los tiempos de contacto
    for config in summary_data:
        N = config['N']
        phi = config['avg_phi']
        n_simulations = config['simulation_times']
        
        print(f"   - Cargando tiempos para N={N} (φ={phi:.4f})...", end="")
        
        # Buscar archivos de tiempos de contacto para este N
        # Formato: times_N{N}_L{L}_TT{TT}_{sim}.csv
        pattern = os.path.join(times_dir, f"times_N{N}_L*_TT*_*.csv")
        contact_files = sorted(glob.glob(pattern))
        
        if len(contact_files) == 0:
            print(f" ✗ No se encontraron archivos (patrón: times_N{N}_L*_TT*_*.csv)")
            continue
        
        all_times = []
        files_found = 0
        for times_file in contact_files:
            try:
                # Usar FileReader como en calculate_times
                times_reader = file_reader_class(times_file)
                times_df = times_reader.read_times()
                times_reader.close_file()
                
                if 't' not in times_df.columns:
                    print(f"\n      ⚠ Columna 't' no encontrada en {os.path.basename(times_file)}")
                    continue
                
                contact_times = times_df['t'].values
                
                # Validaciones (igual que en calculate_times)
                if len(contact_times) == 0:
                    continue
                
                if np.any(contact_times < 0):
                    contact_times = contact_times[contact_times >= 0]
                
                if len(contact_times) < 2:
                    continue
                
                contact_times = np.sort(contact_times)
                
                all_times.extend(contact_times.tolist())
                files_found += 1
                
            except Exception as e:
                print(f"\n      ⚠ Error leyendo {os.path.basename(times_file)}: {e}")
                continue
        
        if files_found == 0:
            print(f" ✗ No se pudieron leer archivos")
            continue
        
        # Calcular tiempos entre contactos (τ_i = t_{i+1} - t_i)
        if len(all_times) > 1:
            all_times = sorted(all_times)
            inter_contact_times = np.diff(all_times)
            # Filtrar valores muy pequeños o cero
            inter_contact_times = inter_contact_times[inter_contact_times > 1e-6]
            
            contact_times_by_N[N] = {
                'phi': phi,
                'inter_contact_times': inter_contact_times,
                'n_contacts': len(all_times),
                'n_inter_contacts': len(inter_contact_times),
                'n_simulations': files_found
            }
            print(f" ✓ {files_found} archivos, {len(all_times)} contactos, {len(inter_contact_times)} τ")
        else:
            print(f" ✗ Datos insuficientes ({len(all_times)} contactos)")
    
    return contact_times_by_N

def goodness_of_fit(fit, n_synthetic=2500):
    data = fit.data
    xmin = fit.xmin
    alpha = fit.alpha
    D_emp = fit.D  # KS empírico

    # Partir datos en cola y parte inferior
    data_below = data[data < xmin]
    n_total = len(data)
    n_tail = len(data[data >= xmin])

    D_synth_list = []
    for _ in range(n_synthetic):
        # Probabilidad de tomar de la cola
        p_tail = n_tail / n_total
        synthetic = []
        for _ in range(n_total):
            if np.random.rand() < p_tail:
                # Genera dato sintetico de una power-law con alpha y xmin
                r = np.random.random()
                x = xmin * (1 - r) ** (-1 / (alpha - 1))
                synthetic.append(x)
            else:
                # Tomar un valor al azar de la parte inferior real
                synthetic.append(np.random.choice(data_below))
        synthetic = np.array(synthetic)

        # Ajustar power-law al sintético
        fit_synth = powerlaw.Fit(synthetic, xmin=xmin, verbose=False)
        D_synth_list.append(fit_synth.D)

    D_synth_list = np.array(D_synth_list)
    p_value = np.mean(D_synth_list > D_emp)
    return p_value

def analyze_power_law_clauset_method(data, n_bootstrap: int = 1000):
    """
    Analiza si los datos siguen una ley de potencias siguiendo EXACTAMENTE
    el método de Clauset et al. 2009 (Box 1, página 663):
    
    1. Estimar x_min y α usando maximum likelihood
    2. Calcular goodness-of-fit con p-value (bootstrap KS)
    3. Comparar con distribuciones alternativas usando likelihood ratio
    
    Args:
        data (array): tiempos entre contactos
        n_bootstrap (int): cantidad de simulaciones bootstrap (1000 recomendado)
    
    Returns:
        dict: Resultados completos del análisis
    """

    if len(data) < 50:
        return None

    # === PASO 1: Estimar parámetros ===
    fit = powerlaw.Fit(data, discrete=False, xmin=None)
    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin
    sigma = fit.power_law.sigma
    D = fit.power_law.D
    n_tail = np.sum(data >= xmin)

    p_boot = goodness_of_fit(fit)

    passes_gof = (p_boot >= 0.1)

    # === PASO 3: Comparaciones con distribuciones alternativas ===
    R_exp, p_exp = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
    R_log, p_log = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)
    _, p_value = fit.distribution_compare('power_law', 'power_law', normalized_ratio=True)
    R_stretch, p_stretch = fit.distribution_compare('power_law', 'stretched_exponential', normalized_ratio=True)
    R_trunc, p_trunc = fit.distribution_compare('power_law', 'truncated_power_law', normalized_ratio=True)

    # === Clasificación final ===
    better_than_exp = (p_exp < 0.05 and R_exp > 0)
    better_than_log = (p_log < 0.05 and R_log > 0)
    worse_than_log = (p_log < 0.05 and R_log < 0)
    needs_cutoff = (p_trunc < 0.05 and R_trunc < 0)

    if not passes_gof:
        classification = "REJECTED"
        support_level = "none"
    elif not better_than_exp:
        classification = "NOT_HEAVY_TAILED"
        support_level = "none"
    elif needs_cutoff:
        classification = "POWER_LAW_WITH_CUTOFF"
        support_level = "moderate"
    elif worse_than_log:
        classification = "LOGNORMAL_BETTER"
        support_level = "none"
    elif better_than_log:
        classification = "POWER_LAW_CONFIRMED"
        support_level = "good"
    else:
        classification = "POWER_LAW_PLAUSIBLE"
        support_level = "moderate"

    # === Empaquetar resultados ===
    results = {
        'alpha': alpha,
        'sigma': sigma,
        'xmin': xmin,
        'D': D,
        'p_boot': p_value,              
        'n_data': len(data),
        'n_tail': n_tail,
        'passes_gof': passes_gof,
        'R_exp': R_exp,
        'p_exp': p_exp,
        'better_than_exp': better_than_exp,
        'R_log': R_log,
        'p_log': p_log,
        'better_than_log': better_than_log,
        'worse_than_log': worse_than_log,
        'R_stretch': R_stretch,
        'p_stretch': p_stretch,
        'R_trunc': R_trunc,
        'p_trunc': p_trunc,
        'needs_cutoff': needs_cutoff,
        'classification': classification,
        'support_level': support_level,
        'fit': fit,
        'data': data
    }

    return results


def create_results_table(contact_times_by_N):
    """
    Crea la tabla de resultados siguiendo el método de Clauset et al. 2009.
    """
    results = []
    
    sorted_N = sorted(contact_times_by_N.keys())
    
    print("\n" + "="*100)
    print("ANÁLISIS DE LEY DE POTENCIAS - MÉTODO DE CLAUSET ET AL. 2009")
    print("="*100)
    print("\nAnalizando cada configuración...")
    
    for N in sorted_N:
        config = contact_times_by_N[N]
        phi = config['phi']
        data = config['inter_contact_times']
        
        print(f"\nN = {N}, φ = {phi:.4f}, n_τ = {len(data)}", end="")
        
        if len(data) < 50:
            print(" -> Datos insuficientes (n < 50), omitiendo...")
            continue
        
        analysis = analyze_power_law_clauset_method(data)
        
        if analysis is None:
            print(" -> Error en análisis, omitiendo...")
            continue
        
        print(f" -> {analysis['classification']}")
        
        result = {
            'N': N,
            'phi': phi,
            'alpha': analysis['alpha'],
            'sigma': analysis['sigma'],
            'xmin': analysis['xmin'],
            'KS_D': analysis['D'],
            'n_data': analysis['n_data'],
            'n_tail': analysis['n_tail'],
            
            # Tests
            'passes_gof': analysis['passes_gof'],
            'better_than_exp': analysis['better_than_exp'],
            'better_than_log': analysis['better_than_log'],
            'needs_cutoff': analysis['needs_cutoff'],
            
            # Valores para tabla
            'R_exp': analysis['R_exp'],
            'p_exp': analysis['p_exp'],
            'R_log': analysis['R_log'],
            'p_log': analysis['p_log'],
            'R_stretch': analysis['R_stretch'],
            'p_stretch': analysis['p_stretch'],
            
            # Clasificación
            'classification': analysis['classification'],
            'support_level': analysis['support_level'],
            'fit': analysis['fit']
        }
        
        results.append(result)
    
    return results

def print_results_table(results, output_dir="output/study2"):
    """
    Genera una tabla de resultados usando tabulate (incluye p_boot) y la guarda en un archivo.
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "data/results_table.txt")

    # --- Cabeceras de la tabla ---
    headers = [
        "N", "φ", "α", "σ(α)", "x_min", "KS(D)",
        "p_boot", "p_exp", "p_log", "R_exp", "R_log",
        "n_tail", "Clasificación"
    ]

    table_data = []
    for r in results:
        # Algunos resultados antiguos pueden no tener p_boot si se generaron antes
        p_boot = r.get("p_boot", float("nan"))

        table_data.append([
            r["N"],
            f"{r['phi']:.4f}",
            f"{r['alpha']:.3f}",
            f"{r['sigma']:.3f}",
            f"{r['xmin']:.3f}",
            f"{r['KS_D']:.4f}",
            f"{p_boot:.3f}",
            f"{r['p_exp']:.4f}",
            f"{r['p_log']:.4f}",
            f"{r['R_exp']:+.2f}",
            f"{r['R_log']:+.2f}",
            r["n_tail"],
            r["classification"]
        ])

    # --- Crear tabla con tabulate ---
    table_str = tabulate(table_data, headers=headers, tablefmt="grid", stralign="center")

    # --- Imprimir y guardar ---
    print("\n" + "="*120)
    print("TABLA DE RESULTADOS - MÉTODO DE CLAUSET ET AL. 2009 (con p_boot)")
    print("="*120)
    print(table_str)
    print("\nGuardando tabla en:", output_path)

    with open(output_path, "w") as f:
        f.write("TABLA DE RESULTADOS - MÉTODO DE CLAUSET ET AL. 2009 (con p_boot)\n\n")
        f.write(table_str)
        f.write("\n")

    print(f"✓ Tabla guardada correctamente en {output_path}")


def plot_alpha_vs_phi(results, output_dir="output/study1/figures"):
    """
    Gráfico de α vs φ, coloreado según clasificación de Clauset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Separar por clasificación
    confirmed = [r for r in results if 'CONFIRMED' in r['classification']]
    plausible = [r for r in results if 'PLAUSIBLE' in r['classification']]
    cutoff = [r for r in results if 'CUTOFF' in r['classification']]
    rejected = [r for r in results if 'REJECTED' in r['classification'] or 
                'NOT_HEAVY' in r['classification'] or 'LOGNORMAL' in r['classification']]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot cada categoría con color diferente
    if confirmed:
        phi_c = [r['phi'] for r in confirmed]
        alpha_c = [r['alpha'] for r in confirmed]
        sigma_c = [r['sigma'] for r in confirmed]
        ax.errorbar(phi_c, alpha_c, yerr=sigma_c, fmt='o', capsize=5, capthick=2,
                   markersize=10, color='green', ecolor='darkgreen', linewidth=2,
                   label='Power Law Confirmada', markeredgewidth=2)
    
    if plausible:
        phi_p = [r['phi'] for r in plausible]
        alpha_p = [r['alpha'] for r in plausible]
        sigma_p = [r['sigma'] for r in plausible]
        ax.errorbar(phi_p, alpha_p, yerr=sigma_p, fmt='s', capsize=5, capthick=2,
                   markersize=10, color='orange', ecolor='darkorange', linewidth=2,
                   label='Power Law Plausible', markeredgewidth=2)
    
    if cutoff:
        phi_t = [r['phi'] for r in cutoff]
        alpha_t = [r['alpha'] for r in cutoff]
        sigma_t = [r['sigma'] for r in cutoff]
        ax.errorbar(phi_t, alpha_t, yerr=sigma_t, fmt='^', capsize=5, capthick=2,
                   markersize=10, color='blue', ecolor='darkblue', linewidth=2,
                   label='Power Law con Cutoff', markeredgewidth=2)
    
    if rejected:
        phi_r = [r['phi'] for r in rejected]
        alpha_r = [r['alpha'] for r in rejected]
        sigma_r = [r['sigma'] for r in rejected]
        ax.errorbar(phi_r, alpha_r, yerr=sigma_r, fmt='x', capsize=5, capthick=2,
                   markersize=12, color='red', ecolor='darkred', linewidth=3,
                   label='NO Power Law', markeredgewidth=3)
    
    ax.axhline(y=2, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='α = 2')
    ax.axhline(y=3, color='gray', linestyle=':', alpha=0.5, linewidth=1.5, label='α = 3')
    
    ax.set_xlabel('Fracción de área ocupada (φ)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Exponente α', fontsize=13, fontweight='bold')
    ax.set_title('Análisis de Ley de Potencias - Método de Clauset et al. 2009\n' +
                'Tiempos entre Contactos vs Densidad', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'alpha_vs_phi_clauset.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'alpha_vs_phi_clauset.pdf'), bbox_inches='tight')
    print(f"✓ Gráfico guardado: alpha_vs_phi_clauset.png (y .pdf)")
    plt.close()

def plot_distributions_with_alternatives(results, output_dir="output/study2/figures"):
    """
    Grafica las distribuciones CCDF comparando con la distribución más apropiada
    según la clasificación de Clauset (power law, lognormal, exponencial, etc.).
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    n_examples = min(6, len(results))
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    example_indices = np.linspace(0, len(results) - 1, n_examples, dtype=int)

    for idx, result_idx in enumerate(example_indices):
        r = results[result_idx]
        ax = axes[idx]

        fit = r["fit"]

        # Color y tipo de comparación según clasificación
        if "CONFIRMED" in r["classification"] or "PLAUSIBLE" in r["classification"]:
            color_data, color_fit = "green", "darkgreen"
            fit_label = f"Power Law α={r['alpha']:.2f}"
            fit.power_law.plot_ccdf(ax=ax, color=color_fit, linestyle="--", linewidth=2.5, label=fit_label)

        elif "CUTOFF" in r["classification"]:
            color_data, color_fit = "blue", "darkblue"
            fit_label = f"Truncated PL α={r['alpha']:.2f}"
            fit.truncated_power_law.plot_ccdf(ax=ax, color=color_fit, linestyle="--", linewidth=2.5, label=fit_label)

        elif "LOGNORMAL" in r["classification"]:
            color_data, color_fit = "orange", "darkorange"
            fit_label = "Lognormal"
            fit.lognormal.plot_ccdf(ax=ax, color=color_fit, linestyle="--", linewidth=2.5, label=fit_label)

        elif "NOT_HEAVY" in r["classification"]:
            color_data, color_fit = "red", "darkred"
            fit_label = "Exponential"
            fit.exponential.plot_ccdf(ax=ax, color=color_fit, linestyle="--", linewidth=2.5, label=fit_label)

        else:
            color_data, color_fit = "gray", "black"
            fit_label = "Power Law (default)"
            fit.power_law.plot_ccdf(ax=ax, color=color_fit, linestyle="--", linewidth=2.5, label=fit_label)

        # Graficar los datos reales
        fit.plot_ccdf(ax=ax, color=color_data, linewidth=2.5, label="Datos", marker="o", markersize=4)

        # Título con resumen
        title = f"N={r['N']}, φ={r['phi']:.4f}\n"
        title += f"{r['classification'].replace('_',' ')} | n={r['n_tail']}/{r['n_data']}"
        ax.set_title(title, fontsize=9, fontweight="bold")

        ax.legend(loc="best", fontsize=8)
        ax.set_xlabel("τ [s]", fontsize=9)
        ax.set_ylabel("P(X ≥ τ)", fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)

    for idx in range(n_examples, 6):
        axes[idx].axis("off")

    plt.suptitle("Comparación de distribuciones según clasificación de Clauset et al. 2009", 
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distributions_comparative.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(output_dir, "distributions_comparative.pdf"), bbox_inches="tight")
    print(f"✓ Gráfico guardado: distributions_comparative.png (y .pdf)")
    plt.close()


def plot_all_distributions_grid(results, output_dir="output/study1/figures"):
    """
    Grid completo con TODAS las configuraciones.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    n_total = len(results)
    n_cols = 4
    n_rows = (n_total + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()
    
    for idx, r in enumerate(results):
        ax = axes[idx]
        
        # Color según clasificación
        if 'CONFIRMED' in r['classification']:
            color_fit = 'green'
        elif 'PLAUSIBLE' in r['classification']:
            color_fit = 'orange'
        elif 'CUTOFF' in r['classification']:
            color_fit = 'blue'
        else:
            color_fit = 'red'
        
        # Datos
        r['fit'].plot_ccdf(ax=ax, color='gray', linewidth=2, 
                          label='Datos', marker='o', markersize=3, alpha=0.6)
        
        # Power Law
        r['fit'].power_law.plot_ccdf(ax=ax, color=color_fit, linestyle='-', 
                                     linewidth=2.5, label=f'PL (α={r["alpha"]:.2f})')
        
        # Exponencial (si es relevante)
        if r['p_exp'] < 0.1:
            try:
                r['fit'].exponential.plot_ccdf(ax=ax, color='blue', linestyle='--', 
                                              linewidth=2, label='Exp', alpha=0.7)
            except:
                pass
        
        # Lognormal (si es relevante)
        if r['p_log'] < 0.1:
            try:
                r['fit'].lognormal.plot_ccdf(ax=ax, color='purple', linestyle='-.', 
                                            linewidth=2, label='LogN', alpha=0.7)
            except:
                pass
        
        title = f'N={r["N"]}, φ={r["phi"]:.3f}\n'
        title += f'α={r["alpha"]:.2f}, D={r["KS_D"]:.3f}'
        
        ax.set_title(title, fontsize=8)
        ax.legend(loc='best', fontsize=7)
        ax.set_xlabel('τ [s]', fontsize=7)
        ax.set_ylabel('P(X ≥ τ)', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.tick_params(labelsize=7)
    
    # Ocultar ejes sobrantes
    for idx in range(n_total, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle('Todas las Configuraciones - Análisis Power Law', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_distributions_grid.png'), 
                dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: all_distributions_grid.png")
    plt.close()

# --- SCRIPT PRINCIPAL ---
if __name__ == "__main__":
    print("\n" + "="*100)
    print("ANÁLISIS DE LEY DE POTENCIAS - MÉTODO DE CLAUSET ET AL. 2009")
    print("Box 1 (página 663): Estimación de parámetros + Goodness-of-fit + Comparaciones")
    print("="*100)
    
    base_dir = "output/study1/"
    data_dir = os.path.join(base_dir, "data")      # Donde está scanning_rate_data.json
    times_dir = os.path.join("data/times")    # Donde están los contact_times_*.txt
    output_dir = os.path.join("output/study2/graphs")

    try:
        # 1. Cargar datos
        print(f"\n1. Cargando datos...")
        print(f"   JSON: {data_dir}/scanning_rate_data.json")
        print(f"   Times: {times_dir}/times_N*_L*_TT*_*.csv")
        contact_times_by_N = load_contact_times(data_dir, times_dir, FileReader)
        print(f"\n   ✓ Configuraciones cargadas: {len(contact_times_by_N)}")
        
        if len(contact_times_by_N) == 0:
            print("\n✗ No hay datos para analizar")
            exit(1)
        
        # 2. Análisis con método de Clauset
        print("\n2. Aplicando método de Clauset et al. 2009...")
        results = create_results_table(contact_times_by_N)
        
        if len(results) == 0:
            print("\n✗ No se generaron resultados")
            exit(1)
        
        # 3. Imprimir tabla de resultados
        print("\n3. Generando reporte de resultados...")
        print_results_table(results)
        
        # 4. Generar gráficos
        print("\n4. Generando gráficos...")
        plot_alpha_vs_phi(results, output_dir)
        plot_distributions_with_alternatives(results, output_dir)
        #plot_all_distributions_grid(results, output_dir)

        print("\n" + "="*100)
        print("✓ ANÁLISIS COMPLETADO")
        print(f"   Archivos guardados en: {output_dir}")
        print("="*100 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()