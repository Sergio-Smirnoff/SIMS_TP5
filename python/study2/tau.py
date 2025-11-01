import sys
from pathlib import Path

import json
import os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import powerlaw

root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from output_reader import FileReader

# Configuración
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize']=16
plt.rcParams['xtick.labelsize']=16
plt.rcParams['ytick.labelsize']=16

def load_contact_times(data_dir, times_dir):
    """
    Carga los tiempos de contactos y calcula tiempos entre contactos.
    """
    contact_times_by_N = {}
    
    summary_file = os.path.join(data_dir, "analysis_results.json")
    
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"No se encontró: {summary_file}")
    
    with open(summary_file, 'r') as f:
        summary_data = json.load(f)
    
    print(f"✓ Cargado {summary_file}")
    print(f"✓ Encontradas {len(summary_data)} configuraciones\n")

    for config in summary_data:
        N = config['N']
        phi = config['avg_phi']
        
        print(f"Cargando N={N} (φ={phi:.4f})...", end="")
        
        files = os.listdir(times_dir)
        pattern = f"times_N{N}_L*_TT*_run*.csv"
        contact_files = [os.path.join(times_dir, f) for f in files if fnmatch.fnmatch(f, pattern)]

        if len(contact_files) == 0:
            print(f" ✗ Sin archivos")
            continue
        
        all_times = []
        for times_file in contact_files:
            try:
                times_reader = FileReader(times_file)
                times_df = times_reader.read_times()
                times_reader.close_file()
                
                if 't' not in times_df.columns:
                    continue
                
                contact_times = times_df['t'].values
                
                if len(contact_times) < 2:
                    continue
                
                contact_times = contact_times[contact_times >= 0]
                contact_times = np.sort(contact_times)
                
                all_times.extend(contact_times.tolist())
                
            except Exception as e:
                print(f"\n   ⚠ Error en {os.path.basename(times_file)}: {e}")
                continue
        
        if len(all_times) > 1:
            all_times = sorted(all_times)
            inter_contact_times = np.diff(all_times)
            inter_contact_times = inter_contact_times[inter_contact_times > 1e-6]
            
            contact_times_by_N[N] = {
                'phi': phi,
                'inter_contact_times': inter_contact_times,
                'n_contacts': len(all_times),
            }
            print(f" ✓ {len(contact_files)} archivos, {len(inter_contact_times)} τ")
        else:
            print(f" ✗ Datos insuficientes")
    
    return contact_times_by_N

def analyze_distribution(data, N, phi):
    """
    Determina qué distribución se ajusta mejor: Power Law, Exponencial o Lognormal.
    """
    if len(data) < 50:
        return None
    
    fit = powerlaw.Fit(data, discrete=False, xmin=None)
    
    # Comparar con exponencial
    R_exp, p_exp = fit.distribution_compare('power_law', 'exponential', 
                                             normalized_ratio=True)
    
    # Comparar con lognormal
    R_log, p_log = fit.distribution_compare('power_law', 'lognormal',
                                             normalized_ratio=True)
    
    # Determinar mejor distribución (umbral p < 0.05 para significancia)
    if p_exp > 0.05 and R_exp < 0:
        # Exponencial es mejor que power law
        best_dist = 'exponential'
        best_color = 'blue'
    elif p_log > 0.05 and R_log < 0:
        # Lognormal es mejor que power law
        best_dist = 'lognormal'
        best_color = 'green'
    else:
        # Power law es la mejor (o no se puede distinguir)
        best_dist = 'power_law'
        best_color = 'red'
    
    result = {
        'N': N,
        'phi': phi,
        'best_dist': best_dist,
        'best_color': best_color,
        
        # Power Law
        'alpha': fit.power_law.alpha,
        'alpha_sigma': fit.power_law.sigma,
        'xmin': fit.power_law.xmin,
        
        # Exponencial
        'lambda': fit.exponential.parameter1,
        
        # Lognormal
        'mu': fit.lognormal.mu,
        'lognormal_sigma': fit.lognormal.sigma,
        
        # Comparaciones
        'R_exp': R_exp,
        'p_exp': p_exp,
        'R_log': R_log,
        'p_log': p_log,
        
        # Datos
        'n_data': len(data),
        'n_tail': np.sum(data >= fit.power_law.xmin),
        'fit': fit,
        'data': data
    }
    
    return result

def plot_individual_distributions(results, output_dir):
    """
    Genera un gráfico individual para cada N mostrando su mejor ajuste.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for r in results:
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Datos empíricos
        r['fit'].plot_ccdf(ax=ax, color='black', linewidth=2.5, 
                          label='Datos empíricos', marker='o', markersize=5, alpha=0.7)
        
        # Graficar las tres distribuciones
        r['fit'].power_law.plot_ccdf(ax=ax, color='red', linestyle='--', 
                                     linewidth=2, label=f'Power Law (α={r["alpha"]:.2f})', alpha=0.7)
        
        r['fit'].exponential.plot_ccdf(ax=ax, color='blue', linestyle='--', 
                                       linewidth=2, label=f'Exponencial (λ={r["lambda"]:.3f})', alpha=0.7)
        
        r['fit'].lognormal.plot_ccdf(ax=ax, color='green', linestyle='--', 
                                     linewidth=2, label=f'Lognormal (μ={r["mu"]:.2f})', alpha=0.7)
        
        # Resaltar la mejor distribución
        if r['best_dist'] == 'power_law':
            r['fit'].power_law.plot_ccdf(ax=ax, color=r['best_color'], linestyle='-', 
                                         linewidth=3.5, label='★ MEJOR AJUSTE: Power Law', zorder=10)
            best_info = f'α = {r["alpha"]:.3f} ± {r["alpha_sigma"]:.3f}'
        elif r['best_dist'] == 'exponential':
            r['fit'].exponential.plot_ccdf(ax=ax, color=r['best_color'], linestyle='-', 
                                           linewidth=3.5, label='★ MEJOR AJUSTE: Exponencial', zorder=10)
            best_info = f'λ = {r["lambda"]:.4f}'
        else:  # lognormal
            r['fit'].lognormal.plot_ccdf(ax=ax, color=r['best_color'], linestyle='-', 
                                         linewidth=3.5, label='★ MEJOR AJUSTE: Lognormal', zorder=10)
            best_info = f'μ = {r["mu"]:.3f}, σ = {r["lognormal_sigma"]:.3f}'
        
        # Título con información
        title = f'N = {r["N"]}, φ = {r["phi"]:.4f}\n'
        title += f'{best_info}\n'
        title += f'x_min = {r["xmin"]:.3f}, n_tail = {r["n_tail"]} / {r["n_data"]}'
        
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=10, framealpha=0.95)
        ax.set_xlabel('τ (tiempo entre contactos) [s]', fontsize=12, fontweight='bold')
        ax.set_ylabel('P(X ≥ τ)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xscale('log')
        ax.set_yscale('log')
        
        plt.tight_layout()
        filename = f'distribution_N{r["N"]}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Guardado: {filename}")

def plot_parameters_vs_phi(results, output_dir):
    """
    Genera gráficos de parámetros vs phi, separados por tipo de distribución.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Separar por tipo de distribución
    powerlaw_results = [r for r in results if r['best_dist'] == 'power_law']
    exp_results = [r for r in results if r['best_dist'] == 'exponential']
    log_results = [r for r in results if r['best_dist'] == 'lognormal']
    
    # ========== GRÁFICO 1: ALPHA vs PHI (solo power laws) ==========
    if len(powerlaw_results) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        phi_vals = [r['phi'] for r in powerlaw_results]
        alpha_vals = [r['alpha'] for r in powerlaw_results]
        sigma_vals = [r['alpha_sigma'] for r in powerlaw_results]
        
        ax.errorbar(phi_vals, alpha_vals, yerr=sigma_vals, 
                   fmt='o', capsize=5, capthick=2, markersize=10,
                   color='red', ecolor='darkred', linewidth=2.5,
                   markerfacecolor='lightcoral', markeredgecolor='darkred', 
                   markeredgewidth=2, label='Power Law')
        
        ax.axhline(y=2, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, label='α = 2')
        ax.axhline(y=3, color='gray', linestyle=':', alpha=0.5, linewidth=1.5, label='α = 3')
        
        ax.set_xlabel('Fracción de área ocupada (φ)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Exponente α', fontsize=13, fontweight='bold')
        ax.set_title('Exponente α de Power Law vs Densidad\n(Solo configuraciones donde Power Law es mejor)', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=11)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'alpha_vs_phi.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'alpha_vs_phi.pdf'), bbox_inches='tight')
        plt.close()
        print(f"   ✓ Guardado: alpha_vs_phi.png")
    
    # ========== GRÁFICO 2: LAMBDA vs PHI (solo exponenciales) ==========
    if len(exp_results) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        phi_vals = [r['phi'] for r in exp_results]
        lambda_vals = [r['lambda'] for r in exp_results]
        
        ax.plot(phi_vals, lambda_vals, 
               'o-', markersize=10, linewidth=2.5,
               color='blue', markerfacecolor='lightblue', 
               markeredgecolor='darkblue', markeredgewidth=2,
               label='Exponencial')
        
        ax.set_xlabel('Fracción de área ocupada (φ)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Parámetro λ', fontsize=13, fontweight='bold')
        ax.set_title('Parámetro λ de Exponencial vs Densidad\n(Solo configuraciones donde Exponencial es mejor)', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', fontsize=11)
        ax.set_ylim(bottom=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lambda_vs_phi.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'lambda_vs_phi.pdf'), bbox_inches='tight')
        plt.close()
        print(f"   ✓ Guardado: lambda_vs_phi.png")
    
    # ========== GRÁFICO 3: RESUMEN COMPLETO ==========
    fig, ax = plt.subplots(figsize=(12, 7))
    
    if len(powerlaw_results) > 0:
        phi_pl = [r['phi'] for r in powerlaw_results]
        alpha_pl = [r['alpha'] for r in powerlaw_results]
        ax.plot(phi_pl, alpha_pl, 'o-', markersize=12, linewidth=2.5,
               color='red', markerfacecolor='lightcoral', 
               markeredgecolor='darkred', markeredgewidth=2,
               label=f'Power Law (n={len(powerlaw_results)})')
    
    if len(exp_results) > 0:
        phi_exp = [r['phi'] for r in exp_results]
        # Normalizar lambda para comparar visualmente (escalar a rango de alpha)
        lambda_exp = [r['lambda'] for r in exp_results]
        ax.plot(phi_exp, lambda_exp, 's-', markersize=12, linewidth=2.5,
               color='blue', markerfacecolor='lightblue', 
               markeredgecolor='darkblue', markeredgewidth=2,
               label=f'Exponencial λ (n={len(exp_results)})')
    
    if len(log_results) > 0:
        phi_log = [r['phi'] for r in log_results]
        mu_log = [r['mu'] for r in log_results]
        ax.plot(phi_log, mu_log, '^-', markersize=12, linewidth=2.5,
               color='green', markerfacecolor='lightgreen', 
               markeredgecolor='darkgreen', markeredgewidth=2,
               label=f'Lognormal μ (n={len(log_results)})')
    
    ax.set_xlabel('Fracción de área ocupada (φ)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Parámetro característico', fontsize=13, fontweight='bold')
    ax.set_title('Distribuciones Mejor Ajustadas vs Densidad\n(Clasificación según test de likelihood ratio)', 
                fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_parameters_vs_phi.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'all_parameters_vs_phi.pdf'), bbox_inches='tight')
    plt.close()
    print(f"   ✓ Guardado: all_parameters_vs_phi.png")

def print_summary_table(results):
    """
    Imprime tabla resumen de resultados.
    """
    print("\n" + "="*120)
    print("TABLA RESUMEN DE ANÁLISIS")
    print("="*120)
    print(f"\n{'N':>5} {'φ':>8} {'Mejor Dist.':>15} {'Parámetro':>25} {'p_exp':>8} {'p_log':>8}")
    print("-" * 120)
    
    for r in results:
        if r['best_dist'] == 'power_law':
            param_str = f"α = {r['alpha']:.3f} ± {r['alpha_sigma']:.3f}"
        elif r['best_dist'] == 'exponential':
            param_str = f"λ = {r['lambda']:.4f}"
        else:
            param_str = f"μ = {r['mu']:.3f}, σ = {r['lognormal_sigma']:.3f}"
        
        print(f"{r['N']:5d} {r['phi']:8.4f} {r['best_dist']:>15} {param_str:>25} "
              f"{r['p_exp']:8.4f} {r['p_log']:8.4f}")
    
    print("="*120)
    
    # Estadísticas
    n_pl = sum(1 for r in results if r['best_dist'] == 'power_law')
    n_exp = sum(1 for r in results if r['best_dist'] == 'exponential')
    n_log = sum(1 for r in results if r['best_dist'] == 'lognormal')
    
    print(f"\nRESUMEN:")
    print(f"  Power Law:    {n_pl:2d} / {len(results)} ({100*n_pl/len(results):.1f}%)")
    print(f"  Exponencial:  {n_exp:2d} / {len(results)} ({100*n_exp/len(results):.1f}%)")
    print(f"  Lognormal:    {n_log:2d} / {len(results)} ({100*n_log/len(results):.1f}%)")
    print("="*120 + "\n")

# ========== SCRIPT PRINCIPAL ==========
if __name__ == "__main__":
    print("\n" + "="*80)
    print("ANÁLISIS DE DISTRIBUCIONES - TIEMPOS ENTRE CONTACTOS")
    print("="*80 + "\n")

    data_dir = "output\\study1\\data"
    times_dir = "data\\times"
    output_dir = "output\\study2\\garphs"

    try:
        # 1. Cargar datos
        print("1. Cargando datos...")
        data = load_contact_times(data_dir, times_dir)
        
        if len(data) == 0:
            raise ValueError("No se cargaron datos")
        
        # 2. Analizar cada N
        print(f"\n2. Analizando distribuciones...")
        results = []
        for N in sorted(data.keys()):
            print(f"   Analizando N={N}...", end="")
            result = analyze_distribution(data[N]['inter_contact_times'], N, data[N]['phi'])
            if result:
                results.append(result)
                print(f" ✓ Mejor: {result['best_dist']}")
            else:
                print(f" ✗ Datos insuficientes")
        
        if len(results) == 0:
            raise ValueError("No se generaron resultados")
        
        # 3. Imprimir resumen
        print_summary_table(results)
        
        # 4. Generar gráficos individuales
        print("3. Generando gráficos individuales por N...")
        plot_individual_distributions(results, output_dir)
        
        # 5. Generar gráficos de parámetros vs phi
        print("\n4. Generando gráficos de parámetros vs φ...")
        plot_parameters_vs_phi(results, output_dir)
        
        print("\n" + "="*80)
        print("✓ ANÁLISIS COMPLETADO")
        print(f"   Archivos guardados en: {output_dir}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()