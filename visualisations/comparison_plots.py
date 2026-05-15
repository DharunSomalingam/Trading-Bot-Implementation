import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

ALGO_COLOURS = {
    'GA':  '#2196F3',   # blue
    'DE':  '#FF9800',   # orange
    'PSO': '#4CAF50',   # green
}

STRAT_COLOURS = {
    '2D_SMA':  '#E91E63',
    '3D_MACD': '#9C27B0',
    '7D_WMA':  '#00BCD4',
    '14D_WMA': '#FF5722',
}

DIMS = {
    '2D_SMA':  2,
    '3D_MACD': 3,
    '7D_WMA':  7,
    '14D_WMA': 14,
}

OUTPUT_DIR = 'visualisations/plot/comparisons'


def _save(fig, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Saved → {path}")
    plt.close(fig)


# ════════════════════════════════════════════════════════════════════════
# Plot 1 — Convergence curves
# One subplot per strategy, all algorithms overlaid, mean ± std band
# ════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════
# Plot 1 — Convergence curves (2x2 Grid)
# ════════════════════════════════════════════════════════════════════════
def plot_convergence(convergence_data, strategies, algorithms):
    n_strats = len(strategies)
    cols = 2
    rows = (n_strats + 1) // 2 

    fig, axes = plt.subplots(rows, cols, figsize=(11, 4.5 * rows), sharey=False)
    
    axes = axes.flatten() 

    fig.suptitle("Convergence Curves — All Algorithms × Strategies",
                 fontsize=14, fontweight='bold', y=1.02)

    for i, strat in enumerate(strategies):
        ax = axes[i]
        ax.set_title(strat, fontsize=12, fontweight='bold')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Fitness ($)")

        for algo in algorithms:
            runs = []
            for key, curve in convergence_data.items():
                if key[0] == strat and key[1] == algo:
                    runs.append(curve)

            if not runs:
                continue

            min_len = min(len(r) for r in runs)
            arr     = np.array([r[:min_len] for r in runs])
            mean    = arr.mean(axis=0)
            std     = arr.std(axis=0)
            iters   = np.arange(min_len)

            colour = ALGO_COLOURS.get(algo, 'gray')
            ax.plot(iters, mean, color=colour, linewidth=2, label=algo)
            ax.fill_between(iters, mean - std, mean + std,
                            color=colour, alpha=0.15)

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=9)
            
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'${x:,.0f}')
        )
        ax.grid(True, alpha=0.3)

    for j in range(n_strats, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    _save(fig, 'plot1_convergence_curves.png')


# ════════════════════════════════════════════════════════════════════════
# Plot 2 — Test performance vs dimensions (line chart)
# ════════════════════════════════════════════════════════════════════════
def plot_test_vs_dimensions(results, strategies, algorithms):
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.set_title("Test Performance vs Problem Dimensionality",
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Dimensions", fontsize=12)
    ax.set_ylabel("Mean Test Fitness ($)", fontsize=12)

    dims   = [DIMS[s] for s in strategies]
    xticks = sorted(set(dims))

    for algo in algorithms:
        means = []
        stds  = []
        xs    = []

        for strat in strategies:
            key = (strat, algo)
            if key not in results:
                continue
            means.append(results[key]['test_mean'])
            stds.append(results[key]['test_std'])
            xs.append(DIMS[strat])

        # sort by dimension
        order = np.argsort(xs)
        xs    = np.array(xs)[order]
        means = np.array(means)[order]
        stds  = np.array(stds)[order]

        colour = ALGO_COLOURS.get(algo, 'gray')
        ax.plot(xs, means, 'o-', color=colour, linewidth=2.5,
                markersize=8, label=algo, zorder=3)
        ax.fill_between(xs, means - stds, means + stds,
                        color=colour, alpha=0.12)

    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{d}D' for d in xticks])
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'${x:,.0f}')
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    for strat in strategies:
        d = DIMS[strat]
        ax.annotate(strat, xy=(d, ax.get_ylim()[0]),
                    xytext=(0, -30), textcoords='offset points',
                    ha='center', fontsize=8, color='gray')

    fig.tight_layout()
    _save(fig, 'plot2_test_vs_dimensions.png')


# ════════════════════════════════════════════════════════════════════════
# Plot 3 — Training vs Test grouped bar chart
# ════════════════════════════════════════════════════════════════════════
def plot_train_vs_test_grouped(results, strategies, algorithms):
    n_strats = len(strategies)
    n_algos  = len(algorithms)
    width    = 0.12
    x        = np.arange(n_strats)

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_title("Training vs Test Performance by Strategy & Algorithm",
                 fontsize=14, fontweight='bold')

    offsets = np.linspace(
        -(n_algos - 1) * width,
         (n_algos - 1) * width,
        n_algos
    )

    for i, algo in enumerate(algorithms):
        train_means, train_stds = [], []
        test_means,  test_stds  = [], []

        for strat in strategies:
            key = (strat, algo)
            if key in results:
                train_means.append(results[key]['train_mean'])
                train_stds.append(results[key]['train_std'])
                test_means.append(results[key]['test_mean'])
                test_stds.append(results[key]['test_std'])
            else:
                train_means.append(0)
                train_stds.append(0)
                test_means.append(0)
                test_stds.append(0)

        colour = ALGO_COLOURS.get(algo, 'gray')

        # Training bars — solid
        ax.bar(x + offsets[i] - width/2, train_means,
               width=width, color=colour, alpha=0.85,
               yerr=train_stds, capsize=3,
               label=f'{algo} train')

        # Test bars — hatched
        ax.bar(x + offsets[i] + width/2, test_means,
               width=width, color=colour, alpha=0.4,
               hatch='//', yerr=test_stds, capsize=3,
               label=f'{algo} test')

    ax.set_xticks(x)
    ax.set_xticklabels(strategies, fontsize=11)
    ax.set_ylabel("Mean Fitness ($)", fontsize=12)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'${x:,.0f}')
    )

    # Custom legend — solid = train, hatched = test
    legend_elements = []
    for algo in algorithms:
        c = ALGO_COLOURS.get(algo, 'gray')
        legend_elements.append(
            mpatches.Patch(facecolor=c, alpha=0.85, label=f'{algo} (train)')
        )
        legend_elements.append(
            mpatches.Patch(facecolor=c, alpha=0.4,
                           hatch='//', label=f'{algo} (test)')
        )
    ax.legend(handles=legend_elements, fontsize=8,
              ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    _save(fig, 'plot3_train_vs_test_grouped.png')


# ════════════════════════════════════════════════════════════════════════
# Plot 4 — Performance of algorithms across different dimensions (TEST ONLY)
# ════════════════════════════════════════════════════════════════════════
def plot_algo_across_dimensions(results, strategies, algorithms):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.set_title("Algorithm Out-of-Sample (Test) Performance Across Dimensions", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Strategy (Dimensions)", fontsize=11)
    ax.set_ylabel("Mean Test Fitness ($)", fontsize=11)

    x      = np.arange(len(strategies))
    width  = 0.25
    offset = np.linspace(
        -(len(algorithms)-1)*width/2,
         (len(algorithms)-1)*width/2,
        len(algorithms)
    )

    for i, algo in enumerate(algorithms):
        means = [results.get((s, algo), {}).get('test_mean', 0) for s in strategies]
        errs  = [results.get((s, algo), {}).get('test_std', 0) for s in strategies]
        
        colour = ALGO_COLOURS.get(algo, 'gray')
        ax.bar(x + offset[i], means, width=width,
               color=colour, alpha=0.85, label=algo,
               yerr=errs, capsize=3)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f'{s}\n({DIMS[s]}D)' for s in strategies], 
        fontsize=9
    )
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'${v:,.0f}')
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    _save(fig, 'plot4_algo_across_dimensions.png')

# ════════════════════════════════════════════════════════════════════════
# Plot 5 — Performance comparison across algorithms, same dimension (TEST ONLY)
# ════════════════════════════════════════════════════════════════════════
# ════════════════════════════════════════════════════════════════════════
# Plot 5 — Performance comparison across algorithms, same dimension (2x2 Grid)
# ════════════════════════════════════════════════════════════════════════
def plot_algo_same_dimension(results, strategies, algorithms):
    n_strats = len(strategies)
    cols = 2
    rows = (n_strats + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(11, 4.5 * rows), sharey=False)
    axes = axes.flatten()

    fig.suptitle("Algorithm Out-of-Sample (Test) Comparison Within Each Strategy", 
                 fontsize=14, fontweight='bold', y=1.02)

    for i, strat in enumerate(strategies):
        ax = axes[i]
        ax.set_title(f"{strat} ({DIMS[strat]}D)", 
                     fontsize=12, fontweight='bold')
        ax.set_xlabel("Algorithm", fontsize=11)
        ax.set_ylabel("Test Fitness ($)", fontsize=11)

        x       = np.arange(len(algorithms))
        colours = [ALGO_COLOURS.get(a, 'gray') for a in algorithms]

        test_means  = [results.get((strat, a), {}).get('test_mean', 0) for a in algorithms]
        test_stds   = [results.get((strat, a), {}).get('test_std', 0) for a in algorithms]

        ax.bar(x, test_means, width=0.6, 
               color=colours, alpha=0.9, 
               yerr=test_stds, capsize=4)

        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, fontsize=11)
        ax.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda v, _: f'${v:,.0f}')
        )
        ax.grid(True, alpha=0.3, axis='y')

    for j in range(n_strats, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    _save(fig, 'plot5_algo_same_dimension.png')

# ════════════════════════════════════════════════════════════════════════
# Master function 
# ════════════════════════════════════════════════════════════════════════
def generate_all_plots(results, convergence_data=None):
    """
    Call after the main experiment loop.

    Args:
        results:          dict from main.py results{}
        convergence_data: dict {(strat, algo, seed): [fitness per iter]}
                          pass None to skip convergence plot
    """
    strategies = sorted(set(k[0] for k in results))
    algorithms = sorted(set(k[1] for k in results))

    print("\nGenerating comparison plots...")

    if convergence_data:
        print("  Plot 1: Convergence curves...")
        plot_convergence(convergence_data, strategies, algorithms)

    print("  Plot 2: Test performance vs dimensions...")
    plot_test_vs_dimensions(results, strategies, algorithms)

    print("  Plot 3: Training vs Test grouped bar...")
    plot_train_vs_test_grouped(results, strategies, algorithms)

    print("  Plot 4: Algorithm performance across dimensions...")
    plot_algo_across_dimensions(results, strategies, algorithms)

    print("  Plot 5: Algorithm comparison same dimension...")
    plot_algo_same_dimension(results, strategies, algorithms)

    print("\nAll plots saved to visualisations/")