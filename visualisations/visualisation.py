from matplotlib import pyplot as plt

def visualise_results(prices, bot, best_params, convergence, strategy_name='SMA'):
    """Create comprehensive visualization"""


    results = bot.backtest_detailed(best_params)

    if results is None:
        return

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(prices, label='Price', color='blue', alpha=0.7, linewidth=1)


    buy_indices = [t['idx'] for t in results['trade_log'] if t['type'] == 'BUY']
    sell_indices = [t['idx'] for t in results['trade_log'] if t['type'] == 'SELL']

    ax1.scatter(buy_indices, prices[buy_indices],
                color='green', marker='^', s=100, label='Buy', zorder=5)
    ax1.scatter(sell_indices, prices[sell_indices],
                color='red', marker='v', s=100, label='Sell', zorder=5)

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price ($)')
    ax1.set_title(f'{strategy_name} Strategy: Price & Trading Signals')
    ax1.legend()
    ax1.grid(True, alpha=0.3)


    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(results['portfolio_history'], color='purple', linewidth=2)
    ax2.axhline(y=1000, color='gray', linestyle='--', label='Initial Capital')
    ax2.fill_between(range(len(results['portfolio_history'])),
                     1000, results['portfolio_history'],
                     alpha=0.3, color='purple')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Portfolio Value ($)')
    ax2.set_title('Portfolio Value Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)


    ax3 = fig.add_subplot(gs[2, 0])
    ax3.plot(convergence, color='orange', linewidth=2)
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Best Fitness ($)')
    ax3.set_title('Optimization Convergence')
    ax3.grid(True, alpha=0.3)

    # 4. Performance Metrics
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')

    # Calculate metrics
    total_return = (results['final_cash'] - 1000) / 1000 * 100
    num_trades = len(results['trade_log'])

    metrics_text = f"""
    PERFORMANCE METRICS
    {'='*30}
    
    Initial Capital:  $1,000.00
    Final Cash:       ${results['final_cash']:.2f}
    Total Return:     {total_return:.2f}%
    
    Number of Trades: {num_trades}
    
    Best Parameters:
    """

    if strategy_name == 'SMA':
        metrics_text += f"  Short Window:  {int(best_params[0])}\n"
        metrics_text += f"  Long Window:   {int(best_params[1])}\n"
    elif strategy_name == 'MACD':
        metrics_text += f"  Fast Period:   {int(best_params[0])}\n"
        metrics_text += f"  Slow Period:   {int(best_params[1])}\n"
        metrics_text += f"  Signal Period: {int(best_params[2])}\n"

    ax4.text(0.1, 0.5, metrics_text, fontsize=11,
             verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle(f'{strategy_name} Trading Bot Optimization Results',
                 fontsize=16, fontweight='bold')

    plt.savefig(f'png/{strategy_name}_optimization_results.png', dpi=150, bbox_inches='tight')
    print(f"\n Visualization saved as '{strategy_name}_optimization_results.png'")
    plt.show()
