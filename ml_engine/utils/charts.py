"""
Scientific Hypertrophy Trainer - Advanced Charting Engine (INTERACTIVE)
Generates premium, dark-themed 'Athlete HUD' visualizations.
"""

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

BG_COLOR = '#181825'
GRID_COLOR = '#2a2b3c'
TEXT_COLOR = '#cdd6f4'
GREEN = '#a6e3a1'
RED = '#f38ba8'
BLUE = '#89b4fa'
PURPLE = '#cba6f7'
YELLOW = '#f9e2af'
TOOLTIP_BG = '#1e1e2e'

def _setup_base_figure(figsize=(8, 4)):
    fig = Figure(figsize=figsize, dpi=100)
    fig.patch.set_facecolor(BG_COLOR)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(True, linestyle='--', alpha=0.2, color=TEXT_COLOR)
    return fig, ax

def _add_tooltip(canvas, fig, ax, data_x, data_ys, labels, is_bar=False):
    """Universal highly-styled tooltip injector"""
    annot = ax.annotate("", xy=(0,0), xytext=(15, 15), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.6", fc=TOOLTIP_BG, ec=BLUE, lw=1.5, alpha=0.95),
                        color="white", fontsize=10, fontweight='bold', zorder=20)
    annot.set_visible(False)
    vline = ax.axvline(x=0, color='white', alpha=0.5, linestyle=':', visible=not is_bar)

    def hover(event):
        if event.inaxes == ax:
            x_val = int(round(event.xdata)) if event.xdata else -1
            if 0 <= x_val < len(data_x):
                if not is_bar: vline.set_xdata([x_val])
                
                # Dynamic positioning so it doesn't clip off screen
                offset_x = -120 if x_val > len(data_x) * 0.6 else 15
                annot.set_position((offset_x, 15))
                
                text = f"📅 {data_x[x_val]}\n"
                for i, y_arr in enumerate(data_ys):
                    text += f"{labels[i]}: {y_arr[x_val]:.1f}\n"
                
                annot.xy = (x_val, np.mean([y[x_val] for y in data_ys]))
                annot.set_text(text.strip())
                annot.set_visible(True)
                canvas.draw_idle()
        else:
            if annot.get_visible():
                annot.set_visible(False)
                canvas.draw_idle()

    canvas.mpl_connect("motion_notify_event", hover)

# --- THE CHARTS ---

def create_sra_curve_chart(dates, fitness, fatigue, readiness):
    fig, ax = _setup_base_figure(figsize=(10, 4))
    if not dates: return FigureCanvas(fig)
    x = np.arange(len(dates))
    
    ax.plot(x, fitness, color=GREEN, linewidth=2.5, label='Fitness')
    ax.plot(x, fatigue, color=RED, linewidth=2.5, label='Fatigue')
    ax.plot(x, readiness, color=BLUE, linewidth=2, linestyle='--', label='Readiness')
    
    ax.fill_between(x, fitness, alpha=0.10, color=GREEN)
    ax.fill_between(x, fatigue, alpha=0.10, color=RED)
    
    ax.set_title("Stimulus-Recovery-Adaptation (SRA) Trajectory", pad=15, fontweight='900', color=TEXT_COLOR)
    
    step = max(1, len(dates) // 7)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=45)
    ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor='white', loc='upper left')
    fig.tight_layout()
    
    canvas = FigureCanvas(fig)
    _add_tooltip(canvas, fig, ax, dates, [readiness, fitness, fatigue], ['🔵 Net Readiness', '🟢 Fitness', '🔴 Fatigue'])
    return canvas

def create_1rm_trend_chart(dates, dict_of_lifts):
    """Plots estimated 1RM progression for top exercises."""
    fig, ax = _setup_base_figure(figsize=(10, 4))
    if not dates or not dict_of_lifts: return FigureCanvas(fig)
    
    x = np.arange(len(dates))
    colors =[BLUE, PURPLE, YELLOW, GREEN, RED]
    
    ys = []
    labels =[]
    for i, (lift_name, maxes) in enumerate(dict_of_lifts.items()):
        c = colors[i % len(colors)]
        ax.plot(x, maxes, color=c, linewidth=3, marker='o', markersize=6, label=lift_name)
        ys.append(maxes)
        labels.append(f"🏋️ {lift_name} (kg)")
        
    ax.set_title("Estimated 1RM Progression (Strength)", pad=15, fontweight='900', color=TEXT_COLOR)
    
    step = max(1, len(dates) // 7)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=45)
    ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor='white', loc='upper left')
    fig.tight_layout()
    
    canvas = FigureCanvas(fig)
    _add_tooltip(canvas, fig, ax, dates, ys, labels)
    return canvas

def create_tonnage_chart(dates, tonnage):
    """Bar chart for Weekly Tonnage (Load x Reps x Sets)"""
    fig, ax = _setup_base_figure(figsize=(5, 4))
    if not dates: return FigureCanvas(fig)
    
    x = np.arange(len(dates))
    bars = ax.bar(x, tonnage, color=PURPLE, width=0.6, alpha=0.8)
    
    ax.set_title("Total Tonnage (Volume Load kg)", pad=15, fontweight='900', color=TEXT_COLOR)
    
    step = max(1, len(dates) // 5)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=45)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + (max(tonnage)*0.02), f'{int(yval)}', ha='center', va='bottom', color=TEXT_COLOR, fontsize=8)
        
    fig.tight_layout()
    return FigureCanvas(fig)

def create_volume_distribution_chart(muscle_groups, sets):
    fig, ax = _setup_base_figure(figsize=(5, 4))
    if not muscle_groups: return FigureCanvas(fig)

    y_pos = np.arange(len(muscle_groups))
    colors =[BLUE if s < 6 else GREEN if s <= 18 else RED for s in sets]
    bars = ax.barh(y_pos, sets, color=colors, height=0.6)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(muscle_groups, fontweight='bold', color=TEXT_COLOR)
    ax.set_title('Weekly Volume Distribution (Sets)', pad=15, fontweight='900', color=TEXT_COLOR)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f'{int(width)}', color=TEXT_COLOR, va='center', fontweight='bold')
        
    fig.tight_layout()
    return FigureCanvas(fig)

def create_recovery_radar_chart(metrics):
    fig = Figure(figsize=(5, 4), dpi=100)
    fig.patch.set_facecolor(BG_COLOR)
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor(BG_COLOR)
    
    categories = list(metrics.keys())
    values = list(metrics.values())
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]
    
    ax.plot(angles, values, color=YELLOW, linewidth=2.5)
    ax.fill(angles, values, color=YELLOW, alpha=0.15)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color=TEXT_COLOR, fontweight='900', size=9)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels([]) 
    
    ax.spines['polar'].set_color(GRID_COLOR)
    ax.grid(color=GRID_COLOR, linestyle='--')
    ax.set_title("Systemic Readiness Index", color=TEXT_COLOR, pad=20, fontweight='900')
    fig.tight_layout()
    return FigureCanvas(fig)