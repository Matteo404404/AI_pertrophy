"""
Scientific Hypertrophy Trainer - Charting Engine
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

def create_sra_curve_chart(dates, fitness, fatigue, readiness):
    fig, ax = _setup_base_figure(figsize=(10, 4))
    if not dates: return FigureCanvas(fig)
    x = np.arange(len(dates))
    ax.plot(x, fitness, color=GREEN, linewidth=2.5, label='Fitness')
    ax.plot(x, fatigue, color=RED, linewidth=2.5, label='Fatigue')
    ax.plot(x, readiness, color=BLUE, linewidth=2, linestyle='--', label='Readiness')
    ax.fill_between(x, fitness, alpha=0.10, color=GREEN)
    ax.fill_between(x, fatigue, alpha=0.10, color=RED)
    ax.set_title("SRA Trajectory (Banister Model)", pad=15, fontweight='900', color=TEXT_COLOR)
    step = max(1, len(dates) // 7)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=45)
    ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor='white', loc='upper left')
    fig.tight_layout()
    canvas = FigureCanvas(fig)
    _add_tooltip(canvas, fig, ax, dates, [readiness, fitness, fatigue],['🔵 Net Readiness', '🟢 Fitness', '🔴 Fatigue'])
    return canvas

def create_1rm_trend_chart(dates, dict_of_lifts):
    fig, ax = _setup_base_figure(figsize=(10, 4))
    if not dates or not dict_of_lifts: return FigureCanvas(fig)
    x = np.arange(len(dates))
    colors =[BLUE, PURPLE, YELLOW, GREEN, RED]
    ys, labels = [],[]
    for i, (lift_name, maxes) in enumerate(dict_of_lifts.items()):
        c = colors[i % len(colors)]
        ax.plot(x, maxes, color=c, linewidth=3, marker='o', markersize=6, label=lift_name)
        ys.append(maxes)
        labels.append(f"🏋️ {lift_name} (kg)")
    ax.set_title("Estimated 1RM Progression", pad=15, fontweight='900', color=TEXT_COLOR)
    step = max(1, len(dates) // 7)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=45)
    ax.legend(facecolor=BG_COLOR, edgecolor=GRID_COLOR, labelcolor='white', loc='upper left')
    fig.tight_layout()
    canvas = FigureCanvas(fig)
    _add_tooltip(canvas, fig, ax, dates, ys, labels)
    return canvas

def create_tonnage_chart(dates, tonnage):
    fig, ax = _setup_base_figure(figsize=(5, 4))
    if not dates: return FigureCanvas(fig)
    x = np.arange(len(dates))
    bars = ax.bar(x, tonnage, color=PURPLE, width=0.6, alpha=0.8)
    ax.set_title("Weekly Tonnage (kg)", pad=15, fontweight='900', color=TEXT_COLOR)
    step = max(1, len(dates) // 5)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([dates[i] for i in range(0, len(dates), step)], rotation=45)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + (max(tonnage)*0.02), f'{int(yval)}', ha='center', va='bottom', color=TEXT_COLOR, fontsize=8)
    fig.tight_layout()
    return FigureCanvas(fig)

def create_muscle_readiness_chart(muscles, readiness_scores):
    fig, ax = _setup_base_figure(figsize=(5, 4))
    if not muscles: return FigureCanvas(fig)
    y_pos = np.arange(len(muscles))
    # 100 = Green (Ready), <50 = Red (Fatigued)
    colors =[GREEN if s > 75 else YELLOW if s > 40 else RED for s in readiness_scores]
    bars = ax.barh(y_pos, readiness_scores, color=colors, height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(muscles, fontweight='bold', color=TEXT_COLOR)
    ax.set_xlim(0, 100)
    ax.set_title('Live Muscle Readiness (%)', pad=15, fontweight='900', color=TEXT_COLOR)
    for bar in bars:
        width = bar.get_width()
        ax.text(width - 15, bar.get_y() + bar.get_height()/2, f'{int(width)}%', color=BG_COLOR, va='center', fontweight='900')
    fig.tight_layout()
    return FigureCanvas(fig)

def create_sfr_scatter_plot(exercises, stimulus, fatigue):
    fig, ax = _setup_base_figure(figsize=(10, 5))
    if not exercises: return FigureCanvas(fig)
    
    # Draw Quadrants
    mid_s = np.mean(stimulus) if stimulus else 5
    mid_f = np.mean(fatigue) if fatigue else 5
    ax.axhline(y=mid_s, color=GRID_COLOR, linestyle='-', zorder=1)
    ax.axvline(x=mid_f, color=GRID_COLOR, linestyle='-', zorder=1)
    
    # Quadrant Labels
    ax.text(min(fatigue), max(stimulus), "OPTIMAL\n(High Stim/Low Fat)", color=GREEN, alpha=0.5, fontsize=10, fontweight='bold', va='top')
    ax.text(max(fatigue), min(stimulus), "JUNK VOLUME\n(Low Stim/High Fat)", color=RED, alpha=0.5, fontsize=10, fontweight='bold', ha='right', va='bottom')
    
    # Plot Points
    sc = ax.scatter(fatigue, stimulus, c=BLUE, s=150, alpha=0.7, edgecolors='white', zorder=5)
    
    ax.set_xlabel("Systemic Fatigue Cost", color=TEXT_COLOR, fontweight='bold', labelpad=10)
    ax.set_ylabel("Hypertrophic Stimulus", color=TEXT_COLOR, fontweight='bold', labelpad=10)
    ax.set_title("Stimulus-to-Fatigue Ratio (SFR) Matrix", pad=15, fontweight='900', color=TEXT_COLOR)
    fig.tight_layout()
    
    # Interactive Hover for Scatter Plot
    canvas = FigureCanvas(fig)
    annot = ax.annotate("", xy=(0,0), xytext=(10, 10), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.5", fc=TOOLTIP_BG, ec=PURPLE, lw=1.5, alpha=0.9),
                        color="white", fontsize=10, fontweight='bold', zorder=20)
    annot.set_visible(False)

    def hover(event):
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                pos = sc.get_offsets()[ind["ind"][0]]
                annot.xy = pos
                ex_idx = ind["ind"][0]
                annot.set_text(f"🏋️ {exercises[ex_idx]}\nStimulus: {stimulus[ex_idx]:.1f}\nFatigue: {fatigue[ex_idx]:.1f}")
                annot.set_visible(True)
                canvas.draw_idle()
            else:
                if annot.get_visible():
                    annot.set_visible(False)
                    canvas.draw_idle()

    canvas.mpl_connect("motion_notify_event", hover)
    return canvas
