import streamlit as st
import matplotlib.pyplot as plt

def display_results_table(y_v2, y_rf, y_lr=None):
    """Display predictions in a clean table."""
    rows = []
    rows.append(('Champion Forecast (v2)', f'{y_v2:,.0f}'))
    if y_lr is not None:
        rows.append(('Linear Regression (v1)', f'{y_lr:,.0f}'))
    if y_rf is not None:
        rows.append(('Random Forest (v1)', f'{y_rf:,.0f}'))

    table_rows = ''
    for model_name, value in rows:
        is_first = (model_name == rows[0][0])
        row_class = ' class="highlight-row"' if is_first else ''
        table_rows += f'<tr{row_class}><td>{model_name}</td><td>{value} kg/ha</td></tr>\n'

    table_html = f"""
    <table class="data-table">
        <thead><tr><th>Model</th><th>Prediction</th></tr></thead>
        <tbody>{table_rows}</tbody>
    </table>
    """
    st.markdown(table_html, unsafe_allow_html=True)

def create_area_chart(df, crop):
    """Modern area chart with green gradient fill for historical yield."""
    crop_df = df[df['Item'] == crop].copy()
    avg_yield = crop_df.groupby('Year')['kg_per_ha_yield'].mean().reset_index()
    years = avg_yield['Year'].values
    yields = avg_yield['kg_per_ha_yield'].values

    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor('transparent')
    ax.set_facecolor('transparent')

    ax.fill_between(years, yields, alpha=0.35, color='#2D5A27')
    ax.plot(years, yields, color='#2D5A27', linewidth=3, marker='o', markersize=6, markerfacecolor='#FFFFFF', markeredgecolor='#2D5A27', markeredgewidth=2, zorder=3)

    ax.set_title(f'{crop} — Yield Trajectory Over Time', fontsize=15, fontweight='700', color='#1A1A2E', pad=18, loc='left')
    ax.set_xlabel('Year', fontsize=12, color='#6B7280', fontweight='500')
    ax.set_ylabel('Average Yield (kg/ha)', fontsize=12, color='#6B7280', fontweight='500')

    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.8, color='#D1D5DB')
    ax.set_axisbelow(True)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
    for spine in ['left', 'bottom']:
        ax.spines[spine].set_color('#E5E7EB')
        ax.spines[spine].set_linewidth(1)

    ax.tick_params(colors='#6B7280', labelsize=11)
    ax.set_xlim(min(years) - 0.5, max(years) + 0.5)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    return fig

def create_importance_chart(importance_dict):
    """Horizontal bar chart for feature importance with earthy green palette."""
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    features, importances = zip(*sorted_items)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('transparent')
    ax.set_facecolor('transparent')

    palette = ['#1B4332', '#2D5A27', '#388E3C', '#4CAF50', '#66BB6A']
    colors = palette[:len(features)]

    bars = ax.barh(list(features), list(importances), color=colors, height=0.55, edgecolor='none', alpha=0.9)

    for bar, imp in zip(bars, importances):
        ax.text(imp + 0.02, bar.get_y() + bar.get_height() / 2, f'{imp * 100:.1f}%', va='center', fontsize=10.5, color='#1A1A2E', fontweight='500')

    ax.set_title('What Drives the Prediction?', fontsize=15, fontweight='700', color='#1A1A2E', pad=18, loc='left')
    ax.set_xlabel('Relative Contribution', fontsize=12, color='#6B7280', fontweight='500')

    xlim = max(importances) * 1.2
    ax.set_xlim(0, min(xlim, 0.6))

    for spine in ['top', 'right', 'left', 'bottom']:
        ax.spines[spine].set_visible(False)

    ax.tick_params(axis='y', labelsize=11, length=0)
    ax.set_yticklabels(list(features), color='#1A1A2E')
    ax.invert_yaxis()
    ax.set_axisbelow(True)

    fig.tight_layout()
    return fig
