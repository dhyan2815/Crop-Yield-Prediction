# Yield Metrics - Modern Minimal UI Redesign

## Overview

**Project:** Crop Yield Prediction Web Application (Streamlit)
**Date:** 2026-03-30
**Goal:** Redesign the visualization layer to be modern, minimal, and more visually appealing while maintaining data clarity.

---

## 1. Design System

### Color Palette (Earthy Green)

| Token | Hex | Usage |
|-------|-----|-------|
| `primary` | `#2D5A27` | Buttons, accents, primary chart color |
| `secondary` | `#7CB342` | Highlights, secondary elements |
| `tertiary` | `#81C784` | Lighter accents, hover states |
| `background` | `#FAFAFA` | Main page background |
| `surface` | `#FFFFFF` | Cards, elevated surfaces |
| `surface_alt` | `#E8F5E9` | Alternating rows, subtle backgrounds |
| `text_primary` | `#1A1A2E` | Headings, primary text |
| `text_secondary` | `#6B7280` | Labels, secondary text |
| `border` | `#E5E7EB` | Subtle borders |
| `warning` | `#F59E0B` | Disclaimer banner |

### Typography

- **Font Family:** System sans-serif stack (`-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`)
- **Headings:**
  - H1 (Page title): 2rem, weight 700
  - H2 (Section headers): 1.5rem, weight 600
  - H3 (Sub-headers): 1.25rem, weight 600
- **Body:** 1rem, weight 400
- **Labels:** 0.875rem, weight 500
- **Captions:** 0.75rem, weight 400

### Spacing System

- Base unit: 8px
- Section padding: 32px
- Card padding: 24px
- Element gap: 16px
- Tight gap: 8px

### Shadows

- **Card shadow:** `0 1px 3px rgba(0,0,0,0.1), 0 1px 2px rgba(0,0,0,0.06)`
- **Elevated shadow:** `0 4px 6px rgba(0,0,0,0.1), 0 2px 4px rgba(0,0,0,0.06)`

---

## 2. Page Layout

### Single Page Flow (Top-to-Bottom)

```
┌─────────────────────────────────────────┐
│  HEADER                                 │
│  🌾 Yield Metrics                       │
│  Welcome subtitle                       │
├─────────────────────────────────────────┤
│  DISCLAIMER BANNER (st.warning)        │
│  Yellow/amber styling                   │
├─────────────────────────────────────────┤
│  INPUT PARAMETERS                       │
│  Crop Select | Year | Pesticide Input  │
│  [Predict Button - Full Width]          │
├─────────────────────────────────────────┤
│  PREDICTION RESULTS TABLE               │
│  Clean bordered table with results       │
├─────────────────────────────────────────┤
│  HISTORICAL YIELD TREND (Area Chart)   │
│  Gradient-filled area visualization     │
├─────────────────────────────────────────┤
│  FEATURE IMPORTANCE (Bar Chart)         │
│  Horizontal bars showing contribution    │
├─────────────────────────────────────────┤
│  FOOTER                                 │
│  Copyright, subtle styling              │
└─────────────────────────────────────────┘
```

---

## 3. Component Specifications

### 3.1 Header

```python
st.title("🌾 Yield Metrics")
st.markdown("""
Welcome to **Yield Metrics** – a crop yield prediction app for India.
""")
```

- No background color
- Minimal styling
- Emoji icon for visual interest

### 3.2 Disclaimer Banner

```python
st.warning("⚠️ **Predictions** are based on historical data...")
```

- Uses Streamlit's built-in `st.warning()` styling
- Amber/yellow color
- Positioned immediately after header

### 3.3 Input Parameters Section

**Layout:** Three columns for inputs + full-width button

```
┌──────────────────────────────────────────────────────────┐
│  📝 Input Parameters                                     │
│  ┌─────────────┐ ┌─────────────┐ ┌───────────────────┐  │
│  │ 🌱 Crop    │ │ 📅 Year     │ │ 🧪 Pesticide (t) │  │
│  │ [Dropdown]  │ │ [Number]   │ │ [Number Input]    │  │
│  └─────────────┘ └─────────────┘ └───────────────────┘  │
│  ┌────────────────────────────────────────────────────┐│
│  │            🚀 Predict Yield                         ││
│  └────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────┘
```

**Styling:**
- Section header with icon
- Three equal-width columns using `st.columns(3)`
- Select box: full width of column, rounded appearance
- Number inputs: minimal styling
- Predict button: `type="primary"`, `use_container_width=True`

### 3.4 Prediction Results Table

**Structure:**
```
┌─────────────────────────────────────────┐
│  📊 Prediction Results                   │
│  ┌─────────────────┬─────────┬───────┐ │
│  │ Model           │ kg/ha   │ %     │ │
│  ├─────────────────┼─────────┼───────┤ │
│  │ Linear Regression│ 3,817   │ 96.8% │ │
│  │ Random Forest   │ 3,942   │ 100%  │ │
│  └─────────────────┴─────────┴───────┘ │
└─────────────────────────────────────────┘
```

**Implementation:** Use `st.table()` or `st.dataframe()` with custom HTML styling via `st.markdown()`

**Styling:**
- White background card
- Subtle border
- Header row: light sage background (`#E8F5E9`)
- Alternating row colors: white / very light gray
- Model names left-aligned
- Numbers right-aligned
- Green accent for Random Forest (best model indicator)

### 3.5 Historical Yield Trend (Area Chart)

**Chart Type:** Area chart with gradient fill

```python
fig, ax = plt.subplots(figsize=(12, 6))

# Create gradient fill effect
ax.fill_between(x, y, alpha=0.3, color='#2D5A27')
ax.plot(x, y, color='#2D5A27', linewidth=2)

# Styling
ax.set_title(f'{crop} - Historical Yield Trend in India',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Year', fontsize=12, color='#6B7280')
ax.set_ylabel('Average Yield (kg/ha)', fontsize=12, color='#6B7280')
ax.grid(True, alpha=0.2, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

**Features:**
- Green gradient fill (`#2D5A27` → transparent)
- Clean line (`#2D5A27`, 2px width)
- Data point markers (small circles)
- Minimal gridlines (light gray, dashed)
- No top/right spines
- Clean axis labels
- Subtle background

### 3.6 Feature Importance Chart

**Chart Type:** Horizontal bar chart

```python
fig, ax = plt.subplots(figsize=(10, 6))

# Sample features for display
features = ['Crop Type', 'Rainfall', 'Temperature', 'Pesticides', 'Year']
importance = [0.45, 0.25, 0.15, 0.10, 0.05]

colors = ['#2D5A27', '#4CAF50', '#66BB6A', '#81C784', '#A5D6A7']

bars = ax.barh(features, importance, color=colors, height=0.6)

# Add percentage labels
for bar, imp in zip(bars, importance):
    ax.text(imp + 0.01, bar.get_y() + bar.get_height()/2,
            f'{imp*100:.0f}%', va='center', fontsize=10)

ax.set_xlim(0, 0.55)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

**Features:**
- Horizontal bars (easy label reading)
- Green color gradient based on importance
- Percentage labels on bars
- Clean styling, no unnecessary elements

---

## 4. Technical Implementation

### 4.1 Streamlit Configuration

```python
st.set_page_config(
    page_title="Yield Metrics",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed"
)
```

### 4.2 Custom CSS Injection

```python
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Main background */
    .stApp {
        background-color: #FAFAFA;
    }

    /* Section headers */
    h2 {
        color: #1A1A2E;
        font-weight: 600;
        margin-top: 2rem;
    }

    /* Cards */
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }

    /* Custom button styling */
    .stButton > button {
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)
```

### 4.3 Plotly Alternative (Optional)

For smoother gradients and better interactivity, consider using Plotly:

```python
import plotly.express as px
import plotly.graph_objects as go

# Area chart with Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=years, y=yields,
    fill='tozeroy',
    fillcolor='rgba(45, 90, 39, 0.3)',
    line=dict(color='#2D5A27', width=2),
    mode='lines+markers'
))
```

---

## 5. File Changes

| File | Change |
|------|--------|
| `app.py` | Complete rewrite of visualization components |

---

## 6. Verification Checklist

- [ ] All visualizations render correctly
- [ ] Color palette applied consistently
- [ ] Charts are responsive and readable
- [ ] Feature importance chart displays correctly
- [ ] Table styling matches design spec
- [ ] No console errors
- [ ] Mobile-friendly layout

---

## 7. Success Criteria

1. Visual design is clean, minimal, and modern
2. Color palette creates cohesive agricultural theme
3. All three visualizations (table, area chart, bar chart) are functional
4. Information hierarchy is clear
5. Charts are readable and professionally styled
6. No visual clutter or unnecessary elements
