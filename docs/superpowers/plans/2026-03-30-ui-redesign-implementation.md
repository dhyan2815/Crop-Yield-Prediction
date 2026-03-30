# UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the Yield Metrics Streamlit app with a modern minimal aesthetic using an earthy green color palette.

**Architecture:** Single-page layout with vertical flow. Custom CSS injected for styling. Matplotlib with custom colors for all charts. Feature importance extracted from Random Forest model.

**Tech Stack:** Streamlit, Matplotlib, Seaborn, scikit-learn (for feature importance)

---

## File Structure

```
app.py                 # Single file - complete rewrite of visualization components
```

---

## Task 1: Update Page Configuration

**Files:**
- Modify: `app.py:28-32`

- [ ] **Step 1: Update st.set_page_config with better icon and collapsed sidebar**

```python
st.set_page_config(
    page_title="Yield Metrics",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed"
)
```

---

## Task 2: Add Custom CSS Styling

**Files:**
- Modify: `app.py` - Add new section after imports and before model loading

- [ ] **Step 1: Add custom CSS for modern minimal styling**

Add this code block after the imports section:

```python
# =============================================================================
# CUSTOM STYLING
# =============================================================================

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #FAFAFA;
    }

    /* Section headers */
    h2 {
        color: #1A1A2E !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        font-size: 1.25rem !important;
    }

    h3 {
        color: #1A1A2E !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }

    /* Success message */
    .stSuccess {
        background-color: #E8F5E9 !important;
        border-left: 4px solid #2D5A27 !important;
    }

    /* Info message */
    .stInfo {
        background-color: #E3F2FD !important;
        border-left: 4px solid #1976D2 !important;
    }

    /* Warning banner */
    .stWarning {
        background-color: #FFF8E1 !important;
        border-left: 4px solid #F59E0B !important;
    }

    /* Metric styling */
    .stMetric {
        background-color: #FFFFFF;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Custom table styling */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        margin: 1rem 0;
        font-size: 0.95rem;
    }

    .custom-table th {
        background-color: #E8F5E9;
        color: #1A1A2E;
        font-weight: 600;
        padding: 12px 16px;
        text-align: left;
        border-bottom: 2px solid #2D5A27;
    }

    .custom-table td {
        padding: 12px 16px;
        border-bottom: 1px solid #E5E7EB;
    }

    .custom-table tr:nth-child(even) {
        background-color: #FAFAFA;
    }

    .custom-table tr:hover {
        background-color: #F5F5F5;
    }

    /* Number cells right-aligned */
    .custom-table td:nth-child(2),
    .custom-table td:nth-child(3) {
        text-align: right;
        font-family: 'SF Mono', Monaco, 'Courier New', monospace;
    }

    /* Best model highlight */
    .best-model {
        background-color: #E8F5E9 !important;
        font-weight: 600;
        color: #2D5A27;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        color: #6B7280;
        font-size: 0.875rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #E5E7EB;
    }

    /* Spinner styling */
    .stSpinner > div {
        border-color: #2D5A27 !important;
    }
</style>
""", unsafe_allow_html=True)
```

---

## Task 3: Add Feature Importance Extraction Function

**Files:**
- Modify: `app.py` - Add after build_prediction_features function (around line 140)

- [ ] **Step 1: Add function to extract feature importance from Random Forest**

Add this code after the `build_prediction_features` function:

```python
def get_feature_importance(rf_model, crop_columns: list) -> dict:
    """
    Extract and normalize feature importance from Random Forest model.

    Returns a dictionary of feature names to importance values.
    """
    # Get feature importances from RF model
    importances = rf_model.feature_importances_
    feature_names = rf_model.feature_names_in_

    # Create importance dictionary
    importance_dict = dict(zip(feature_names, importances))

    # Group crop columns into single "Crop Type" importance
    crop_importance = sum(
        importance_dict.get(col, 0) for col in crop_columns
    )

    # Build display-friendly importance dict
    display_importance = {
        'Crop Type': crop_importance,
        'Rainfall': importance_dict.get('average_rain_fall_mm_per_year', 0),
        'Temperature': importance_dict.get('avg_temp', 0),
        'Pesticides': importance_dict.get('pesticides_tonnes', 0),
        'Year': importance_dict.get('year_normalized', 0),
    }

    # Normalize to percentages
    total = sum(display_importance.values())
    if total > 0:
        display_importance = {
            k: v / total for k, v in display_importance.items()
        }

    return display_importance
```

---

## Task 4: Update create_prediction_plot to Create Results Table

**Files:**
- Modify: `app.py` - Replace create_prediction_plot function (lines 143-164)

- [ ] **Step 1: Replace create_prediction_plot with results table function**

Replace the existing function with:

```python
def display_prediction_table(y_lr: float, y_rf: float):
    """
    Display prediction results in a clean, styled table.

    Shows both model predictions with percentage comparison to RF.
    """
    # Calculate max value for percentage comparison
    max_pred = max(y_lr, y_rf)
    lr_pct = (y_lr / max_pred) * 100 if max_pred > 0 else 0
    rf_pct = (y_rf / max_pred) * 100 if max_pred > 0 else 0

    # Build HTML table
    table_html = """
    <table class="custom-table">
        <thead>
            <tr>
                <th>Model</th>
                <th>Prediction (kg/ha)</th>
                <th>Relative Score</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Linear Regression</td>
                <td>{:.0f}</td>
                <td>{:.1f}%</td>
            </tr>
            <tr class="best-model">
                <td>Random Forest</td>
                <td>{:.0f}</td>
                <td>{:.1f}%</td>
            </tr>
        </tbody>
    </table>
    """.format(y_lr, lr_pct, y_rf, rf_pct)

    st.markdown(table_html, unsafe_allow_html=True)
```

---

## Task 5: Update Historical Trend Chart (Area Chart)

**Files:**
- Modify: `app.py` - Replace create_trend_plot function (lines 167-185)

- [ ] **Step 1: Replace with modern area chart**

```python
def create_trend_plot(df: pd.DataFrame, crop: str):
    """Create area chart showing historical yield trend for a crop."""
    # Filter for selected crop
    crop_df = df[df['Item'] == crop].copy()
    avg_yield = crop_df.groupby('Year')['kg_per_ha_yield'].mean().reset_index()

    # Prepare data
    years = avg_yield['Year'].values
    yields = avg_yield['kg_per_ha_yield'].values

    # Create figure with modern styling
    fig, ax = plt.subplots(figsize=(12, 6))

    # Set background color
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')

    # Create area chart with gradient effect
    ax.fill_between(years, yields,
                    alpha=0.3,
                    color='#2D5A27',
                    label='Yield Area')

    # Add line on top
    ax.plot(years, yields,
            color='#2D5A27',
            linewidth=2.5,
            marker='o',
            markersize=6,
            markerfacecolor='#FFFFFF',
            markeredgecolor='#2D5A27',
            markeredgewidth=2,
            label='Yield')

    # Styling
    ax.set_title(f'{crop} - Historical Yield Trend in India',
                 fontsize=14,
                 fontweight='bold',
                 color='#1A1A2E',
                 pad=20)

    ax.set_xlabel('Year', fontsize=12, color='#6B7280')
    ax.set_ylabel('Average Yield (kg/ha)', fontsize=12, color='#6B7280')

    # Grid styling
    ax.grid(True, alpha=0.3, linestyle='--', color='#E5E7EB')
    ax.set_axisbelow(True)

    # Spine styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E5E7EB')
    ax.spines['bottom'].set_color('#E5E7EB')

    # Tick styling
    ax.tick_params(colors='#6B7280')
    plt.xticks(rotation=45)

    # Set x-axis limits with padding
    ax.set_xlim(years.min() - 0.5, years.max() + 0.5)

    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    return fig
```

---

## Task 6: Create Feature Importance Chart

**Files:**
- Modify: `app.py` - Add new function after create_trend_plot

- [ ] **Step 1: Add create_feature_importance_chart function**

Add this code after the `create_trend_plot` function:

```python
def create_feature_importance_chart(importance_dict: dict):
    """Create horizontal bar chart showing feature importance."""
    # Sort by importance
    features = list(importance_dict.keys())
    importances = list(importance_dict.values())

    # Sort descending
    sorted_pairs = sorted(zip(importances, features), reverse=True)
    importances, features = zip(*sorted_pairs)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Set background
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#FAFAFA')

    # Color gradient (darker for higher importance)
    colors = ['#2D5A27', '#388E3C', '#4CAF50', '#66BB6A', '#81C784']

    # Create horizontal bars
    bars = ax.barh(features, importances,
                   color=colors[:len(features)],
                   height=0.6,
                   edgecolor='none')

    # Add percentage labels
    for bar, imp in zip(bars, importances):
        width = bar.get_width()
        ax.text(width + 0.01,
                bar.get_y() + bar.get_height() / 2,
                f'{imp * 100:.1f}%',
                va='center',
                ha='left',
                fontsize=10,
                color='#1A1A2E')

    # Styling
    ax.set_xlim(0, max(importances) * 1.15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E5E7EB')
    ax.spines['bottom'].set_color('#E5E7EB')

    ax.set_xlabel('Relative Importance', fontsize=12, color='#6B7280')
    ax.tick_params(colors='#6B7280', labelsize=10)

    # Invert y-axis so highest importance is at top
    ax.invert_yaxis()

    plt.tight_layout()
    return fig
```

---

## Task 7: Update Main Application UI

**Files:**
- Modify: `app.py` - Replace main() function (lines 205-300+)

- [ ] **Step 1: Replace the main() function with updated UI**

Replace the entire `main()` function with this updated version:

```python
def main():
    """Main application function."""

    # Header
    st.title("🌾 Yield Metrics")
    st.markdown("""
    <div style="color: #6B7280; margin-bottom: 1.5rem;">
    Predict crop yields in India based on historical data, weather patterns, and agricultural inputs.
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer
    st.warning("⚠️ **Predictions** are based on historical data and may not reflect current conditions. Use results for guidance only.")

    # Load models
    lr_model, rf_model = load_models()
    if lr_model is None or rf_model is None:
        st.error("Failed to load models. Please ensure training has been completed.")
        return

    # Get available options
    available_crops, min_year, max_year = get_available_options()
    if not available_crops:
        st.error("No data available. Please check data files.")
        return

    dataset_stats = get_dataset_stats()
    crop_columns = get_crop_columns(available_crops)

    # =============================================================================
    # INPUT PARAMETERS SECTION
    # =============================================================================
    st.header("📝 Input Parameters")
    st.caption(f"Available years: {min_year} - {max_year} | Crops: {len(available_crops)}")

    # Create three columns for inputs
    col1, col2, col3 = st.columns(3)

    with col1:
        crop = st.selectbox("🌱 Select Crop", available_crops, help="Choose the crop for yield prediction")

    with col2:
        year = st.number_input("📅 Select Year",
                              min_value=min_year,
                              max_value=max_year,
                              value=max_year,
                              step=1)

    with col3:
        # Pesticide usage
        default_pest = float(dataset_stats.get('pesticide_median', 5000.0))
        min_pest = float(dataset_stats.get('pesticide_min', 0.0))
        max_pest = float(dataset_stats.get('pesticide_max', default_pest * 10))
        step_pest = float(max(1.0, round((max_pest - min_pest) / 200.0)))
        pesticides = st.number_input(
            "🧪 Pesticide (tonnes)",
            min_value=min_pest,
            max_value=max_pest,
            value=default_pest,
            step=step_pest,
            help=f"Range: {min_pest:.0f} - {max_pest:.0f}"
        )

    # Predict button - full width
    st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
    predict_clicked = st.button("🚀 Predict Yield",
                                type="primary",
                                use_container_width=True)

    # =============================================================================
    # PREDICTION RESULTS SECTION
    # =============================================================================
    if predict_clicked:
        # Validate inputs
        is_valid, errors = validate_inputs(crop, year, pesticides, min_year, max_year)
        if not is_valid:
            for error in errors:
                st.error(f"❌ {error}")
            return

        # Show loading state
        with st.spinner("🔄 Analyzing data and making predictions..."):

            # Get historical data for crop-year
            df = load_features_data()
            match = df[(df['Item'] == crop) & (df['Year'] == int(year))]

            if match.empty:
                st.error(f"No data available for {crop} in {year}. Please try a different year.")
                return

            # Use historical rainfall and temperature
            rainfall = float(match['average_rain_fall_mm_per_year'].iloc[0])
            temp = float(match['avg_temp'].iloc[0])

            st.info(f"📊 Using historical data: Rainfall={rainfall:.0f}mm, Temperature={temp:.1f}°C")

            # Build features and predict
            try:
                input_features = build_prediction_features(
                    crop, year, pesticides, rainfall, temp, crop_columns
                )

                # Get predictions
                yield_lr = lr_model.predict(input_features)[0]
                yield_rf = rf_model.predict(input_features)[0]

                # Display success
                st.success("✅ Prediction Complete!")

                # Prediction Results Table
                st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
                st.header("📊 Prediction Results")
                display_prediction_table(yield_lr, yield_rf)

                # Historical Trend Chart
                st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
                st.header("📈 Historical Yield Trend")
                trend_fig = create_trend_plot(df, crop)
                st.pyplot(trend_fig)

                # Feature Importance Chart
                st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
                st.header("💡 Feature Importance")
                importance_dict = get_feature_importance(rf_model, crop_columns)
                importance_fig = create_feature_importance_chart(importance_dict)
                st.pyplot(importance_fig)

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    # =============================================================================
    # FOOTER
    # =============================================================================
    st.markdown("""
    <div class="footer">
        <p>🌾 Yield Metrics | Built with Streamlit</p>
        <p style="font-size: 0.75rem; margin-top: 0.5rem;">
        Model training data: India Crop Yield Dataset (1990-2013)
        </p>
    </div>
    """, unsafe_allow_html=True)
```

---

## Task 8: Test and Verify

- [ ] **Step 1: Run the Streamlit app**

```bash
streamlit run app.py
```

- [ ] **Step 2: Test the following scenarios**

1. Select a crop and year → Click Predict → Verify table displays correctly
2. Verify area chart shows gradient fill
3. Verify feature importance chart displays
4. Check all colors match earthy green palette

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: implement modern minimal UI redesign with earthy green palette"
```

---

## Summary

After completing all tasks, the app will have:

1. ✅ Custom CSS for modern minimal styling
2. ✅ Earthy green color palette (#2D5A27, #7CB342, etc.)
3. ✅ Clean prediction results table with styling
4. ✅ Area chart with gradient fill for historical trends
5. ✅ Horizontal bar chart for feature importance
6. ✅ Consistent spacing and typography
7. ✅ Professional footer

---

## Verification Checklist

- [ ] Page loads with #FAFAFA background
- [ ] Header displays correctly with subtitle
- [ ] Warning banner styled properly
- [ ] Three-column input layout works
- [ ] Prediction button is full-width
- [ ] Results table has styled borders and alternating rows
- [ ] Area chart has green gradient fill
- [ ] Feature importance chart shows horizontal bars
- [ ] Footer displays at bottom of page
- [ ] All visualizations use earthy green palette
