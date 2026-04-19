# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

- **Run app**: `streamlit run app.py`
- **Run tests**: `pytest tests/`
- **Train models**: `python scripts/train_models_v2.py`

## High-Level Architecture

**Streamlit web app** — `app.py` is main entry point. Serves crop yield predictions with two model tiers:
- **v1 models**: Linear Regression + Random Forest (legacy)
- **v2/champion model**: Advanced Random Forest with satellite/soil/economic features

**Modular utilities** in `utils/`:
- `data_loader.py` — loads models and feature data
- `predictor.py` — builds features, runs predictions
- `visualizations.py` — renders charts

**Pipeline scripts** in `scripts/`:
- `config.py` — paths and feature schemas
- `feature_engineer_v2.py` — creates V2 features (NDVI, soil pH, heat stress, MSP)
- `train_models_v2.py` — trains champion model

**Data**:
- `data/raw/` — source CSVs
- `data/processed/Feature_Engineered_Crop_Yield_Data.csv` — feature-engineered dataset
- `models/` — trained `.pkl` files

**Models** expect features in schema defined by `scripts/config.py`:
- `CORE_FEATURES`: rainfall, temp, pesticides
- `V2_FEATURES`: heat stress, drought, NDVI, soil metrics
- `ENGINEERED_FEATURES`: polynomial/interaction terms

### Auto-Update Memory (MANDATORY)

**Update memory files AS YOU GO, not at the end.** When you learn something new, update immediately.

| Trigger | Action |
|---------|--------|
| User shares a fact about themselves | → Update `memory-profile.md` |
| User states a preference | → Update `memory-preferences.md` |
| A decision is made | → Update `memory-decisions.md` with date |
| Completing substantive work | → Add to `memory-sessions.md` |

**Skip:** Quick factual questions, trivial tasks with no new info.

**DO NOT ASK. Just update the files when you learn something.**