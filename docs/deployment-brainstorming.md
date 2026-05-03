# BRAINSTORMING: Solving the Large Model Production Gap

## 1. The Core Challenge
The project has successfully moved from "Static Predictions" to "Responsive Forecasting" locally. However, the resulting Random Forest model is **235 MB**, which exceeds GitHub's **100 MB** file limit. Consequently:
*   **Production (Streamlit Cloud)** is running outdated/mismatched model artifacts.
*   **User Experience** in production remains static/inconsistent despite codebase fixes.

## 2. Proposed Strategic Approach: "Hybrid Deployment & Metadata Contract"
We will shift from a "Push-All-to-Git" approach to a **Decoupled Architecture**.

### Why this approach?
1.  **Safety**: It prevents accidental repository bloat.
2.  **Consistency**: It ensures the production app *must* match the local `feature_columns.json` metadata.
3.  **Scalability**: If the model grows to 1GB tomorrow, the system won't break.

## 3. Deployment Alternatives for Large Models
Since Streamlit Cloud relies on GitHub, we need to bypass the file size limit. Here are the considered paths:

### Path A: The Cloud Storage Bridge (Recommended)
Store the `model.pkl` on **Google Drive** or **AWS S3**.
*   **How it works**: `utils/data_loader.py` is updated with a `check_and_download()` function. If the model isn't found locally (common in a fresh Cloud container), it pulls it from a persistent URL.
*   **Production Impact**: High reliability, zero GitHub size issues.

### Path B: Git LFS (Git Large File Storage)
Install Git LFS locally to track `.pkl` files.
*   **How it works**: Git stores a "pointer" file in the repo; the actual data lives on GitHub's LFS servers.
*   **Production Impact**: Streamlit Cloud natively supports Git LFS, so it "just works."

### Path C: Model Distillation/Compression
Use `joblib` with high compression (`compress=9`) or switch to **XGBoost/LightGBM** which produce much smaller model files for the same accuracy.
*   **How it works**: Retrain with a more compact algorithm.
*   **Production Impact**: Small file size (~20-50MB), allows direct GitHub push.

## 4. Architectural Metric: "The 100MB Boundary"
We will use the **100MB file limit** as a development trigger:
*   **If < 100MB**: Direct Git commit.
*   **If > 100MB**: Mandatory use of **Path A** or **Path B**.

## 5. Implementation Roadmap
1.  **Compression Test**: Attempt to save the current model with `compress=9`.
2.  **LFS Setup**: If compression fails to reach < 100MB, initialize Git LFS.
3.  **Loader Update**: Update `data_loader.py` to handle missing artifacts gracefully with helpful UI prompts for the user.
