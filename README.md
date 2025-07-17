# 🌾 Crop Yield Prediction 

A machine learning-based system to predict crop yields in India using historical agricultural data and climate patterns such as rainfall, temperature, and pesticide usage.

---

## 🚀 Project Overview

This project aims to provide region-specific crop yield forecasts to assist farmers, researchers, and policymakers in making data-driven agricultural decisions.

---

## 🎯 Problem Statement

Accurate crop yield prediction is essential for optimizing agricultural output and resource allocation. This project uses machine learning to analyze historical climate and yield data to forecast crop productivity, supporting proactive planning in Indian agriculture.

---

## 🧠 Features

- ✅ Predicts crop yield based on:
  - Crop type
  - Year (optional, defaults to current)
  - Pesticide usage
  - Automatically fetched rainfall and temperature data via APIs

- 📊 Visualizes predictions and trends using Matplotlib and Seaborn
- 🖥️ Web-based UI for easy input and results display using Flask

---

## 💡 Inputs & Outputs

### **User Inputs**
- `Crop` 
- `Year` (optional)
- `Pesticide Usage`

### **Backend-Automated Inputs (via API)**
- `Average Rainfall (mm)`
- `Average Temperature (°C)`

### **Outputs**
- Predicted crop yield in **kg/ha** or **hg/ha**
- **Line/Bar** plot of predicted yield trends
- **Heatmap** or distribution plot of influencing factors

---

## 🧱 Tech Stack

| Layer       | Tools/Frameworks                                |
|-------------|--------------------------------------------------|
| Language    | **Python**                                           |
| Backend     | **Flask**                                            |
| ML Modeling |**Scikit-learn, Linear Regression, Random Forest,  pandas, numpy**            |
| Visualization | **Matplotlib, Seaborn**                           |
| Frontend UI | **HTML/CSS (via Flask templates)**                   |

---
