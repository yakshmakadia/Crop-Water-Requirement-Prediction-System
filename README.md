#  ML-Based Crop Water Requirement Prediction System

> Intelligent irrigation prediction for Rajkot, Gujarat using LSTM deep learning.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

##  Project Overview

With growing water scarcity and unpredictable monsoons, especially in Gujarat, there is a critical need for precise irrigation planning.  
This project uses an **LSTM-based deep learning model** trained on historical climate, soil, and crop data to predict daily crop water requirements for **cotton, wheat, and groundnut** in Rajkot.

The project aligns with:
- **SDG 13: Climate Action**
- **SDG 2: Zero Hunger**
- **SDG 6: Clean Water and Sanitation**

---

##  Tech Stack

- Python, Jupyter Notebook
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn

---

##  Model Highlights

- Predicts soil moisture & irrigation needs
- Captures seasonal variability & monsoon anomalies
- Mean Absolute Error (MAE): **0.2008**
- RMSE: **0.2983**

---

##  Key Results

- Improved irrigation planning
- Lower water wastage
- Adaptation to climate variability in semi-arid regions

---

##  Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
