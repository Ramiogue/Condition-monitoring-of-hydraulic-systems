# 🔧 Condition Monitoring of Hydraulic Systems

[![Streamlit App](https://img.shields.io/badge/Live_App-Click_Here-green)](https://condition-monitoring-of-hydraulic-systems-mh3smqbwswqegvwymtng.streamlit.app/)

This project uses machine learning to perform condition monitoring on hydraulic system components using multivariate sensor data. A trained classifier detects degradation in the **cooler, valve, pump, and accumulator**, and generates interpretable maintenance alerts via a Streamlit web app.

> 📘 Inspired by:  
> Kleiner, M. et al. (2022). _Condition Monitoring of Hydraulic Systems Using Machine Learning Algorithms_  
> Energies, 15(17), 6217. https://doi.org/10.3390/en15176217

---

## 🎯 Objective

- ✅ Predict health conditions of 4 hydraulic components  
- ✅ Analyze multichannel time-series data from a hydraulic test rig  
- ✅ Enable early maintenance action using machine learning  
- ✅ Provide human-readable maintenance suggestions  
- ✅ Deploy as an interactive web app via Streamlit Cloud

---

## 📚 Dataset Summary

- **Source**: Kaggle  
- **System**: Hydraulic test rig with primary and secondary circuits  
- **Total cycles**: 2205  
- **Cycle duration**: 60 seconds  
- **Measurements**:
  - 8 sensors @ 1 Hz (60 values × 8)
  - 2 sensors @ 10 Hz (600 values × 2)
  - 7 sensors @ 100 Hz (6000 values × 7)
- **Total attributes per sample**: 43,680  
- **Target labels** (from `profile.txt`):
  - `cooler_condition`
  - `valve_condition`
  - `pump_leakage`
  - `accumulator_pressure`

> 🧪 The system repeats 60s load cycles while recording process variables and varying the physical condition of each component.

---

## 🛠️ Approach

### Data Preprocessing
- Loaded 16 raw sensor files (excluding PS4)
- Merged with target condition labels
- Filtered stable samples only (`stable_flag = 1`)

### Feature Engineering
- Extracted **16 features** per sensor:
  - `mean`, `std`, `min`, `max`, `median`, `ptp`
  - `skewness`, `kurtosis`, `sum`, `abs energy`
  - `first location of min`, `first location of max`
  - `last location of min`, `last location of max`
  - `absolute sum of changes`, `mean absolute change`
- Total: `16 sensors × 16 features = 256 features`

### Model
- Multi-label classification using:
  - `MultiOutputClassifier(RandomForestClassifier)`
- Trained using 5-fold cross-validation

---

## ✅ Model Performance

### 🎯 Final Test Set – Target: `valve_condition`

```text
              precision    recall  f1-score   support

          73       0.99      1.00      0.99        75
          80       1.00      1.00      1.00        57
          90       1.00      0.99      0.99        75
         100       1.00      1.00      1.00       218

    accuracy                           1.00       425
   macro avg       1.00      1.00      1.00       425
weighted avg       1.00      1.00      1.00       425
```

📊 **More target results available in the Colab notebook**  
🔗 [Open Full Evaluation in Google Colab](https://colab.research.google.com/drive/15a3CKbxCV-GtqiFkTI_Ky-AZ7JnQz2KF)

---

## 🌐 App Features

- 📁 Upload raw sensor data file (`.txt` or `.csv`, tab-separated, 16 columns)  
- 🧠 Performs feature extraction internally  
- 🎯 Predicts health of all 4 components  
- 📋 Provides maintenance alerts like:

```text
🔴 The valve is failing – urgent attention required  
🟡 The pump shows weak leakage – schedule inspection  
🔴 The accumulator pressure is low – recharge soon  
```

---

## 📸 App Screenshots

Below are example outputs of the deployed Streamlit app:

![Screenshot 1](https://github.com/user-attachments/assets/92023474-01c9-4ca2-a8f8-807db9f50520)  
![Screenshot 2](https://github.com/user-attachments/assets/76af9418-eb4e-4212-b2f3-3bfac7d20a27)

---

## 📄 Documents & How to Run This Project Yourself

### 🧪 Train Your Own Model from Scratch (Locally or in Colab)
1. Download the dataset from Kaggle or the [original paper](https://www.mdpi.com/1996-1073/15/17/6217).
2. Use the [Colab notebook](https://colab.research.google.com/drive/15a3CKbxCV-GtqiFkTI_Ky-AZ7JnQz2KF) to:
   - Load all 16 sensor files and `profile.txt`
   - Preprocess data (`stable_flag = 1`)
   - Extract statistical features
   - Train a `MultiOutputClassifier(RandomForestClassifier)`
   - Save your model using `joblib.dump(model, "model.pkl")`

---

### 🚀 Run the Streamlit App Locally
1. Clone this repo  
   ```bash
   git clone https://github.com/yourusername/hydraulic-monitoring.git
   cd hydraulic-monitoring
   ```

2. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app  
   ```bash
   streamlit run app.py
   ```

---

## 🧪 Simulate Sensor Input

Use this script to simulate a valid `.txt` input for testing:

```python
import numpy as np, pandas as pd
data = np.random.normal(0, 1, size=(600, 16))  # 600 time steps × 16 sensors
pd.DataFrame(data).to_csv("test_sensor_data.txt", sep="\t", index=False, header=False)
```

---

## ⚙️ Tech Stack

| Tool/Library       | Purpose                           |
|--------------------|-----------------------------------|
| Python 3.9         | Core language                     |
| Pandas / NumPy     | Data processing                   |
| Scikit-learn       | Model training (Random Forest)    |
| Streamlit          | Web app deployment                |
| Joblib             | Model serialization               |
| Google Colab       | Training and experimentation      |
| GitHub             | Version control                   |
| Streamlit Cloud    | Live deployment hosting           |

---

## 🤖 Acknowledgements

- **Kleiner et al. (2022)** – for the hydraulic monitoring dataset and methodology  
- **ChatGPT by OpenAI** – for assisting with preprocessing logic, evaluation pipeline, model building, Streamlit deployment, and documentation  
- **Kaggle Community** – for providing high-quality sensor data for reproducible research

---

## 👤 Author

**Name**: Tshepang Ramaoka  
📧 **Email**: ramaokafelicia@gmail.com  
🌍 **Location**: South Africa  
🔗 **Colab Notebook**: [Open here](https://colab.research.google.com/drive/15a3CKbxCV-GtqiFkTI_Ky-AZ7JnQz2KF)

---

## 🚀 Future Enhancements

- Add FastAPI REST endpoint for API-based predictions  
- Connect to real-time sensor data streams via MQTT or Kafka  
- Add support for model retraining with new incoming data  
- Explore LSTM or hybrid deep learning models for temporal learning  
- Create a dashboard for monitoring historical predictions and alerts

---





