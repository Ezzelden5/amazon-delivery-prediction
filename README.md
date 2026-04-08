# 🚚 Amazon Delivery Time Prediction

A complete **Data Science & Machine Learning** project that predicts delivery times (in minutes) for Amazon's last-mile logistics, deployed as an interactive **Streamlit web application**.

---

## 📌 Project Overview

**Business Problem:**  
Customers want to know *when exactly* their order will arrive. By predicting delivery times accurately, logistics companies can set realistic expectations, optimize routes, and reduce complaints caused by late deliveries.

**Solution:**  
A machine learning model (XGBoost) trained on 43,500+ delivery records that predicts delivery time based on agent info, distance, weather, traffic, vehicle type, and product category.

---

## 📊 Dataset

- **Source:** [Amazon Delivery Dataset — Kaggle](https://www.kaggle.com/datasets/sujalsuthar/amazon-delivery-dataset)
- **Rows:** 43,739 (raw) → 43,523 (after cleaning)
- **Columns:** 16 (original) → 14 features used in model
- **Target Variable:** `Delivery_Time` (in minutes)

### Key Features:
| Feature | Type | Description |
|---|---|---|
| Agent_Age | Numerical | Age of delivery agent |
| Agent_Rating | Numerical | Rating of delivery agent (1-5) |
| distance_km | Engineered | Haversine distance from store to customer |
| pickup_delay_min | Engineered | Minutes between order and pickup |
| hour_of_order | Engineered | Hour when order was placed |
| day_of_week | Engineered | Day (0=Mon, 6=Sun) |
| is_weekend | Engineered | Binary: weekend or not |
| is_night_order | Engineered | Binary: after 8pm or not |
| month | Engineered | Month of the order |
| Weather | Categorical | Sunny, Cloudy, Fog, Stormy, etc. |
| Traffic | Categorical | Low, Medium, High, Jam |
| Vehicle | Categorical | motorcycle, scooter, van, bicycle |
| Area | Categorical | Metropolitan, Urban, Semi-Urban, Other |
| Category | Categorical | Product category (16 types) |

---

## 🔬 Project Pipeline

### 1. Data Cleaning
- Stripped whitespace from categorical columns
- Converted 'NaN' strings to actual null values
- Filled missing values (median for ratings, mode for weather/traffic)
- Capped Agent_Rating outliers (6.0 → 5.0)
- Removed agents under 18 years old
- Removed impossible GPS coordinates (distance > 100 km)
- Fixed midnight crossover in pickup delay calculation
- Fixed typo: "Metropolitian"

### 2. Feature Engineering (7 new features)
- `distance_km` — Haversine formula from GPS coordinates
- `pickup_delay_min` — Time between order and pickup
- `hour_of_order` — Extracted from Order_Time
- `day_of_week` — Extracted from Order_Date
- `is_weekend` — Binary flag
- `is_night_order` — Binary flag (after 8pm)
- `month` — Extracted from Order_Date

### 3. Exploratory Data Analysis
- 10+ plots including histograms, boxplots, bar charts, scatter plots, heatmaps, violin plots
- Statistical analysis of delivery time by traffic, weather, area, vehicle
- Temporal trends (hourly, daily patterns)

### 4. Model Building (9 algorithms compared)
| Model | Test R² (%) | Test MAE (min) |
|---|---|---|
| LightGBM | 81.98 | 17.17 |
| Random Forest | 81.76 | 17.14 |
| **XGBoost (Final)** | **81.88** | **17.30** |
| CatBoost | 81.58 | 17.41 |
| Decision Tree | 81.28 | 17.38 |
| Linear Regression | 56.20 | 27.02 |
| Ridge | 56.20 | 27.02 |
| Lasso | 55.20 | 27.29 |
| ElasticNet | 33.71 | 32.60 |

### 5. Hyperparameter Tuning
- **Method:** RandomizedSearchCV (5-fold CV)
- **Best Parameters:** n_estimators=200, max_depth=9, learning_rate=0.05, subsample=0.8
- **Best R²:** 81.875%

### 6. Deployment
- **Streamlit** multi-page web app with:
  - 🏠 Home page — Project overview
  - 🔮 Prediction page — Interactive delivery time predictor
  - 📊 Dashboard — Analytics with filters
  - 🤖 Model Info — Algorithm details and evaluation

---

## 🚀 How to Run

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run the Streamlit App
```bash
streamlit run app.py
```

### Run the Notebook
Open `amazon_delivery_time.ipynb` in Jupyter Notebook or Google Colab.

---

## 📁 Project Structure
```
amazon-delivery-prediction/
│
├── amazon_delivery.csv                    # Dataset
├── amazon_delivery_time.ipynb             # Full analysis notebook
├── app.py                                 # Streamlit deployment app
├── final_model_delivery_pipeline.pkl      # Saved model pipeline
├── requirements.txt                       # Python dependencies
├── README.md                              # This file
```

---

## 🛠️ Technologies Used
- **Python** 3.9+
- **Pandas** & **NumPy** — Data manipulation
- **Plotly** & **Matplotlib** & **Seaborn** — Visualization
- **Scikit-learn** — Preprocessing, pipeline, evaluation
- **XGBoost** — Final model
- **LightGBM** & **CatBoost** — Model comparison
- **Geopy** — Distance calculation
- **Streamlit** — Web deployment
- **Joblib** — Model serialization

---

## 📈 Key Results
- **Best Model:** XGBoost (R² = 81.9%)
- **Average Prediction Error:** ~17 minutes
- **Top Feature:** Product Category (29% importance)
- **5-Fold CV:** Stable performance across all folds

---

## 👤 Author
[Your Name]

---

## 📄 License
This project is for educational purposes.

---

## 🙏 Acknowledgements
- [Amazon Delivery Dataset — Kaggle](https://www.kaggle.com/datasets/sujalsuthar/amazon-delivery-dataset)
- [Epsilon AI](https://github.com/Epsilon-AI) — Course materials and guidance
