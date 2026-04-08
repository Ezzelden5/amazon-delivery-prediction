import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import joblib

# Load cleaned data
df = pd.read_csv('cleaned_amazon_delivery.csv')

# Drop unnecessary columns
df_ml = df.drop(columns=['Order_ID', 'Order_Date', 'Order_Time', 'Pickup_Time', 'Order_Time_parsed', 'Pickup_Time_parsed', 'Store_Latitude', 'Store_Longitude', 'Drop_Latitude', 'Drop_Longitude', 'Store_Point', 'Drop_Point'])

X = df_ml.drop('Delivery_Time', axis=1)
y = df_ml['Delivery_Time']

# Define preprocessing
ord_encod_traffic = ['Low', 'Medium', 'High', 'Jam']
ord_encod_weather = ['Sunny', 'Windy', 'Sandstorms', 'Cloudy', 'Fog', 'Stormy']
ord_encod_vehicle = ['bicycle', 'scooter', 'motorcycle', 'van']
ord_encod_area = ['Other', 'Semi-Urban', 'Urban', 'Metropolitian']

scaler_col = X.select_dtypes(include='number').columns

preprocessing = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), scaler_col),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['Category']),
        ('ord', OrdinalEncoder(categories=[ord_encod_traffic, ord_encod_weather, ord_encod_vehicle, ord_encod_area]), ['Traffic', 'Weather', 'Vehicle', 'Area'])
    ],
    remainder='passthrough'
)

# Pipeline
xgb_pipeline = Pipeline(steps=[
    ('Preprocessing', preprocessing),
    ('Model', XGBRegressor(n_estimators=150, max_depth=7, learning_rate=0.1, random_state=42))
])

# Fit the model
xgb_pipeline.fit(X, y)

print("Model trained successfully.")

# Save the pipeline
best_final_pipeline = xgb_pipeline
joblib.dump(best_final_pipeline, 'final_model_delivery_pipeline(grid3).pkl')

print("Model saved successfully.")