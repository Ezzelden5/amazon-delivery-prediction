
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
from datetime import datetime



st.set_page_config(
    page_title="Amazon Delivery time prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)



@st.cache_resource
def load_pipeline():
    try:
        return joblib.load('final_model_delivery_pipeline(grid3).pkl')
    except Exception:
        return None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('cleaned_amazon_delivery.csv')
        return df
    except Exception:
        return pd.DataFrame()

pipeline = load_pipeline()
df = load_data()


st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=150)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page:", [
    "1. Business Case & Data",
    "2. Data Insights (EDA)",
    "3. Delivery Predictor"
])


if page == "1. Business Case & Data":

    st.title("Amazon Delivery Time Prediction")
    st.divider()

    st.subheader("The Business Problem")
    st.write("""
    In supply chain management, Last-Mile Delivery refers to the final step of the shipping process, where a package travels from a local hub to the customer's front door.
    This short final leg accounts for up to 53% of total shipping costs.

    Unpredictable delivery times lead to frustrated customers, wasted driver hours, and increased fuel costs.
    Our goal is to accurately predict delivery so dispatchers can optimize routes and customers are kept informed.
    """)

    st.subheader("Data info")


    st.markdown("""
    **Every day, millions of orders are delivered worldwide.**
     Customers expect fast and accurate delivery times.
     When estimated time become wrong, companies face:
     - Angry customers will have bad reviews and refunds
     - Wasted driver time and fuel
     - Poor planning during rain, traffic jams, or peak hours

     **Business impact:**
     - Give customers accurate estimated time he will have a higher satisfaction
     - Help managers assign the right driver/vehicle
     - Reduce delays in bad weather or heavy traffic
    - Save money on operations and improve ratings
    Delivery_Time: The total time (in minutes) it took for the package to go from the store/hub to the customer's front door
     Agent_Age: The physical age of the delivery courier
    Agent_Rating: The average rating (out of 5.0) given to the driver by previous customers
    distance_km: The physical distance between the store and the customer
    Area: The density of the delivery zone
    Traffic: The traffic severity
    Weather: The weather condition
    Vehicle: The mode of transport used by the driver
    Category: The type of product being delivered
    pickup_delay_min: The preparation time. The delay between when the order was placed and when the driver picked it up
    hour_of_order: The hour of the day when the order was placed (0-23)
    day_of_week: The day of the week when the order was placed (0=Monday, 6=Sunday)
    is_weekend: Whether the order was placed on a weekend (1) or a weekday (0)
    is_night_order: Whether the order was placed during night hours (1) or day hours (0)
        """, unsafe_allow_html=True)



elif page == "2. Data Insights (EDA)":

    st.title("Logistical Analysis")

    st.subheader("Q1: How much do Weather and Traffic compound delivery delays?")
    avg_weather_traffic = df.groupby(['Traffic', 'Weather'])['Delivery_Time'].mean().reset_index()
    fig1 = px.bar(
        avg_weather_traffic,
        x='Traffic',
        y='Delivery_Time',
        color='Weather',
        barmode='group',
        title="Average Delivery Time by Traffic & Weather",
        category_orders={'Traffic': ['Low', 'Medium', 'High', 'Jam']}
    )
    fig1.update_yaxes(title="Avg Time (Minutes)")
    st.plotly_chart(fig1, use_container_width=True)

    st.divider()

    st.subheader("3. Impact of Age and Rating on Long-Distance Deliveries")


    long_distance = df[df['distance_km'] > 8].copy()

    long_distance['Rating_Category'] = long_distance['Agent_Rating'].apply(
        lambda x: 'Top Rated (4.5 - 5.0)' if x >= 4.5 else 'Average/Low (< 4.5)')
    avg_data = long_distance.groupby(['Agent_Age', 'Rating_Category'])['Delivery_Time'].mean().reset_index()

    fig_long_distance = px.bar(
        avg_data,
        x='Agent_Age',
        y='Delivery_Time',
        color='Rating_Category',
        barmode='group', 
        title='Average Delivery Time by Age and Rating (> 8 km)',
        labels={
            'Agent_Age': 'Agent Age', 
            'Delivery_Time': 'Avg Delivery Time (Minutes)', 
            'Rating_Category': 'Driver Rating'
        },
    color_discrete_sequence=['#232F3E', '#FF9900'] 
        )

    st.plotly_chart(fig_long_distance, use_container_width=True)

    st.divider()

    col_1, col_2 = st.columns(2)
    with col_1:
        st.subheader("Q2: Which city zones struggle the most?")
        avg_area = df.groupby('Area')['Delivery_Time'].mean().reset_index().sort_values('Delivery_Time', ascending=False)
        fig2 = px.bar(
            avg_area,
            x='Area',
            y='Delivery_Time',
            color='Area',
            title='Avg Delivery Time by Area',
            text_auto='.1f'
        )
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col_2:
        st.subheader("Q3: Are we using the right vehicles?")
        fig3 = px.histogram(
            df,
            x='Vehicle',
            y='Delivery_Time',
            color='Vehicle',
            title='Delivery Time Variance by Vehicle'
        )
        fig3.update_layout(showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)


elif page == "3. Delivery Predictor":

    st.title("Delivery Prediction")

    input_col1, input_col2, input_col3 = st.columns(3)

    with input_col1:
        st.subheader("Driver and Order Info")
        agent_age = st.slider("Agent Age", min_value=18, max_value=60, value=30)
        agent_rating = st.slider("Agent Rating", min_value=1.0, max_value=5.0, value=4.5, step=0.1)
        category = st.selectbox("Product Category", ['Clothing', 'Electronics', 'Sports', 'Cosmetics', 'Toys', 'Snacks', 'Shoes', 'Apparel', 'Jewelry', 'Outdoors', 'Grocery', 'Books', 'Kitchen', 'Home', 'Pet Supplies', 'Skincare'])
        prep_time_mins = st.number_input("Prep Delay (mins)", min_value=0, max_value=120, value=15)

    with input_col2:
        st.subheader("Logistics and Environment")
        distance_km = st.number_input("Delivery Distance (km)", min_value=0.1, max_value=50.0, value=5.0, step=0.5)
        vehicle = st.selectbox("Vehicle Type", ['motorcycle', 'scooter', 'van', 'bicycle'])
        area = st.selectbox("City Area", ['Metropolitian', 'Urban', 'Semi-Urban', 'Other'])
        weather = st.selectbox("Weather Condition", ['Sunny', 'Cloudy', 'Windy', 'Fog', 'Sandstorms', 'Stormy'])
        traffic = st.selectbox("Traffic Condition", ['Low', 'Medium', 'High', 'Jam'])

    with input_col3:
        st.subheader("Time")
        order_time = st.time_input("Order Time", value=datetime.strptime("14:30", "%H:%M").time())
        order_day = st.selectbox("Day of the Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        is_night_input = st.selectbox("Is this a Night Order?", ["No", "Yes"])

    day_mapping = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    is_night_val = 1 if is_night_input == "Yes" else 0

    st.divider()

    if pipeline is None:
        st.error("error")
    else:
        if st.button("Predict", use_container_width=True, type="primary"):

            input_data = {
                'Agent_Age': [agent_age],
                'Agent_Rating': [agent_rating],
                'Weather': [weather],
                'Traffic': [traffic],
                'Vehicle': [vehicle],
                'Area': [area],
                'Category': [category],
                'distance_km': [distance_km],
                'pickup_delay_min': [prep_time_mins],
                'hour_of_order': [order_time.hour],
                'day_of_week': [day_mapping[order_day]],
                'is_weekend': [1 if day_mapping[order_day] >= 5 else 0],
                'is_night_order': [is_night_val]
            }

            input_df = pd.DataFrame(input_data)

            try:
                prediction = pipeline.predict(input_df)[0]
                st.success("Time prediction")
                st.metric(label="Estimated Total Delivery Time", value=f"{int(np.round(prediction))} Minutes")
            except Exception as e:
                st.error(f"Prediction Error: {e}")
