import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Page title
st.set_page_config(page_title="Toronto Ferry Analytics", layout="wide")

st.title("🚢 Real-Time Ferry Ticket Sales & Redemption Analytics")

# Load data
df = pd.read_csv("data/Toronto Island Ferry Tickets.csv")

model = joblib.load("ferry_ticket_model.pkl")

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

df["Hour"] = df["Timestamp"].dt.hour
df["Month"] = df["Timestamp"].dt.month
df["Day"] = df["Timestamp"].dt.day
df["Day_of_Week"] = df["Timestamp"].dt.day_name()

# Sidebar
st.sidebar.title("Dashboard Filters")

selected_month = st.sidebar.selectbox(
    "Select Month",
    ["All"] + sorted(df["Month"].unique().tolist())
)

if selected_month != "All":
    df = df[df["Month"] == selected_month]

# KPI
total_sales = df["Sales Count"].sum()
total_redemption = df["Redemption Count"].sum()
net = total_sales - total_redemption

col1, col2, col3 = st.columns(3)

col1.metric("Total Sales", f"{total_sales:,}")
col2.metric("Total Redemption", f"{total_redemption:,}")
col3.metric("Net Passenger Movement", f"{net:,}")

st.divider()

# Hourly Sales
hourly_sales = df.groupby("Hour")["Sales Count"].sum()

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(hourly_sales.index, hourly_sales.values, marker="o")
ax.set_title("Hourly Ticket Sales")
ax.set_xlabel("Hour")
ax.set_ylabel("Sales")

st.pyplot(fig)

# Hourly Redemption
hourly_redemption = df.groupby("Hour")["Redemption Count"].sum()

fig, ax = plt.subplots(figsize=(10,4))
ax.plot(hourly_redemption.index, hourly_redemption.values, marker="o")

ax.set_title("Hourly Ticket Redemption")
ax.set_xlabel("Hour")
ax.set_ylabel("Redemption")

st.pyplot(fig)

# Monthly Sales
monthly_sales = df.groupby("Month")["Sales Count"].sum()

fig, ax = plt.subplots(figsize=(10,4))
ax.bar(monthly_sales.index, monthly_sales.values)

ax.set_title("Monthly Ticket Sales")

st.pyplot(fig)

#Monthly Redemption graph

monthly_redemption = df.groupby("Month")["Redemption Count"].sum()

fig, ax = plt.subplots(figsize=(10,4))
ax.bar(monthly_redemption.index, monthly_redemption.values)

ax.set_title("Monthly Ticket Redemption")

st.pyplot(fig)

#Day of Week Sales

day_sales = df.groupby("Day_of_Week")["Sales Count"].sum()

order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

day_sales = day_sales.reindex(order)

fig, ax = plt.subplots(figsize=(10,4))
ax.bar(day_sales.index, day_sales.values)

ax.set_title("Sales by Day of Week")
plt.xticks(rotation=45)

st.pyplot(fig)

#Day of Week Redemption

day_redemption = df.groupby("Day_of_Week")["Redemption Count"].sum()

day_redemption = day_redemption.reindex(order)

fig, ax = plt.subplots(figsize=(10,4))
ax.bar(day_redemption.index, day_redemption.values)

ax.set_title("Redemption by Day of Week")
plt.xticks(rotation=45)

st.pyplot(fig)

st.subheader("Dataset Preview")

st.dataframe(df.head(20))

csv = df.to_csv(index=False)

st.download_button(
    "Download Dataset",
    csv,
    "Toronto_Ferry_Data.csv",
    "text/csv"
)

st.header("🔮 Ticket Sales Prediction")

year = st.number_input("Year", min_value=2015, max_value=2035, value=2025)
month = st.slider("Month", 1, 12, 1)
day = st.slider("Day", 1, 31, 1)
hour = st.slider("Hour", 0, 23, 12)
minute = st.selectbox("Minute", [0, 15, 30, 45])

day_name = st.selectbox(
    "Day of Week",
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
)

day_mapping = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6
}

day_encoded = day_mapping[day_name]

if st.button("Predict Ticket Sales"):
    input_data = pd.DataFrame({
        "Year": [year],
        "Month": [month],
        "Day": [day],
        "Hour": [hour],
        "Minute": [minute],
        "Day_of_Week": [day_encoded]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Ticket Sales: {prediction[0]:.0f}")