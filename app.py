# 1. IMPORTS
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sqlalchemy import create_engine
import os
from openai import OpenAI

# 2. CONFIG
st.set_page_config(page_title="Sales Advisor AI", layout="wide")
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])


# 3. LOAD DATA
@st.cache_data
def load_data():
    df = pd.read_csv("sample_data/Business_sales_EDA.csv")
    df = df.dropna()
    return df


df = load_data()


# FEATURE ENGINEERING
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Month"] = df["Order_Date"].dt.month

categorical_cols = ["Region", "Category", "Sub_Category", "Product_Name"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

target = "Profit"

drop_cols = ["Order_ID", "Customer_Name", "Order_Date", "Country", "Revenue", "Profit"]

feature_cols = [col for col in df_encoded.columns if col not in drop_cols]

X = df_encoded[feature_cols]
y = df_encoded[target]


# TRAIN MODELS
@st.cache_resource
def train_models():
    rf = RandomForestRegressor(n_estimators=120, random_state=42)
    xgb = XGBRegressor(n_estimators=120, random_state=42)

    rf.fit(X, y)
    xgb.fit(X, y)

    return rf, xgb

rf_model, xgb_model = train_models()


def predict_profit(input_df):
    return (rf_model.predict(input_df)[0] + xgb_model.predict(input_df)[0]) / 2


# PRESCRIPTIVE ENGINE
def optimize_strategy(base_input):

    results = []

    for discount in np.arange(0, 0.5, 0.05):
        for price_factor in np.arange(0.8, 1.3, 0.1):

            temp = base_input.copy()
            temp["Discount"] = discount
            temp["Unit_Price"] = base_input["Unit_Price"] * price_factor

            input_df = pd.DataFrame([temp])
            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=X.columns, fill_value=0)

            profit = predict_profit(input_df)
            revenue = temp["Unit_Price"] * temp["Quantity"] * (1 - discount)

            results.append({
                "discount": discount,
                "price": temp["Unit_Price"],
                "profit": profit,
                "revenue": revenue
            })

    return pd.DataFrame(results)


# UI
st.title("📊 AI Sales Optimization Dashboard")

col1, col2 = st.columns(2)

# -------- INPUT PANEL --------
with col1:
    st.subheader("🔧 Define Scenario")

    region = st.selectbox("Region", df["Region"].unique())
    product = st.selectbox("Product", df["Product_Name"].unique())

    quantity = st.slider("Quantity", 1, 20, 5)
    price = st.slider("Unit Price", 10, 500, 100)
    discount = st.slider("Discount (%)", 0, 50, 10) / 100

    query = st.text_area("💬 Ask",
                         "How can I maximize profit for this scenario?")


# -------- BASE INPUT --------
base_input = {"Quantity": quantity, "Unit_Price": price, "Discount": discount, "Month": 6}

# Add categorical selections
base_input[f"Region_{region}"] = 1
base_input[f"Product_Name_{product}"] = 1


# REAL-TIME COMPUTATION
results_df = optimize_strategy(base_input)

best = results_df.loc[results_df["profit"].idxmax()]


# OUTPUT PANEL
with col2:
    st.subheader("📈 Best Strategy")

    st.metric("Optimal Discount", f"{round(best['discount']*100,1)}%")
    st.metric("Optimal Price", f"${round(best['price'],2)}")
    st.metric("Expected Profit", f"${int(best['profit'])}")


# TRADEOFF CHART
st.subheader("📊 Profit vs Revenue Tradeoff")

fig, ax = plt.subplots()

ax.scatter(results_df["revenue"], results_df["profit"])
ax.set_xlabel("Revenue")
ax.set_ylabel("Profit")
ax.set_title("Tradeoff Curve")

st.pyplot(fig)


# REGION-WISE RECOMMENDATIONS
st.subheader("🌍 Region-wise Strategy Comparison")

region_results = []

for r in df["Region"].unique():

    temp_input = base_input.copy()
    temp_input = {k:0 for k in X.columns}

    temp_input["Quantity"] = quantity
    temp_input["Unit_Price"] = price
    temp_input["Discount"] = discount
    temp_input[f"Region_{r}"] = 1

    temp_input = pd.DataFrame([temp_input])

    profit = predict_profit(temp_input)

    region_results.append({
        "Region": r,
        "Expected Profit": int(profit)
    })

region_df = pd.DataFrame(region_results)

st.dataframe(region_df)


# LLM EXPLANATION
prompt = f"""
User scenario:
Region: {region}
Product: {product}
Quantity: {quantity}
Price: {price}
Discount: {discount}

Best strategy:
{best.to_dict()}

Explain clearly.
"""

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}]
)

st.subheader("🤖 AI Insight")
st.write(response.choices[0].message.content)


# DATABASE
engine = create_engine("sqlite:///results.db")

pd.DataFrame([best]).to_sql("results", engine, if_exists="append", index=False)