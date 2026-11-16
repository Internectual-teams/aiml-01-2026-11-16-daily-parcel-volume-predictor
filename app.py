import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from datetime import date, timedelta
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Bharat Logistics | Parcel Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

DATA_FILE = Path("processed_data.csv")

HUBS = {
    "HUB001": "Chennai Central",
    "HUB002": "Bengaluru South",
    "HUB003": "Hyderabad West"
}

NAVY = "#101C3A"
BLUE = "#2563EB"
PURPLE = "#7C3AED"
LIGHT_BG = "#F4F6FB"

# =========================================================
# PROFESSIONAL ENTERPRISE CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background-color: #F4F6FB;
}

header[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stSidebar"] {
    background: #101C3A;
    border-right: 1px solid #263657;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] .stRadio label {
    background: transparent;
    border-radius: 9px;
    padding: 9px 10px;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: #263657;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #DCE5F5 !important;
}

.main-title {
    font-size: 30px;
    font-weight: 800;
    color: #101C3A;
    margin-bottom: 3px;
}

.subtitle {
    color: #64748B;
    font-size: 14px;
    margin-bottom: 25px;
}

.hero {
    background: linear-gradient(120deg, #101C3A, #263F73);
    padding: 30px;
    border-radius: 18px;
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    color: white !important;
    font-size: 29px;
    font-weight: 800;
}

.hero p {
    color: #DCE7FF !important;
    font-size: 14px;
}

.kpi-card {
    background: #FFFFFF;
    border: 1px solid #E5EAF3;
    border-radius: 15px;
    padding: 20px;
    min-height: 125px;
    box-shadow: 0 3px 12px rgba(16, 28, 58, 0.04);
}

.kpi-label {
    color: #64748B;
    font-size: 13px;
    font-weight: 600;
}

.kpi-value {
    color: #101C3A;
    font-size: 27px;
    font-weight: 800;
    margin-top: 10px;
}

.kpi-caption {
    color: #64748B;
    font-size: 12px;
    margin-top: 5px;
}

.section-title {
    color: #101C3A;
    font-size: 19px;
    font-weight: 800;
    margin-top: 25px;
    margin-bottom: 12px;
}

.info-card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #E5EAF3;
    margin-bottom: 15px;
}

.insight-card {
    background: #F0ECFF;
    border-left: 5px solid #7C3AED;
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 12px;
    color: #33216B;
}

.status-good {
    background: #DCFCE7;
    color: #166534;
    padding: 7px 13px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
}

.status-high {
    background: #FEE2E2;
    color: #991B1B;
    padding: 7px 13px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 700;
}

.stButton button {
    background: #2563EB;
    color: white !important;
    border: none;
    border-radius: 9px;
    font-weight: 700;
    padding: 10px 20px;
}

.stButton button:hover {
    background: #1D4ED8;
    color: white !important;
}

.stDownloadButton button {
    background: #101C3A;
    color: white !important;
    border-radius: 9px;
}

div[data-baseweb="select"] {
    background-color: white;
}

[data-testid="stDataFrame"] {
    background: white;
    border-radius: 12px;
}

hr {
    border-color: #E5EAF3;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# DATA GENERATION
# =========================================================

def create_demo_data():

    np.random.seed(42)

    records = []

    today = date.today()

    hub_base = {
        "HUB001": 850,
        "HUB002": 680,
        "HUB003": 520
    }

    for days_ago in range(180, -1, -1):

        current_date = today - timedelta(days=days_ago)

        for hub, base in hub_base.items():

            weekend_effect = -100 if current_date.weekday() >= 5 else 0

            seasonal_effect = (
                80 * np.sin(current_date.timetuple().tm_yday / 30)
            )

            random_effect = np.random.randint(-100, 120)

            volume = (
                base
                + weekend_effect
                + seasonal_effect
                + random_effect
            )

            volume = max(100, int(volume))

            records.append({
                "delivery_date": current_date,
                "hub_id": hub,
                "parcel_volume": volume
            })

    return pd.DataFrame(records)


# =========================================================
# LOAD AND VALIDATE DATA
# =========================================================

def load_data():

    if DATA_FILE.exists():

        try:
            df = pd.read_csv(DATA_FILE)

            required_columns = [
                "delivery_date",
                "hub_id",
                "parcel_volume"
            ]

            if not all(col in df.columns for col in required_columns):
                st.error("Dataset columns are incorrect.")
                st.stop()

            df["delivery_date"] = pd.to_datetime(
                df["delivery_date"], errors="coerce"
            )

            df["parcel_volume"] = pd.to_numeric(
                df["parcel_volume"], errors="coerce"
            )

            df = df.dropna(
                subset=["delivery_date", "hub_id", "parcel_volume"]
            )

            df["hub_id"] = df["hub_id"].astype(str)

            df = df[df["parcel_volume"] >= 0]

            if df.empty:
                st.error("No valid records found in the dataset.")
                st.stop()

            return df

        except Exception as e:
            st.error(f"Unable to read dataset: {e}")
            st.stop()

    df = create_demo_data()

    df.to_csv(DATA_FILE, index=False)

    return df


df = load_data()


# =========================================================
# MODEL FEATURES
# =========================================================

def prepare_features(data):

    data = data.copy()

    data["delivery_date"] = pd.to_datetime(data["delivery_date"])

    data["day"] = data["delivery_date"].dt.day
    data["month"] = data["delivery_date"].dt.month
    data["weekday"] = data["delivery_date"].dt.weekday
    data["dayofyear"] = data["delivery_date"].dt.dayofyear
    data["year"] = data["delivery_date"].dt.year

    data["days_from_start"] = (
        data["delivery_date"] -
        data["delivery_date"].min()
    ).dt.days

    data["hub_code"] = (
        data["hub_id"]
        .map({hub: i for i, hub in enumerate(HUBS.keys())})
        .fillna(3)
    )

    return data


FEATURES = [
    "day",
    "month",
    "weekday",
    "dayofyear",
    "year",
    "days_from_start",
    "hub_code"
]


# =========================================================
# TRAIN AI MODEL
# =========================================================

def train_model(data):

    processed = prepare_features(data)

    X = processed[FEATURES]
    y = processed["parcel_volume"]

    model = RandomForestRegressor(
        n_estimators=150,
        max_depth=12,
        min_samples_split=3,
        random_state=42
    )

    model.fit(X, y)

    return model


model = train_model(df)


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_volume(selected_hub, selected_date):

    latest_date = pd.to_datetime(df["delivery_date"]).max()

    prediction_date = pd.Timestamp(selected_date)

    input_data = pd.DataFrame([{
        "delivery_date": prediction_date,
        "hub_id": selected_hub
    }])

    input_data["day"] = prediction_date.day
    input_data["month"] = prediction_date.month
    input_data["weekday"] = prediction_date.weekday()
    input_data["dayofyear"] = prediction_date.dayofyear
    input_data["year"] = prediction_date.year

    input_data["days_from_start"] = (
        prediction_date -
        pd.to_datetime(df["delivery_date"]).min()
    ).days

    input_data["hub_code"] = (
        list(HUBS.keys()).index(selected_hub)
        if selected_hub in HUBS else 3
    )

    prediction = model.predict(
        input_data[FEATURES]
    )[0]

    return max(0, int(round(prediction)))


# =========================================================
# COMMON KPI COMPONENT
# =========================================================

def kpi_card(label, value, caption):

    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-caption">{caption}</div>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <div style="padding:15px 5px 25px 5px;">
        <h2 style="color:white;font-size:23px;">
            📦 BHARAT LOGISTICS
        </h2>
        <p style="color:#AFC2E8;font-size:12px;">
            PARCEL INTELLIGENCE PLATFORM
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### WORKSPACE")

    page = st.radio(
        "Navigation",
        [
            "Executive Overview",
            "AI Forecasting",
            "Hub Management",
            "Demand Analytics",
            "Smart Insights",
            "Reports",
            "Model Performance",
            "Settings"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("""
    <div style="background:#1C2C50;padding:15px;border-radius:12px;">
        <p style="font-size:12px;color:#AFC2E8;">
            SYSTEM STATUS
        </p>
        <h4 style="color:#4ADE80;">
            ● Operational
        </h4>
        <p style="font-size:11px;color:#AFC2E8;">
            Parcel Intelligence Platform
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.caption("Bharat Logistics © 2026")


# =========================================================
# EXECUTIVE OVERVIEW
# =========================================================

if page == "Executive Overview":

    total_parcels = int(df["parcel_volume"].sum())

    avg_volume = int(df["parcel_volume"].mean())

    peak_volume = int(df["parcel_volume"].max())

    active_hubs = df["hub_id"].nunique()

    latest_date = df["delivery_date"].max()

    today_data = df[
        df["delivery_date"] == latest_date
    ]

    today_volume = int(today_data["parcel_volume"].sum())

    st.markdown("""
    <div class="hero">
        <h1>Welcome to Parcel Intelligence</h1>
        <p>
            Enterprise logistics analytics and AI-powered
            parcel demand forecasting.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="main-title">Executive Overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Monitor your logistics operations and business performance.</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card(
            "TOTAL PARCELS",
            f"{total_parcels:,}",
            "Across all available records"
        )

    with c2:
        kpi_card(
            "LATEST RECORDED VOLUME",
            f"{today_volume:,}",
            latest_date.strftime("%d %b %Y")
        )

    with c3:
        kpi_card(
            "AVERAGE DAILY VOLUME",
            f"{avg_volume:,}",
            "Average per hub per day"
        )

    with c4:
        kpi_card(
            "ACTIVE HUBS",
            active_hubs,
            "Connected delivery locations"
        )

    c5, c6 = st.columns(2)

    with c5:
        kpi_card(
            "PEAK PARCEL VOLUME",
            f"{peak_volume:,}",
            "Highest single-hub daily volume"
        )

    with c6:
        kpi_card(
            "DATA RECORDS",
            f"{len(df):,}",
            "Available operational records"
        )

    st.markdown(
        '<div class="section-title">Parcel Volume Performance</div>',
        unsafe_allow_html=True
    )

    daily = (
        df.groupby("delivery_date")["parcel_volume"]
        .sum()
        .reset_index()
    )

    fig = px.area(
        daily,
        x="delivery_date",
        y="parcel_volume",
        title="Daily Parcel Volume Trend"
    )

    fig.update_traces(
        line_color=BLUE,
        fillcolor="rgba(37,99,235,0.15)"
    )

    fig.update_layout(
        template="plotly_white",
        height=350,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis_title="Date",
        yaxis_title="Parcel Volume"
    )

    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)

    with left:

        hub_data = (
            df.groupby("hub_id")["parcel_volume"]
            .sum()
            .reset_index()
        )

        hub_data["Hub"] = hub_data["hub_id"].map(HUBS).fillna(
            hub_data["hub_id"]
        )

        fig = px.bar(
            hub_data,
            x="Hub",
            y="parcel_volume",
            title="Hub-wise Parcel Distribution",
            color="Hub",
            color_discrete_sequence=[
                "#2563EB",
                "#7C3AED",
                "#0EA5E9"
            ]
        )

        fig.update_layout(
            template="plotly_white",
            showlegend=False,
            height=330
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:

        fig = px.pie(
            hub_data,
            names="Hub",
            values="parcel_volume",
            hole=0.55,
            title="Parcel Contribution by Hub",
            color_discrete_sequence=[
                "#2563EB",
                "#7C3AED",
                "#0EA5E9"
            ]
        )

        fig.update_layout(
            template="plotly_white",
            height=330
        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="section-title">Recent Operational Records</div>',
        unsafe_allow_html=True
    )

    recent = df.sort_values(
        "delivery_date", ascending=False
    ).head(10).copy()

    recent["Hub Name"] = recent["hub_id"].map(HUBS).fillna(
        recent["hub_id"]
    )

    recent = recent.rename(columns={
        "delivery_date": "Delivery Date",
        "parcel_volume": "Parcel Volume"
    })

    st.dataframe(
        recent[["Delivery Date", "Hub Name", "Parcel Volume"]],
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# AI FORECASTING
# =========================================================

elif page == "AI Forecasting":

    st.markdown(
        '<div class="main-title">AI Forecasting Centre</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Predict future parcel demand using machine learning.</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <h3 style="color:#7C3AED;">✦ Intelligent Parcel Forecasting</h3>
        <p style="color:#64748B;">
            Generate parcel volume predictions for individual
            delivery hubs and future operational dates.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:

        selected_hub = st.selectbox(
            "Select Delivery Hub",
            list(HUBS.keys()),
            format_func=lambda x: HUBS[x]
        )

    with col2:

        selected_date = st.date_input(
            "Select Forecast Date",
            value=date.today() + timedelta(days=1),
            min_value=date.today(),
            max_value=date.today() + timedelta(days=365)
        )

    if st.button("Generate AI Forecast", use_container_width=True):

        prediction = predict_volume(
            selected_hub,
            selected_date
        )

        historical = df[
            df["hub_id"] == selected_hub
        ]

        average = int(historical["parcel_volume"].mean())

        if prediction > average * 1.25:
            demand = "HIGH"
            color = "#DC2626"
            recommendation = (
                "Consider increasing delivery staff and "
                "transport capacity."
            )

        elif prediction < average * 0.75:
            demand = "LOW"
            color = "#2563EB"
            recommendation = (
                "Current operational resources may be sufficient."
            )

        else:
            demand = "NORMAL"
            color = "#16A34A"
            recommendation = (
                "Maintain the current delivery resource allocation."
            )

        st.session_state["last_prediction"] = {
            "hub": selected_hub,
            "date": selected_date,
            "prediction": prediction,
            "demand": demand
        }

        st.markdown("---")

        st.markdown(
            '<div class="section-title">Forecast Results</div>',
            unsafe_allow_html=True
        )

        a, b, c = st.columns(3)

        with a:
            kpi_card(
                "PREDICTED PARCELS",
                f"{prediction:,}",
                "AI forecast result"
            )

        with b:
            kpi_card(
                "HISTORICAL AVERAGE",
                f"{average:,}",
                "Average daily hub volume"
            )

        with c:
            kpi_card(
                "DEMAND LEVEL",
                demand,
                "Forecast classification"
            )

        st.markdown(
            f"""
            <div class="info-card">
                <h3 style="color:{color};">
                    Demand Status: {demand}
                </h3>
                <p style="color:#334155;">
                    <b>Hub:</b> {HUBS[selected_hub]}<br>
                    <b>Forecast Date:</b> {selected_date.strftime("%d %B %Y")}<br>
                    <b>Predicted Volume:</b> {prediction:,} parcels
                </p>
                <hr>
                <p style="color:#334155;">
                    <b>Business Recommendation:</b><br>
                    {recommendation}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction,
            title={"text": "Predicted Parcel Volume"},
            gauge={
                "axis": {"range": [0, max(prediction * 1.5, 100)]},
                "bar": {"color": PURPLE},
                "bgcolor": "#EDE9FE",
                "steps": [
                    {"range": [0, max(prediction * 0.5, 50)],
                     "color": "#E0E7FF"},
                    {"range": [max(prediction * 0.5, 50),
                               max(prediction, 100)],
                     "color": "#DDD6FE"}
                ]
            }
        ))

        gauge.update_layout(
            template="plotly_white",
            height=300
        )

        st.plotly_chart(gauge, use_container_width=True)

        result_df = pd.DataFrame([{
            "Hub ID": selected_hub,
            "Hub Name": HUBS[selected_hub],
            "Forecast Date": selected_date,
            "Predicted Parcel Volume": prediction,
            "Demand Level": demand
        }])

        st.download_button(
            "Download Forecast Report",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="parcel_forecast_report.csv",
            mime="text/csv"
        )

    else:
        st.info(
            "Select a delivery hub and forecast date, "
            "then generate your AI prediction."
        )


# =========================================================
# HUB MANAGEMENT
# =========================================================

elif page == "Hub Management":

    st.markdown(
        '<div class="main-title">Hub Operations Management</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Monitor individual delivery hub performance.</div>',
        unsafe_allow_html=True
    )

    selected_hub = st.selectbox(
        "Choose Hub",
        list(HUBS.keys()),
        format_func=lambda x: HUBS[x]
    )

    hub_df = df[df["hub_id"] == selected_hub].copy()

    hub_total = int(hub_df["parcel_volume"].sum())

    hub_average = int(hub_df["parcel_volume"].mean())

    hub_peak = int(hub_df["parcel_volume"].max())

    a, b, c = st.columns(3)

    with a:
        kpi_card("TOTAL PARCELS", f"{hub_total:,}", "Selected hub")

    with b:
        kpi_card("DAILY AVERAGE", f"{hub_average:,}", "Average volume")

    with c:
        kpi_card("PEAK VOLUME", f"{hub_peak:,}", "Maximum recorded")

    hub_daily = (
        hub_df.groupby("delivery_date")["parcel_volume"]
        .sum()
        .reset_index()
    )

    fig = px.line(
        hub_daily,
        x="delivery_date",
        y="parcel_volume",
        title=f"{HUBS[selected_hub]} – Parcel Volume Trend",
        markers=True
    )

    fig.update_traces(
        line_color=PURPLE,
        line_width=3
    )

    fig.update_layout(
        template="plotly_white",
        height=380
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        '<div class="section-title">Hub Performance Comparison</div>',
        unsafe_allow_html=True
    )

    comparison = (
        df.groupby("hub_id")["parcel_volume"]
        .mean()
        .reset_index()
    )

    comparison["Hub Name"] = comparison["hub_id"].map(HUBS).fillna(
        comparison["hub_id"]
    )

    comparison = comparison.rename(
        columns={"parcel_volume": "Average Parcel Volume"}
    )

    st.dataframe(
        comparison[["Hub Name", "Average Parcel Volume"]],
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# DEMAND ANALYTICS
# =========================================================

elif page == "Demand Analytics":

    st.markdown(
        '<div class="main-title">Demand Analytics</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Explore historical parcel demand and volume patterns.</div>',
        unsafe_allow_html=True
    )

    period = st.selectbox(
        "Analysis Period",
        ["Daily", "Weekly", "Monthly"]
    )

    analytics_df = df.copy()

    analytics_df["delivery_date"] = pd.to_datetime(
        analytics_df["delivery_date"]
    )

    if period == "Daily":
        analytics_df["Period"] = (
            analytics_df["delivery_date"].dt.strftime("%d %b %Y")
        )

    elif period == "Weekly":
        analytics_df["Period"] = (
            analytics_df["delivery_date"].dt.to_period("W").astype(str)
        )

    else:
        analytics_df["Period"] = (
            analytics_df["delivery_date"].dt.to_period("M").astype(str)
        )

    trend = (
        analytics_df.groupby("Period")["parcel_volume"]
        .sum()
        .reset_index()
    )

    fig = px.bar(
        trend,
        x="Period",
        y="parcel_volume",
        title=f"{period} Parcel Demand Analysis",
        color="parcel_volume",
        color_continuous_scale="Blues"
    )

    fig.update_layout(
        template="plotly_white",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    hub_trend = (
        analytics_df.groupby(["Period", "hub_id"])["parcel_volume"]
        .sum()
        .reset_index()
    )

    hub_trend["Hub Name"] = hub_trend["hub_id"].map(HUBS).fillna(
        hub_trend["hub_id"]
    )

    fig = px.line(
        hub_trend,
        x="Period",
        y="parcel_volume",
        color="Hub Name",
        title="Hub-wise Demand Comparison",
        markers=True
    )

    fig.update_layout(
        template="plotly_white",
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    weekday_data = df.copy()

    weekday_data["Weekday"] = (
        pd.to_datetime(weekday_data["delivery_date"])
        .dt.day_name()
    )

    weekday_order = [
        "Monday", "Tuesday", "Wednesday",
        "Thursday", "Friday", "Saturday", "Sunday"
    ]

    weekday_data = (
        weekday_data.groupby("Weekday")["parcel_volume"]
        .mean()
        .reindex(weekday_order)
        .reset_index()
    )

    fig = px.bar(
        weekday_data,
        x="Weekday",
        y="parcel_volume",
        title="Average Parcel Volume by Day of Week",
        color="parcel_volume",
        color_continuous_scale="Purples"
    )

    fig.update_layout(
        template="plotly_white",
        height=350
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# SMART INSIGHTS
# =========================================================

elif page == "Smart Insights":

    st.markdown(
        '<div class="main-title">Smart Business Insights</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Discover useful operational patterns from parcel data.</div>',
        unsafe_allow_html=True
    )

    hub_avg = (
        df.groupby("hub_id")["parcel_volume"]
        .mean()
        .sort_values(ascending=False)
    )

    highest_hub = hub_avg.index[0]

    lowest_hub = hub_avg.index[-1]

    daily_total = (
        df.groupby("delivery_date")["parcel_volume"]
        .sum()
    )

    busiest_day = daily_total.idxmax()

    busiest_volume = int(daily_total.max())

    quiet_day = daily_total.idxmin()

    quiet_volume = int(daily_total.min())

    insights = [
        (
            "🏆 Highest Average Volume",
            f"{HUBS.get(highest_hub, highest_hub)} "
            f"has the highest average parcel volume "
            f"at {int(hub_avg.iloc[0]):,} parcels per day."
        ),
        (
            "📉 Lowest Average Volume",
            f"{HUBS.get(lowest_hub, lowest_hub)} "
            f"has the lowest average parcel volume "
            f"at {int(hub_avg.iloc[-1]):,} parcels per day."
        ),
        (
            "📈 Busiest Recorded Day",
            f"{pd.to_datetime(busiest_day).strftime('%d %B %Y')} "
            f"recorded {busiest_volume:,} parcels across all hubs."
        ),
        (
            "📊 Lowest Recorded Day",
            f"{pd.to_datetime(quiet_day).strftime('%d %B %Y')} "
            f"recorded {quiet_volume:,} parcels across all hubs."
        )
    ]

    for title, description in insights:

        st.markdown(
            f"""
            <div class="insight-card">
                <h4>{title}</h4>
                <p>{description}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown(
        '<div class="section-title">Operational Recommendations</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <h4 style="color:#101C3A;">Resource Planning</h4>
        <p style="color:#64748B;">
            Review high-volume hub performance when allocating
            delivery staff and transportation resources.
        </p>
        <hr>
        <h4 style="color:#101C3A;">Demand Monitoring</h4>
        <p style="color:#64748B;">
            Monitor daily changes in parcel demand to identify
            unusual operational patterns.
        </p>
        <hr>
        <h4 style="color:#101C3A;">Hub Optimization</h4>
        <p style="color:#64748B;">
            Compare hub performance regularly to support
            better operational decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# REPORTS
# =========================================================

elif page == "Reports":

    st.markdown(
        '<div class="main-title">Reports & Downloads</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Access and export business reports.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Available Reports</div>',
        unsafe_allow_html=True
    )

    report_type = st.selectbox(
        "Select Report",
        [
            "Complete Parcel Dataset",
            "Hub-wise Performance",
            "Daily Parcel Summary"
        ]
    )

    if report_type == "Complete Parcel Dataset":

        report = df.copy()

    elif report_type == "Hub-wise Performance":

        report = (
            df.groupby("hub_id")
            .agg(
                Total_Parcels=("parcel_volume", "sum"),
                Average_Parcels=("parcel_volume", "mean"),
                Peak_Parcels=("parcel_volume", "max"),
                Total_Records=("parcel_volume", "count")
            )
            .reset_index()
        )

        report["Hub_Name"] = report["hub_id"].map(HUBS).fillna(
            report["hub_id"]
        )

    else:

        report = (
            df.groupby("delivery_date")
            .agg(
                Total_Parcels=("parcel_volume", "sum"),
                Average_Parcels=("parcel_volume", "mean")
            )
            .reset_index()
        )

    st.dataframe(
        report,
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "⬇ Download Selected Report",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="bharat_logistics_report.csv",
        mime="text/csv",
        use_container_width=True
    )

    if "last_prediction" in st.session_state:

        st.markdown(
            '<div class="section-title">Latest Forecast Report</div>',
            unsafe_allow_html=True
        )

        result = st.session_state["last_prediction"]

        forecast_report = pd.DataFrame([{
            "Hub": HUBS[result["hub"]],
            "Forecast Date": result["date"],
            "Predicted Parcels": result["prediction"],
            "Demand Level": result["demand"]
        }])

        st.dataframe(
            forecast_report,
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download Forecast History",
            data=forecast_report.to_csv(index=False).encode("utf-8"),
            file_name="forecast_history.csv",
            mime="text/csv"
        )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="main-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Evaluate the forecasting model using historical data.</div>',
        unsafe_allow_html=True
    )

    processed = prepare_features(df)

    X = processed[FEATURES]
    y = processed["parcel_volume"]

    if len(processed) >= 10:

        split_index = int(len(processed) * 0.8)

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]

        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        evaluation_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            random_state=42
        )

        evaluation_model.fit(X_train, y_train)

        predictions = evaluation_model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)

        r2 = r2_score(y_test, predictions)

        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))

        a, b, c = st.columns(3)

        with a:
            kpi_card(
                "MEAN ABSOLUTE ERROR",
                f"{mae:.2f}",
                "Average absolute prediction error"
            )

        with b:
            kpi_card(
                "R² SCORE",
                f"{r2:.3f}",
                "Model evaluation score"
            )

        with c:
            kpi_card(
                "ROOT MEAN SQUARED ERROR",
                f"{rmse:.2f}",
                "Prediction error measure"
            )

        evaluation = pd.DataFrame({
            "Actual Volume": y_test.values,
            "Predicted Volume": predictions
        })

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=evaluation["Actual Volume"],
            mode="lines+markers",
            name="Actual Volume",
            line=dict(color=BLUE)
        ))

        fig.add_trace(go.Scatter(
            y=evaluation["Predicted Volume"],
            mode="lines+markers",
            name="Predicted Volume",
            line=dict(color=PURPLE)
        ))

        fig.update_layout(
            title="Actual vs Predicted Parcel Volume",
            template="plotly_white",
            height=400,
            xaxis_title="Test Records",
            yaxis_title="Parcel Volume"
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="section-title">Evaluation Records</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            evaluation.head(50),
            use_container_width=True,
            hide_index=True
        )

    else:

        st.warning(
            "At least 10 valid records are required "
            "to evaluate model performance."
        )


# =========================================================
# SETTINGS
# =========================================================

elif page == "Settings":

    st.markdown(
        '<div class="main-title">Platform Settings</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Manage data and platform information.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Dataset Management</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Upload your operational parcel dataset to update "
        "the analytics and forecasting data."
    )

    st.info(
        "Required CSV columns: delivery_date, hub_id, parcel_volume"
    )

    uploaded_file = st.file_uploader(
        "Upload Parcel Dataset",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            new_data = pd.read_csv(uploaded_file)

            required = [
                "delivery_date",
                "hub_id",
                "parcel_volume"
            ]

            if not all(col in new_data.columns for col in required):

                st.error("Uploaded file does not contain all required columns.")

            else:

                new_data["delivery_date"] = pd.to_datetime(
                    new_data["delivery_date"], errors="coerce"
                )

                new_data["parcel_volume"] = pd.to_numeric(
                    new_data["parcel_volume"], errors="coerce"
                )

                new_data = new_data.dropna(
                    subset=required
                )

                new_data["hub_id"] = new_data["hub_id"].astype(str)

                new_data = new_data[
                    new_data["parcel_volume"] >= 0
                ]

                if new_data.empty:
                    st.error("The uploaded dataset has no valid records.")

                else:

                    st.success(
                        f"{len(new_data)} valid records are ready to import."
                    )

                    st.dataframe(
                        new_data.head(10),
                        use_container_width=True
                    )

                    if st.button("Import Dataset"):

                        new_data.to_csv(
                            DATA_FILE,
                            index=False
                        )

                        st.cache_data.clear()
                        st.cache_resource.clear()

                        st.success("Dataset imported successfully.")

                        st.rerun()

        except Exception as e:

            st.error(f"Upload failed: {e}")

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Platform Information</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="info-card">
        <h3 style="color:#101C3A;">
            Bharat Logistics – Parcel Intelligence
        </h3>
        <p style="color:#64748B;">
            AI-powered parcel volume forecasting and logistics
            analytics platform.
        </p>
        <hr>
        <b>Platform:</b> Parcel Intelligence<br><br>
        <b>Technology:</b> Machine Learning and Data Analytics<br><br>
        <b>Forecasting:</b> Random Forest Regression<br><br>
        <b>Data Format:</b> CSV
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;padding:12px;color:#94A3B8;font-size:12px;">
        BHARAT LOGISTICS | PARCEL INTELLIGENCE PLATFORM<br>
        AI-Powered Logistics Analytics
    </div>
    """,
    unsafe_allow_html=True
)