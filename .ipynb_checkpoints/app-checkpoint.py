import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================
# PAGE CONFIG + PREMIUM UI
# =====================================================
st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛒",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    h1, h2, h3 { color: #F5F5F5; }
    .card {
        background-color: #1E222A;
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# =====================================================
# HEADER
# =====================================================
st.markdown(
    "<h1 style='text-align:center;'>🛒 Shopper Spectrum</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;color:gray;'>Customer Segmentation & Intelligent Product Recommendation System</p>",
    unsafe_allow_html=True
)
st.divider()

# =====================================================
# LOAD & CLEAN DATA (CACHED)
# =====================================================
@st.cache_data
def load_data():
    df = pd.read_csv("online_retail.csv")
    df = df.dropna(subset=["CustomerID"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    return df

df = load_data()

# =====================================================
# RFM + CLUSTERING MODEL (CACHED)
# =====================================================
@st.cache_resource
def build_rfm_model(df):
    reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum")
    ).reset_index()

    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[["Recency", "Frequency", "Monetary"]])

    kmeans = KMeans(n_clusters=4, random_state=42)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

    labels = {
        0: "High-Value",
        1: "Regular",
        2: "Occasional",
        3: "At-Risk"
    }
    rfm["Segment"] = rfm["Cluster"].map(labels)

    return rfm, scaler, kmeans, labels

rfm, scaler, kmeans, cluster_labels = build_rfm_model(df)

# =====================================================
# RECOMMENDATION SYSTEM (CACHED)
# =====================================================
@st.cache_resource
def build_recommendation(df):
    matrix = df.pivot_table(
        index="CustomerID",
        columns="Description",
        values="Quantity",
        aggfunc="sum",
        fill_value=0
    )
    product_matrix = matrix.T
    similarity = cosine_similarity(product_matrix)

    similarity_df = pd.DataFrame(
        similarity,
        index=product_matrix.index,
        columns=product_matrix.index
    )
    return similarity_df

product_similarity_df = build_recommendation(df)

def recommend_products(product_name, top_n=5):
    product_name = product_name.upper()
    matches = [p for p in product_similarity_df.index if product_name in p]
    if not matches:
        return None
    product = matches[0]
    return (
        product_similarity_df[product]
        .sort_values(ascending=False)
        .iloc[1:top_n+1]
    )

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.markdown("## 📊 Navigation")
menu = st.sidebar.radio(
    "",
    ["🎯 Product Recommendation", "👤 Customer Segmentation"]
)
st.sidebar.markdown("---")
st.sidebar.info("Built using RFM Analysis & Collaborative Filtering")

# =====================================================
# MODULE 1: PRODUCT RECOMMENDATION
# =====================================================
if menu == "🎯 Product Recommendation":
    st.subheader("🎯 Smart Product Recommendation")

    product_name = st.text_input(
        "🔍 Enter Product Name",
        placeholder="e.g. WHITE HANGING HEART"
    )

    if st.button("✨ Get Recommendations"):
        results = recommend_products(product_name)

        if results is None:
            st.error("❌ Product not found. Try another keyword.")
        else:
            st.success("✅ Top Recommended Products")
            for prod, score in results.items():
                st.markdown(
                    f"""
                    <div class="card">
                        <b>{prod}</b><br>
                        Similarity Score:
                        <span style="color:#00E676;">{score:.2f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# =====================================================
# MODULE 2: CUSTOMER SEGMENTATION
# =====================================================
else:
    st.subheader("👤 Customer Segmentation Predictor")

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input("📅 Recency (days)", min_value=0)
    with col2:
        frequency = st.number_input("🔁 Frequency", min_value=0)
    with col3:
        monetary = st.number_input("💰 Monetary Value", min_value=0.0)

    if st.button("🚀 Predict Segment"):
        input_data = scaler.transform([[recency, frequency, monetary]])
        cluster = kmeans.predict(input_data)[0]
        segment = cluster_labels[cluster]

        color_map = {
            "High-Value": "#00E676",
            "Regular": "#448AFF",
            "Occasional": "#FFD54F",
            "At-Risk": "#FF5252"
        }

        st.markdown(
            f"""
            <div class="card" style="text-align:center;">
                <h3 style="color:{color_map[segment]};">
                    {segment} Customer
                </h3>
            </div>
            """,
            unsafe_allow_html=True
        )
