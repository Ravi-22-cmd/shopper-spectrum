import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import time

# =====================================================
# CORE NEXUS CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="NEXUS OS | DIGITAL RAIN",
    page_icon="🌌",
    layout="wide",
)

# =====================================================
# MATRIX RAIN & CYBERPUNK CSS
# =====================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syncopate:wght@400;700&family=Space+Grotesk:wght@300;500;700&display=swap');

    :root {
        --glow-cyan: #00f2ff;
        --glow-magenta: #ff00ff;
        --matrix-green: rgba(0, 255, 70, 0.15);
    }

    /* Matrix Digital Rain Background */
    .stApp {
        background: #020617;
        color: #f8fafc;
        font-family: 'Space Grotesk', sans-serif;
        overflow: hidden;
    }

    .stApp::before {
        content: "10101101010101010101010101011010101010101010101010101";
        position: fixed;
        top: -100%;
        left: 0;
        width: 100%;
        height: 200%;
        color: var(--matrix-green);
        font-family: monospace;
        font-size: 20px;
        line-height: 1;
        white-space: pre-wrap;
        word-break: break-all;
        animation: rain 20s linear infinite;
        z-index: -1;
        opacity: 0.3;
        pointer-events: none;
    }

    @keyframes rain {
        from { transform: translateY(0); }
        to { transform: translateY(50%); }
    }

    /* Glassmorphism Cyber Card */
    .cyber-card {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(0, 242, 255, 0.2);
        border-radius: 15px;
        padding: 25px;
        backdrop-filter: blur(10px);
        transition: 0.4s ease;
        position: relative;
    }

    .cyber-card:hover {
        border-color: var(--glow-cyan);
        box-shadow: 0 0 20px rgba(0, 242, 255, 0.4);
        transform: scale(1.01);
    }

    /* Titles with Glitch Text Shadow */
    h1, h2, h3 {
        font-family: 'Syncopate', sans-serif !important;
        background: linear-gradient(to right, #00f2ff, #ff00ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 0px rgba(255, 0, 255, 0.2);
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #020617; }
    ::-webkit-scrollbar-thumb { background: var(--glow-cyan); border-radius: 10px; }

</style>
""", unsafe_allow_html=True)

# =====================================================
# DATA ENGINE
# =====================================================
@st.cache_data
def load_nexus_data():
    try:
        df = pd.read_csv("online_retail.csv")
        df = df.dropna(subset=["CustomerID"])
        df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
        df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
        return df
    except: return pd.DataFrame()

df = load_nexus_data()

# =====================================================
# COMPONENTS
# =====================================================
def draw_counter(label, value, prefix=""):
    html = f"""
    <div class="cyber-card">
        <div style="text-align:center;">
            <div style="color: #94a3b8; font-size: 0.7rem; letter-spacing: 2px;">{label}</div>
            <div style="font-family: 'Syncopate', sans-serif; font-size: 2rem; color: #00f2ff; text-shadow: 0 0 10px #00f2ff;" id="{label.replace(' ', '')}">0</div>
        </div>
    </div>
    <script>
    (function() {{
        let start = 0; const end = {value}; const duration = 2000;
        const element = document.getElementById('{label.replace(' ', '')}');
        let startTime = null;
        function animate(currentTime) {{
            if (!startTime) startTime = currentTime;
            const progress = Math.min((currentTime - startTime) / duration, 1);
            element.innerText = "{prefix}" + Math.floor(progress * end).toLocaleString();
            if (progress < 1) requestAnimationFrame(animate);
        }}
        requestAnimationFrame(animate);
    }})();
    </script>
    """
    st.components.v1.html(html, height=130)

# =====================================================
# NAVIGATION & SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown("<h1 style='font-size: 1rem;'>NEXUS TERMINAL</h1>", unsafe_allow_html=True)
    mode = st.radio("INTERFACE", ["🛰️ DASHBOARD", "🧠 NEURAL MAP", "⚡ SYNC ENGINE"])
    st.divider()
    region = st.selectbox("REGION", ["GLOBAL"] + sorted(df["Country"].unique().tolist()) if not df.empty else ["GLOBAL"])

df_view = df if region == "GLOBAL" else df[df["Country"] == region]

# =====================================================
# INTERFACE LOGIC
# =====================================================
if mode == "🛰️ DASHBOARD":
    st.markdown("<h2>System Metrics</h2>", unsafe_allow_html=True)
    if not df_view.empty:
        c1, c2, c3 = st.columns(3)
        with c1: draw_counter("TOTAL REVENUE", int(df_view['TotalPrice'].sum()), "$")
        with c2: draw_counter("ENTITY COUNT", int(df_view['CustomerID'].nunique()))
        with c3: draw_counter("COMMAND LOGS", int(df_view['InvoiceNo'].nunique()))

        st.markdown("<br>", unsafe_allow_html=True)
        trend = df_view.set_index('InvoiceDate').resample('M')['TotalPrice'].sum().reset_index()
        fig = px.line(trend, x='InvoiceDate', y='TotalPrice', template="plotly_dark")
        fig.update_traces(line_color='#00f2ff', line_width=4)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                          xaxis=dict(showgrid=False), yaxis=dict(gridcolor='rgba(0,242,255,0.1)'))
        st.plotly_chart(fig, use_container_width=True)

elif mode == "🧠 NEURAL MAP":
    st.markdown("<h2>Cluster Displacement</h2>", unsafe_allow_html=True)
    # 
    ref = df_view["InvoiceDate"].max() + pd.Timedelta(days=1)
    rfm = df_view.groupby("CustomerID").agg({"InvoiceDate": lambda x: (ref - x.max()).days,
                                             "InvoiceNo": "nunique", "TotalPrice": "sum"}).reset_index()
    rfm.columns = ["ID", "R", "F", "M"]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(rfm[["R", "F", "M"]])
    km = KMeans(n_clusters=4, random_state=42).fit(scaled)
    rfm["Segment"] = km.labels_.astype(str)

    fig_3d = px.scatter_3d(rfm, x='R', y='F', z='M', color='Segment', 
                           color_discrete_sequence=px.colors.qualitative.Alphabet)
    fig_3d.update_layout(scene=dict(xaxis_backgroundcolor="#020617", yaxis_backgroundcolor="#020617", zaxis_backgroundcolor="#020617"),
                         paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0,r=0,t=0,b=0))
    st.plotly_chart(fig_3d, use_container_width=True)

elif mode == "⚡ SYNC ENGINE":
    st.markdown("<h2>Neural Link Sync</h2>", unsafe_allow_html=True)
    
    # Sirf un products ko lein jo kam se kam 5 baar bikke hain (Noise hatane ke liye)
    product_counts = df['Description'].value_counts()
    popular_products = product_counts[product_counts > 5].index
    df_reduced = df[df['Description'].isin(popular_products)]

    @st.cache_resource
    def get_reliable_matrix(_data):
        # Pivot table banate waqt CustomerID aur Description ka sahi alignment zaroori hai
        mat = _data.pivot_table(index="CustomerID", columns="Description", values="Quantity", aggfunc="sum").fillna(0)
        # Cosine Similarity calculate karein
        sim = cosine_similarity(mat.T)
        return pd.DataFrame(sim, index=mat.columns, columns=mat.columns)

    with st.spinner("CALIBRATING NEURAL LINKS..."):
        sim_df = get_reliable_matrix(df_reduced.sample(min(30000, len(df_reduced))))
    
    target = st.selectbox("SEARCH NODE (Try 'WHITE HANGING HEART T-LIGHT HOLDER')", [""] + sim_df.index.tolist())
    
    if target:
        # Top 5 matches nikaalein (khud product ko chhod kar)
        recs = sim_df[target].sort_values(ascending=False)[1:6]
        
        if recs.max() == 0:
            st.warning("⚠️ SYNC FAILED: Is product ke liye koi behavioral link nahi mila. Doosra node select karein.")
        else:
            for name, score in recs.items():
                if score > 0: # Sirf positive matches dikhayein
                    st.markdown(f"""
                    <div class="cyber-card" style="margin-bottom:10px; border-left: 5px solid #ff00ff;">
                        <div style="display:flex; justify-content:space-between;">
                            <span style="font-weight:bold;">{name}</span>
                            <span style="color:#00f2ff; font-family:Syncopate;">{(score*100):.1f}%</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

st.markdown("<p style='text-align:center; opacity:0.2; margin-top:50px;'>OS VERSION 10.0 // MATRIX CONNECTED</p>", unsafe_allow_html=True)