import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import numpy as np
from dotenv import load_dotenv
from google import genai

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)
LLM_MODEL = "gemini-3.1-flash-lite-preview"

st.set_page_config(
    page_title="HomeScope AI | King County",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — Modern tech / Duolingo-inspired
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@500;600;700&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { font-family: 'Inter', sans-serif !important; }
h1, h2, h3, .logo-text { font-family: 'Space Grotesk', sans-serif !important; }

.stApp {
    background: #0f172a;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1e293b; }
::-webkit-scrollbar-thumb { background: #6366f1; border-radius: 10px; }

/* ── Header ── */
h1 {
    background: linear-gradient(135deg, #818cf8, #6366f1, #a78bfa) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    text-align: center !important;
    font-size: 2.6rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
    border: none !important;
    padding-bottom: 0 !important;
}

h2 {
    color: #e2e8f0 !important;
    font-size: 1.3rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.3px;
}

h3 {
    color: #cbd5e1 !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    justify-content: center;
    background: #1e293b;
    border-radius: 16px;
    padding: 6px;
    border: 1px solid #334155;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #94a3b8;
    border-radius: 12px;
    padding: 12px 28px;
    font-weight: 600;
    font-size: 0.95rem;
    border: none;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(99, 102, 241, 0.1);
    color: #c7d2fe;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
}
.stTabs [data-baseweb="tab-panel"] {
    padding-top: 20px;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}
.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* ── KPI Cards ── */
[data-testid="stMetric"] {
    background: linear-gradient(145deg, #1e293b, #253348);
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 20px 22px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.15);
    border-color: #6366f1;
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stMetricDelta"] {
    font-weight: 600 !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #1e293b;
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] h2 {
    color: #c7d2fe !important;
    font-size: 1rem !important;
}
[data-testid="stSidebar"] .stCaption {
    color: #64748b !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 14px 28px !important;
    font-size: 1rem !important;
    letter-spacing: 0.3px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.5) !important;
    background: linear-gradient(135deg, #818cf8, #a78bfa) !important;
}
.stButton > button:active {
    transform: translateY(-1px) !important;
}

/* ── Dividers ── */
hr {
    border-color: #1e293b !important;
    margin: 25px 0 !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border: 1px solid #334155;
    border-radius: 12px;
    overflow: hidden;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: #1e293b !important;
    border-radius: 12px !important;
    border: 1px solid #334155 !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
}

/* ── Custom Cards ── */
.glass-card {
    background: rgba(30, 41, 59, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 22px 26px;
    margin: 12px 0;
}

.tip-card {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.08));
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 14px;
    padding: 16px 20px;
    margin: 12px 0 20px 0;
    color: #c7d2fe;
    font-size: 0.9rem;
    line-height: 1.6;
}
.tip-card b { color: #a5b4fc; }

.success-card {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.08), rgba(16, 185, 129, 0.08));
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 14px;
    padding: 16px 20px;
    margin: 12px 0 20px 0;
    color: #86efac;
    font-size: 0.9rem;
}

.warn-card {
    background: linear-gradient(135deg, rgba(251, 191, 36, 0.08), rgba(245, 158, 11, 0.08));
    border: 1px solid rgba(251, 191, 36, 0.3);
    border-radius: 14px;
    padding: 16px 20px;
    margin: 12px 0 20px 0;
    color: #fde68a;
    font-size: 0.9rem;
}

.ai-response {
    background: linear-gradient(145deg, #1e293b, #253348);
    border: 1px solid #6366f1;
    border-radius: 16px;
    padding: 24px 28px;
    margin: 15px 0;
    color: #e2e8f0;
    line-height: 1.7;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.15);
}
.ai-response p { color: #cbd5e1; }

.step-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-weight: 800;
    font-size: 0.75rem;
    padding: 4px 12px;
    border-radius: 20px;
    margin-right: 8px;
    letter-spacing: 0.5px;
}

.section-label {
    color: #94a3b8;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
    margin-bottom: 10px;
}

/* ── Welcome Modal ── */
.welcome-overlay {
    position: fixed;
    top: 0; left: 0;
    width: 100vw; height: 100vh;
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(8px);
    z-index: 99999;
    display: flex;
    align-items: center;
    justify-content: center;
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}
.welcome-modal {
    background: linear-gradient(160deg, #1e293b, #0f172a);
    border: 1px solid #6366f1;
    border-radius: 24px;
    padding: 40px 44px;
    max-width: 640px;
    width: 90%;
    box-shadow: 0 25px 60px rgba(99, 102, 241, 0.25);
    animation: slideUp 0.4s ease 0.1s both;
    text-align: center;
}
.welcome-modal h2 {
    font-family: 'Space Grotesk', sans-serif !important;
    background: linear-gradient(135deg, #818cf8, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    margin-bottom: 8px;
}
.welcome-modal .subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-bottom: 28px;
}
.welcome-modal .feature-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin: 24px 0;
    text-align: left;
}
.welcome-modal .feature {
    background: rgba(99, 102, 241, 0.06);
    border: 1px solid rgba(99, 102, 241, 0.15);
    border-radius: 12px;
    padding: 14px 16px;
}
.welcome-modal .feature .icon {
    font-size: 1.4rem;
    margin-bottom: 6px;
}
.welcome-modal .feature .title {
    color: #e2e8f0;
    font-weight: 700;
    font-size: 0.88rem;
}
.welcome-modal .feature .desc {
    color: #94a3b8;
    font-size: 0.78rem;
    margin-top: 2px;
}
.welcome-modal .cta-btn {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-weight: 700;
    font-size: 1rem;
    padding: 14px 40px;
    border-radius: 14px;
    border: none;
    cursor: pointer;
    margin-top: 24px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    text-decoration: none;
}
.welcome-modal .cta-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.6);
}
.welcome-modal .tech-stack {
    color: #64748b;
    font-size: 0.72rem;
    margin-top: 20px;
    letter-spacing: 0.5px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Welcome Popup (one-time)
# ---------------------------------------------------------------------------
if "welcomed" not in st.session_state:
    st.session_state.welcomed = False

if not st.session_state.welcomed:
    st.markdown("""
    <div class="welcome-overlay" id="welcomeOverlay">
        <div class="welcome-modal">
            <div style="font-size:3rem; margin-bottom:10px;">🏡</div>
            <h2>Bienvenue sur HomeScope AI</h2>
            <p class="subtitle">Votre analyseur immobilier intelligent pour le comté de King (Seattle)</p>

            <div class="feature-grid">
                <div class="feature">
                    <div class="icon">📊</div>
                    <div class="title">Exploration du marché</div>
                    <div class="desc">Filtrez et visualisez 21 613 propriétés avec des graphiques interactifs</div>
                </div>
                <div class="feature">
                    <div class="icon">🔍</div>
                    <div class="title">Analyse détaillée</div>
                    <div class="desc">Étudiez une propriété et comparez-la à des biens similaires</div>
                </div>
                <div class="feature">
                    <div class="icon">🤖</div>
                    <div class="title">IA intégrée</div>
                    <div class="desc">Résumés de marché et recommandations générés par Google Gemini</div>
                </div>
                <div class="feature">
                    <div class="icon">🗺️</div>
                    <div class="title">Cartes interactives</div>
                    <div class="desc">Localisez les propriétés sur la carte de Seattle</div>
                </div>
            </div>

            <button class="cta-btn" onclick="
                document.getElementById('welcomeOverlay').style.opacity='0';
                setTimeout(function(){document.getElementById('welcomeOverlay').style.display='none';},300);
            ">Commencer l'exploration</button>

            <div class="tech-stack">STREAMLIT  •  PANDAS  •  MATPLOTLIB  •  GOOGLE GEMINI  •  PYTHON</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.session_state.welcomed = True

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("HomeScope AI")
st.markdown(
    "<p style='text-align:center; color:#94a3b8; margin-top:-8px; font-size:1rem; letter-spacing:0.5px;'>"
    "Analyseur immobilier intelligent &mdash; King County, Seattle"
    "</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("kc_house_data.csv")
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%S")
    df["price_per_sqft"] = df["price"] / df["sqft_living"]
    df["age"] = df["date"].dt.year - df["yr_built"]
    df["is_renovated"] = df["yr_renovated"] > 0
    df["has_basement"] = df["sqft_basement"] > 0
    return df

df = load_data()

# Matplotlib dark theme
plt.rcParams.update({
    "figure.facecolor": "#1e293b",
    "axes.facecolor": "#0f172a",
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#94a3b8",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "text.color": "#e2e8f0",
    "grid.color": "#334155",
    "grid.alpha": 0.5,
})

# Accent colors
C_PRIMARY = "#6366f1"    # indigo
C_SECONDARY = "#8b5cf6"  # violet
C_ACCENT = "#22d3ee"     # cyan
C_SUCCESS = "#22c55e"    # green
C_WARN = "#f59e0b"       # amber
C_TEXT = "#e2e8f0"

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        "<div style='text-align:center; padding:15px 0 8px 0;'>"
        "<span style='font-size:2.2rem;'>🏡</span><br>"
        "<span style='font-family:Space Grotesk,sans-serif; "
        "background:linear-gradient(135deg,#818cf8,#a78bfa); "
        "-webkit-background-clip:text; -webkit-text-fill-color:transparent; "
        "font-weight:800; font-size:1.15rem;'>HomeScope AI</span><br>"
        "<span style='color:#64748b; font-size:0.75rem; letter-spacing:1px;'>"
        "21 613 PROPRIÉTÉS &bull; SEATTLE, WA</span>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊  Exploration du marché", "🔍  Analyse de propriété"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Market Exploration
# ═══════════════════════════════════════════════════════════════════════════
with tab1:

    st.markdown(
        "<div class='tip-card'>"
        "💡 <b>Astuce :</b> Utilisez les filtres dans la barre latérale pour cibler un segment. "
        "Tout se met à jour en temps réel. Descendez pour le résumé IA !"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Filters (sidebar) ──
    with st.sidebar:
        st.header("⚙️ Filtres")

        min_price, max_price = int(df["price"].min()), int(df["price"].max())
        price_range = st.slider(
            "💰 Prix", min_value=min_price, max_value=max_price,
            value=(min_price, max_price), step=10_000, format="$%d",
            help="Glissez pour filtrer par prix.",
        )

        bedrooms_options = sorted(df["bedrooms"].unique())
        selected_bedrooms = st.multiselect(
            "🛏️ Chambres", bedrooms_options, default=bedrooms_options,
            help="Sélectionnez le nombre de chambres.",
        )

        zipcode_options = sorted(df["zipcode"].unique())
        selected_zipcodes = st.multiselect(
            "📍 Code postal", zipcode_options, default=zipcode_options,
            help="Filtrez par quartier.",
        )

        min_grade, max_grade = int(df["grade"].min()), int(df["grade"].max())
        grade_range = st.slider(
            "⭐ Grade", min_value=min_grade, max_value=max_grade,
            value=(min_grade, max_grade),
            help="1-3 faible, 7 moyen, 11-13 élevé.",
        )

        waterfront_only = st.checkbox("🌊 Front de mer seulement")

        min_year, max_year = int(df["yr_built"].min()), int(df["yr_built"].max())
        year_range = st.slider(
            "📅 Année de construction",
            min_value=min_year, max_value=max_year,
            value=(min_year, max_year),
        )

        st.markdown("---")
        st.markdown(
            "<p style='color:#64748b; font-size:0.72rem; text-align:center;'>"
            "Les résultats se mettent à jour automatiquement</p>",
            unsafe_allow_html=True,
        )

    # Apply filters
    filtered = df[
        (df["price"].between(price_range[0], price_range[1]))
        & (df["bedrooms"].isin(selected_bedrooms))
        & (df["zipcode"].isin(selected_zipcodes))
        & (df["grade"].between(grade_range[0], grade_range[1]))
        & (df["yr_built"].between(year_range[0], year_range[1]))
    ]
    if waterfront_only:
        filtered = filtered[filtered["waterfront"] == 1]

    if len(filtered) == 0:
        st.warning("Aucune propriété ne correspond. Élargissez vos filtres.")
        st.stop()

    # ── KPIs ──
    st.markdown("<p class='section-label'>Aperçu du segment</p>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Propriétés", f"{len(filtered):,}")
    col2.metric("Prix moyen", f"{filtered['price'].mean():,.0f} $")
    col3.metric("Prix médian", f"{filtered['price'].median():,.0f} $")
    col4.metric("Prix / pi²", f"{filtered['price_per_sqft'].mean():,.0f} $")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Min", f"{filtered['price'].min():,.0f} $")
    col6.metric("Max", f"{filtered['price'].max():,.0f} $")
    col7.metric("Âge moyen", f"{filtered['age'].mean():,.0f} ans")
    pct_wf_val = (filtered["waterfront"].sum() / len(filtered) * 100) if len(filtered) > 0 else 0
    col8.metric("Front de mer", f"{pct_wf_val:.1f}%")

    st.markdown("---")

    # ── Charts ──
    st.markdown("<p class='section-label'>Visualisations</p>", unsafe_allow_html=True)

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.hist(filtered["price"], bins=50, color=C_PRIMARY, edgecolor="#0f172a", alpha=0.9)
        ax1.set_title("Distribution des prix", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Prix ($)")
        ax1.set_ylabel("Propriétés")
        ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax1.grid(True, axis="y")
        plt.tight_layout()
        st.pyplot(fig1)

    with chart_col2:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        sc = ax2.scatter(
            filtered["sqft_living"], filtered["price"],
            c=filtered["grade"], cmap="cool", alpha=0.5, s=8,
        )
        ax2.set_title("Prix vs Superficie (par grade)", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Superficie (pi²)")
        ax2.set_ylabel("Prix ($)")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        cbar = plt.colorbar(sc, ax=ax2)
        cbar.set_label("Grade", color="#94a3b8")
        cbar.ax.yaxis.set_tick_params(color="#94a3b8")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        avg_by_bed = filtered.groupby("bedrooms")["price"].mean().sort_index()
        ax4.bar(avg_by_bed.index.astype(str), avg_by_bed.values, color=C_ACCENT, edgecolor="#0f172a")
        ax4.set_title("Prix moyen par chambres", fontsize=13, fontweight="bold")
        ax4.set_xlabel("Chambres")
        ax4.set_ylabel("Prix moyen ($)")
        ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax4.grid(True, axis="y")
        plt.tight_layout()
        st.pyplot(fig4)

    with chart_col4:
        fig6, ax6 = plt.subplots(figsize=(8, 4))
        top_zips = filtered.groupby("zipcode")["price"].mean().nlargest(10).sort_values()
        ax6.barh(top_zips.index.astype(str), top_zips.values, color=C_SECONDARY, edgecolor="#0f172a")
        ax6.set_title("Top 10 quartiers les plus chers", fontsize=13, fontweight="bold")
        ax6.set_xlabel("Prix moyen ($)")
        ax6.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax6.grid(True, axis="x")
        plt.tight_layout()
        st.pyplot(fig6)

    # Correlation matrix
    st.markdown("---")
    with st.expander("🔬 Matrice de corrélation (avancé)"):
        st.markdown(
            "<div class='tip-card'>"
            "<b>+1</b> = forte relation positive &nbsp; | &nbsp; "
            "<b>-1</b> = relation inverse &nbsp; | &nbsp; "
            "<b>0</b> = aucune relation"
            "</div>",
            unsafe_allow_html=True,
        )
        fig3, ax3 = plt.subplots(figsize=(12, 8))
        corr_cols = [
            "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
            "floors", "waterfront", "view", "condition", "grade",
            "sqft_above", "sqft_basement", "yr_built", "price_per_sqft", "age",
        ]
        corr = filtered[corr_cols].corr()
        sns.heatmap(
            corr, annot=True, fmt=".2f", cmap="RdYlBu_r", ax=ax3,
            square=True, linewidths=0.5, linecolor="#334155",
            cbar_kws={"shrink": 0.8},
        )
        ax3.set_title("Matrice de corrélation", fontsize=14, fontweight="bold", pad=15)
        plt.tight_layout()
        st.pyplot(fig3)

    # Map
    st.markdown("---")
    st.markdown("<p class='section-label'>Carte des propriétés</p>", unsafe_allow_html=True)
    map_data = filtered[["lat", "long"]].rename(columns={"long": "lon"})
    st.map(map_data, zoom=9)

    # ── LLM Summary ──
    st.markdown("---")
    st.markdown("<p class='section-label'>Intelligence artificielle</p>", unsafe_allow_html=True)

    if st.button("🤖  Générer un résumé IA du marché", key="btn_resume", use_container_width=True):
        with st.spinner("Gemini analyse le marché..."):
            n = len(filtered)
            mean_price = filtered["price"].mean()
            median_price = filtered["price"].median()
            min_p = filtered["price"].min()
            max_p = filtered["price"].max()
            mean_psqft = filtered["price_per_sqft"].mean()
            grade_dist = filtered["grade"].value_counts().sort_index().to_dict()
            pct_wf = (filtered["waterfront"].sum() / n * 100) if n > 0 else 0

            prompt = f"""Tu es un analyste immobilier senior. Voici les statistiques d'un segment du marché immobilier du comté de King (Seattle) :
- Nombre de propriétés : {n}
- Prix moyen : {mean_price:,.0f} $
- Prix médian : {median_price:,.0f} $
- Prix min / max : {min_p:,.0f} $ / {max_p:,.0f} $
- Prix moyen par pi² : {mean_psqft:,.0f} $
- Répartition par grade : {grade_dist}
- % front de mer : {pct_wf:.1f}%

Rédige un résumé exécutif de ce segment en 3-4 paragraphes. Identifie les tendances clés et les opportunités d'investissement."""

            response = client.models.generate_content(model=LLM_MODEL, contents=prompt)
            st.markdown(f"<div class='ai-response'>{response.text}</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Property Analysis
# ═══════════════════════════════════════════════════════════════════════════
with tab2:

    st.markdown(
        "<div class='tip-card'>"
        "💡 Choisissez un quartier, puis une propriété. L'app trouve automatiquement "
        "les comparables et vous donne un verdict IA."
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Selection ──
    st.markdown(
        "<p class='section-label'><span class='step-badge'>1</span> Sélection</p>",
        unsafe_allow_html=True,
    )

    sel_col1, sel_col2, sel_col3 = st.columns(3)
    with sel_col1:
        zip_filter = st.selectbox("📍 Quartier", sorted(df["zipcode"].unique()), key="tab2_zip")
    with sel_col2:
        beds_in_zip = sorted(df[df["zipcode"] == zip_filter]["bedrooms"].unique())
        bed_filter = st.selectbox("🛏️ Chambres", beds_in_zip, key="tab2_bed")
    with sel_col3:
        subset = df[(df["zipcode"] == zip_filter) & (df["bedrooms"] == bed_filter)]
        options = subset.apply(lambda r: f"ID {r['id']} — {r['price']:,.0f} $", axis=1)
        if len(subset) == 0:
            st.warning("Aucune propriété trouvée.")
            st.stop()
        selected_label = st.selectbox("🏠 Propriété", options.values, key="tab2_prop")

    selected_idx = options[options == selected_label].index[0]
    prop = df.loc[selected_idx]

    st.markdown("---")

    # ── Property Card ──
    st.markdown(
        "<p class='section-label'><span class='step-badge'>2</span> Fiche de la propriété</p>",
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prix", f"{prop['price']:,.0f} $")
    c2.metric("Chambres", int(prop["bedrooms"]))
    c3.metric("Salles de bain", prop["bathrooms"])
    c4.metric("Superficie", f"{int(prop['sqft_living']):,} pi²")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Terrain", f"{int(prop['sqft_lot']):,} pi²")
    c6.metric("Grade", f"{int(prop['grade'])}/13")
    c7.metric("Condition", f"{int(prop['condition'])}/5")
    c8.metric("Construite", int(prop["yr_built"]))

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Étages", prop["floors"])
    c10.metric("Vue", f"{int(prop['view'])}/4")
    c11.metric("Bord de mer", "Oui" if prop["waterfront"] else "Non")
    c12.metric("Rénovée", str(int(prop["yr_renovated"])) if prop["yr_renovated"] > 0 else "Non")

    st.map(pd.DataFrame({"lat": [prop["lat"]], "lon": [prop["long"]]}), zoom=13)

    st.markdown("---")

    # ── Comparables ──
    st.markdown(
        "<p class='section-label'><span class='step-badge'>3</span> Comparables</p>",
        unsafe_allow_html=True,
    )

    search_levels = [
        {"label": "Même quartier, même chambres, ±20%", "zip_same": True, "bed_tol": 0, "sqft_tol": 0.20},
        {"label": "Même quartier, chambres ±1, ±30%", "zip_same": True, "bed_tol": 1, "sqft_tol": 0.30},
        {"label": "Même quartier, chambres ±2, ±50%", "zip_same": True, "bed_tol": 2, "sqft_tol": 0.50},
        {"label": "Tous quartiers, chambres ±1, ±30%", "zip_same": False, "bed_tol": 1, "sqft_tol": 0.30},
    ]

    comps = pd.DataFrame()
    used_level = None
    for level in search_levels:
        sqft_low = prop["sqft_living"] * (1 - level["sqft_tol"])
        sqft_high = prop["sqft_living"] * (1 + level["sqft_tol"])
        bed_low = prop["bedrooms"] - level["bed_tol"]
        bed_high = prop["bedrooms"] + level["bed_tol"]
        mask = (
            (df["bedrooms"].between(bed_low, bed_high))
            & (df["sqft_living"].between(sqft_low, sqft_high))
            & (df.index != selected_idx)
        )
        if level["zip_same"]:
            mask = mask & (df["zipcode"] == prop["zipcode"])
        comps = df[mask]
        if len(comps) >= 3:
            used_level = level
            break

    if len(comps) == 0:
        st.warning("Aucun comparable trouvé. Essayez une autre propriété.")
    else:
        if used_level and used_level != search_levels[0]:
            st.markdown(
                f"<div class='warn-card'>🔄 Critères élargis : <b>{used_level['label']}</b></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='success-card'>✅ <b>{len(comps)}</b> comparables trouvés (critères stricts)</div>",
                unsafe_allow_html=True,
            )

        comps_display = comps.head(10)
        n_comps = len(comps)
        mean_comp_price = comps["price"].mean()
        ecart = prop["price"] - mean_comp_price
        ecart_pct = (ecart / mean_comp_price) * 100
        surcote_decote = "SURCOTE" if ecart > 0 else "DÉCOTE"

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Comparables", n_comps)
        mc2.metric("Prix moy. comps", f"{mean_comp_price:,.0f} $")
        mc3.metric(
            f"Écart ({surcote_decote})",
            f"{ecart:+,.0f} $",
            delta=f"{ecart_pct:+.1f}%",
            delta_color="inverse",
        )

        if ecart > 0:
            st.markdown(
                f"<div class='warn-card'>⚠️ <b>Surcote de {abs(ecart_pct):.1f}%</b> — "
                f"plus chère que le marché local</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"<div class='success-card'>💰 <b>Décote de {abs(ecart_pct):.1f}%</b> — "
                f"potentielle bonne affaire !</div>",
                unsafe_allow_html=True,
            )

        display_cols = ["id", "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "grade", "condition", "yr_built"]
        st.dataframe(
            comps_display[display_cols].style.format({
                "price": "${:,.0f}", "sqft_living": "{:,.0f}", "sqft_lot": "{:,.0f}",
            }),
            use_container_width=True,
        )

        st.markdown("---")

        # ── Charts ──
        st.markdown(
            "<p class='section-label'><span class='step-badge'>4</span> Comparaison visuelle</p>",
            unsafe_allow_html=True,
        )

        viz_col1, viz_col2 = st.columns(2)

        with viz_col1:
            fig5, ax5 = plt.subplots(figsize=(8, 5))
            comp_ids = comps_display["id"].astype(str).tolist()
            comp_prices = comps_display["price"].tolist()
            all_ids = comp_ids + [str(int(prop["id"]))]
            all_prices = comp_prices + [prop["price"]]
            colors = [C_ACCENT] * len(comp_ids) + [C_PRIMARY]

            ax5.bar(range(len(all_ids)), all_prices, color=colors, edgecolor="#0f172a", width=0.7)
            ax5.set_xticks(range(len(all_ids)))
            ax5.set_xticklabels(all_ids, rotation=45, ha="right", fontsize=7)
            ax5.set_title("Prix : Sélectionnée vs Comparables", fontsize=13, fontweight="bold")
            ax5.set_ylabel("Prix ($)")
            ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            ax5.grid(True, axis="y")
            ax5.annotate(
                "★ Sélectionnée",
                xy=(len(all_ids) - 1, prop["price"]),
                xytext=(len(all_ids) - 1, prop["price"] * 1.10),
                ha="center", fontsize=9, fontweight="bold", color=C_PRIMARY,
                arrowprops=dict(arrowstyle="->", color=C_PRIMARY, lw=2),
            )
            plt.tight_layout()
            st.pyplot(fig5)

        with viz_col2:
            categories = ["Prix/pi²", "Grade", "Condition", "Vue", "Superficie"]
            max_price_sqft = df["price_per_sqft"].max()
            max_sqft = df["sqft_living"].max()

            prop_values = [
                prop["price_per_sqft"] / max_price_sqft,
                prop["grade"] / 13,
                prop["condition"] / 5,
                max(prop["view"] / 4, 0.05),
                prop["sqft_living"] / max_sqft,
            ]
            comp_values = [
                comps["price_per_sqft"].mean() / max_price_sqft,
                comps["grade"].mean() / 13,
                comps["condition"].mean() / 5,
                max(comps["view"].mean() / 4, 0.05),
                comps["sqft_living"].mean() / max_sqft,
            ]

            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            prop_values += prop_values[:1]
            comp_values += comp_values[:1]
            angles += angles[:1]

            fig7, ax7 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax7.set_facecolor("#0f172a")
            fig7.set_facecolor("#1e293b")

            ax7.fill(angles, prop_values, color=C_PRIMARY, alpha=0.25)
            ax7.plot(angles, prop_values, color=C_PRIMARY, linewidth=2.5, label="Sélectionnée")

            ax7.fill(angles, comp_values, color=C_ACCENT, alpha=0.15)
            ax7.plot(angles, comp_values, color=C_ACCENT, linewidth=2.5, label="Moy. comparables")

            ax7.set_xticks(angles[:-1])
            ax7.set_xticklabels(categories, fontsize=9, color="#e2e8f0")
            ax7.set_yticklabels([])
            ax7.set_title("Radar comparatif", fontsize=13, fontweight="bold", color="#e2e8f0", pad=20)
            ax7.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=9,
                       facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
            ax7.spines["polar"].set_color("#334155")
            ax7.grid(color="#334155")
            plt.tight_layout()
            st.pyplot(fig7)

        st.markdown("---")

        # ── LLM Recommendation ──
        st.markdown(
            "<p class='section-label'><span class='step-badge'>5</span> Verdict IA</p>",
            unsafe_allow_html=True,
        )

        if st.button("🤖  Obtenir la recommandation IA", key="btn_reco", use_container_width=True):
            with st.spinner("Gemini analyse la propriété..."):
                renovated = str(int(prop["yr_renovated"])) if prop["yr_renovated"] > 0 else "Non"
                waterfront_str = "Oui" if prop["waterfront"] else "Non"

                prompt_reco = f"""Tu es un analyste immobilier senior. Évalue cette propriété pour un investisseur :

PROPRIÉTÉ ANALYSÉE :
- Prix : {prop['price']:,.0f} $
- Chambres : {int(prop['bedrooms'])} | Salles de bain : {prop['bathrooms']}
- Superficie : {int(prop['sqft_living'])} pi² | Terrain : {int(prop['sqft_lot'])} pi²
- Grade : {int(prop['grade'])}/13 | Condition : {int(prop['condition'])}/5
- Année de construction : {int(prop['yr_built'])} | Rénovée : {renovated}
- Front de mer : {waterfront_str} | Vue : {int(prop['view'])}/4

ANALYSE COMPARATIVE :
- Nombre de comparables trouvés : {n_comps}
- Prix moyen des comparables : {mean_comp_price:,.0f} $
- Écart vs comparables : {ecart:+,.0f} $ ({ecart_pct:+.1f}%)
- Statut : {surcote_decote}

Rédige une recommandation d'investissement en 3-4 paragraphes. Inclus : évaluation du prix, forces et faiblesses, verdict final (Acheter / À surveiller / Éviter) avec justification."""

                response_reco = client.models.generate_content(model=LLM_MODEL, contents=prompt_reco)
                st.markdown(f"<div class='ai-response'>{response_reco.text}</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align:center; padding:20px 0;'>"
    "<span style='font-family:Space Grotesk,sans-serif; color:#475569; font-size:0.85rem; font-weight:600;'>"
    "HomeScope AI</span>"
    "<span style='color:#334155; font-size:0.85rem;'> &nbsp;|&nbsp; </span>"
    "<span style='color:#64748b; font-size:0.75rem;'>"
    "Streamlit &bull; Pandas &bull; Matplotlib &bull; Google Gemini"
    "</span>"
    "</div>",
    unsafe_allow_html=True,
)
