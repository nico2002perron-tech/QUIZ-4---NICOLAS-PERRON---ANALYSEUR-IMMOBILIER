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
    page_title="Analyseur Immobilier - King County",
    page_icon="🏠",
    layout="wide",
)

# ---------------------------------------------------------------------------
# CSS personnalisé
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Fond principal */
    .stApp {
        background: linear-gradient(180deg, #0a1628 0%, #111d33 100%);
    }

    /* Titre principal */
    h1 {
        color: #e8c547 !important;
        text-align: center;
        font-size: 2.4rem !important;
        letter-spacing: 1px;
        border-bottom: 2px solid #1e3a5f;
        padding-bottom: 15px;
    }

    /* Sous-titres */
    h2, h3 {
        color: #7eb8da !important;
    }

    /* Onglets */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        justify-content: center;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #112240;
        color: #8892b0;
        border-radius: 8px 8px 0 0;
        padding: 10px 28px;
        font-weight: 600;
        border: 1px solid #1e3a5f;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3a5f !important;
        color: #e8c547 !important;
        border-bottom: 3px solid #e8c547;
    }

    /* Cartes KPI */
    [data-testid="stMetric"] {
        background: linear-gradient(135deg, #112240, #1a2f4e);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    [data-testid="stMetricLabel"] {
        color: #8892b0 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    [data-testid="stMetricValue"] {
        color: #e8c547 !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a, #112240);
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] h2 {
        color: #e8c547 !important;
        font-size: 1.1rem !important;
    }

    /* Boutons */
    .stButton > button {
        background: linear-gradient(135deg, #e8c547, #d4a833) !important;
        color: #0a1628 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(232, 197, 71, 0.3) !important;
    }

    /* Séparateurs */
    hr {
        border-color: #1e3a5f !important;
    }

    /* Dataframes */
    [data-testid="stDataFrame"] {
        border: 1px solid #1e3a5f;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Titre
# ---------------------------------------------------------------------------
st.title("🏠 Analyseur Immobilier — King County (Seattle)")
st.markdown(
    "<p style='text-align:center; color:#8892b0; margin-top:-10px;'>"
    "Exploration du marché &bull; Analyse de propriétés &bull; Intelligence artificielle"
    "</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Étape 0 — Chargement et préparation des données
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

# Palette matplotlib globale (fond sombre)
plt.rcParams.update({
    "figure.facecolor": "#112240",
    "axes.facecolor": "#0a1628",
    "axes.edgecolor": "#1e3a5f",
    "axes.labelcolor": "#8892b0",
    "xtick.color": "#8892b0",
    "ytick.color": "#8892b0",
    "text.color": "#ccd6f6",
    "grid.color": "#1e3a5f",
    "grid.alpha": 0.4,
})

# ---------------------------------------------------------------------------
# Onglets
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Exploration du marché", "🔍 Analyse d'une propriété"])

# ═══════════════════════════════════════════════════════════════════════════
# ONGLET 1 — Exploration du marché
# ═══════════════════════════════════════════════════════════════════════════
with tab1:

    # --- A. Filtres interactifs (sidebar) ---
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Filtres")

    min_price, max_price = int(df["price"].min()), int(df["price"].max())
    price_range = st.sidebar.slider(
        "Fourchette de prix ($)",
        min_value=min_price, max_value=max_price,
        value=(min_price, max_price),
        step=10_000,
        format="$%d",
    )

    bedrooms_options = sorted(df["bedrooms"].unique())
    selected_bedrooms = st.sidebar.multiselect(
        "Nombre de chambres", bedrooms_options, default=bedrooms_options
    )

    zipcode_options = sorted(df["zipcode"].unique())
    selected_zipcodes = st.sidebar.multiselect(
        "Code postal (zipcode)", zipcode_options, default=zipcode_options
    )

    min_grade, max_grade = int(df["grade"].min()), int(df["grade"].max())
    grade_range = st.sidebar.slider(
        "Grade de construction",
        min_value=min_grade, max_value=max_grade,
        value=(min_grade, max_grade),
    )

    waterfront_only = st.sidebar.checkbox("Front de mer uniquement")

    min_year, max_year = int(df["yr_built"].min()), int(df["yr_built"].max())
    year_range = st.sidebar.slider(
        "Année de construction",
        min_value=min_year, max_value=max_year,
        value=(min_year, max_year),
    )

    # Application des filtres
    filtered = df[
        (df["price"].between(price_range[0], price_range[1]))
        & (df["bedrooms"].isin(selected_bedrooms))
        & (df["zipcode"].isin(selected_zipcodes))
        & (df["grade"].between(grade_range[0], grade_range[1]))
        & (df["yr_built"].between(year_range[0], year_range[1]))
    ]
    if waterfront_only:
        filtered = filtered[filtered["waterfront"] == 1]

    # --- B. Métriques clés (KPIs) ---
    st.subheader("📈 Métriques clés")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Propriétés", f"{len(filtered):,}")
    col2.metric("Prix moyen", f"{filtered['price'].mean():,.0f} $")
    col3.metric("Prix médian", f"{filtered['price'].median():,.0f} $")
    col4.metric("Prix / pi²", f"{filtered['price_per_sqft'].mean():,.0f} $")

    # Rangée bonus
    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Prix minimum", f"{filtered['price'].min():,.0f} $")
    col6.metric("Prix maximum", f"{filtered['price'].max():,.0f} $")
    col7.metric("Âge moyen", f"{filtered['age'].mean():,.0f} ans")
    col8.metric("% Front de mer", f"{(filtered['waterfront'].sum() / len(filtered) * 100) if len(filtered) > 0 else 0:.1f}%")

    st.markdown("---")

    # --- C. Visualisations (matplotlib) ---
    st.subheader("📊 Visualisations")

    # Graphiques en 2 colonnes
    chart_col1, chart_col2 = st.columns(2)

    # 1) Histogramme de la distribution des prix
    with chart_col1:
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        ax1.hist(filtered["price"], bins=50, color="#e8c547", edgecolor="#0a1628", alpha=0.9)
        ax1.set_title("Distribution des prix", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Prix ($)")
        ax1.set_ylabel("Nombre de propriétés")
        ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax1.grid(True, axis="y")
        plt.tight_layout()
        st.pyplot(fig1)

    # 2) Scatter plot : prix vs superficie, coloré par grade
    with chart_col2:
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        scatter = ax2.scatter(
            filtered["sqft_living"],
            filtered["price"],
            c=filtered["grade"],
            cmap="plasma",
            alpha=0.5,
            s=8,
        )
        ax2.set_title("Prix vs Superficie (par grade)", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Superficie habitable (pi²)")
        ax2.set_ylabel("Prix ($)")
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label("Grade", color="#8892b0")
        cbar.ax.yaxis.set_tick_params(color="#8892b0")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#8892b0")
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig2)

    chart_col3, chart_col4 = st.columns(2)

    # 3) Diagramme en barres : prix moyen par nombre de chambres
    with chart_col3:
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        avg_by_bed = filtered.groupby("bedrooms")["price"].mean().sort_index()
        bars = ax4.bar(avg_by_bed.index.astype(str), avg_by_bed.values, color="#4fc3f7", edgecolor="#0a1628")
        ax4.set_title("Prix moyen par nombre de chambres", fontsize=13, fontweight="bold")
        ax4.set_xlabel("Chambres")
        ax4.set_ylabel("Prix moyen ($)")
        ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax4.grid(True, axis="y")
        plt.tight_layout()
        st.pyplot(fig4)

    # 4) Top 10 zipcodes les plus chers
    with chart_col4:
        fig6, ax6 = plt.subplots(figsize=(8, 4))
        top_zips = filtered.groupby("zipcode")["price"].mean().nlargest(10).sort_values()
        ax6.barh(top_zips.index.astype(str), top_zips.values, color="#e8c547", edgecolor="#0a1628")
        ax6.set_title("Top 10 des codes postaux (prix moyen)", fontsize=13, fontweight="bold")
        ax6.set_xlabel("Prix moyen ($)")
        ax6.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax6.grid(True, axis="x")
        plt.tight_layout()
        st.pyplot(fig6)

    # 5) Matrice de corrélation (pleine largeur)
    st.markdown("---")
    fig3, ax3 = plt.subplots(figsize=(12, 8))
    corr_cols = [
        "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
        "floors", "waterfront", "view", "condition", "grade",
        "sqft_above", "sqft_basement", "yr_built", "price_per_sqft", "age",
    ]
    corr = filtered[corr_cols].corr()
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="YlOrBr", ax=ax3,
        square=True, linewidths=0.5, linecolor="#1e3a5f",
        cbar_kws={"shrink": 0.8},
    )
    ax3.set_title("Matrice de corrélation", fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout()
    st.pyplot(fig3)

    # 6) Carte interactive des propriétés
    st.markdown("---")
    st.subheader("🗺️ Carte des propriétés")
    map_data = filtered[["lat", "long"]].rename(columns={"long": "lon"})
    st.map(map_data, zoom=9)

    # --- D. Résumé généré par LLM ---
    st.markdown("---")
    st.subheader("🤖 Résumé du marché (IA)")
    if st.button("Générer un résumé du marché", key="btn_resume"):
        with st.spinner("L'IA analyse le marché..."):
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

            response = client.models.generate_content(
                model=LLM_MODEL,
                contents=prompt,
            )
            st.markdown(response.text)

# ═══════════════════════════════════════════════════════════════════════════
# ONGLET 2 — Analyse d'une propriété
# ═══════════════════════════════════════════════════════════════════════════
with tab2:

    # --- A. Sélection de la propriété ---
    st.subheader("🔎 Sélection de la propriété")

    col_zip, col_bed = st.columns(2)
    with col_zip:
        zip_filter = st.selectbox("Code postal", sorted(df["zipcode"].unique()), key="tab2_zip")
    with col_bed:
        beds_in_zip = sorted(df[df["zipcode"] == zip_filter]["bedrooms"].unique())
        bed_filter = st.selectbox("Nombre de chambres", beds_in_zip, key="tab2_bed")

    subset = df[(df["zipcode"] == zip_filter) & (df["bedrooms"] == bed_filter)]
    options = subset.apply(lambda r: f"ID {r['id']} — {r['price']:,.0f} $", axis=1)

    if len(subset) == 0:
        st.warning("Aucune propriété trouvée avec ces critères.")
        st.stop()

    selected_label = st.selectbox("Propriété", options.values, key="tab2_prop")
    selected_idx = options[options == selected_label].index[0]
    prop = df.loc[selected_idx]

    st.markdown("---")

    # --- B. Fiche descriptive ---
    st.subheader("🏡 Fiche descriptive")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Prix", f"{prop['price']:,.0f} $")
    c2.metric("Chambres", int(prop["bedrooms"]))
    c3.metric("Salles de bain", prop["bathrooms"])
    c4.metric("Superficie", f"{int(prop['sqft_living']):,} pi²")

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Terrain", f"{int(prop['sqft_lot']):,} pi²")
    c6.metric("Grade", f"{int(prop['grade'])}/13")
    c7.metric("Condition", f"{int(prop['condition'])}/5")
    c8.metric("Année", int(prop["yr_built"]))

    c9, c10, c11, c12 = st.columns(4)
    c9.metric("Étages", prop["floors"])
    c10.metric("Vue", f"{int(prop['view'])}/4")
    c11.metric("Front de mer", "Oui" if prop["waterfront"] else "Non")
    c12.metric("Rénovée", str(int(prop["yr_renovated"])) if prop["yr_renovated"] > 0 else "Non")

    # Localisation sur carte
    st.map(
        pd.DataFrame({"lat": [prop["lat"]], "lon": [prop["long"]]}),
        zoom=13,
    )

    st.markdown("---")

    # --- C. Recherche de comparables (élargissement automatique) ---
    st.subheader("⚖️ Propriétés comparables")

    # Niveaux de recherche : on élargit progressivement si pas assez de résultats
    search_levels = [
        {"label": "Strict (même zip, même chambres, ±20% sup.)", "zip_same": True, "bed_tol": 0, "sqft_tol": 0.20},
        {"label": "Élargi (même zip, chambres ±1, ±30% sup.)", "zip_same": True, "bed_tol": 1, "sqft_tol": 0.30},
        {"label": "Large (même zip, chambres ±2, ±50% sup.)", "zip_same": True, "bed_tol": 2, "sqft_tol": 0.50},
        {"label": "Très large (tous zips, chambres ±1, ±30% sup.)", "zip_same": False, "bed_tol": 1, "sqft_tol": 0.30},
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
        st.info("Aucun comparable trouvé, même avec des critères élargis.")
    else:
        if used_level and used_level != search_levels[0]:
            st.caption(f"🔄 Critères élargis automatiquement : {used_level['label']}")
        comps_display = comps.head(10)
        n_comps = len(comps)
        mean_comp_price = comps["price"].mean()
        ecart = prop["price"] - mean_comp_price
        ecart_pct = (ecart / mean_comp_price) * 100
        surcote_decote = "SURCOTE" if ecart > 0 else "DÉCOTE"

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Comparables trouvés", n_comps)
        mc2.metric("Prix moyen comparables", f"{mean_comp_price:,.0f} $")
        mc3.metric(
            f"Écart ({surcote_decote})",
            f"{ecart:+,.0f} $",
            delta=f"{ecart_pct:+.1f}%",
            delta_color="inverse",
        )

        display_cols = ["id", "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "grade", "condition", "yr_built"]
        st.dataframe(
            comps_display[display_cols].style.format({
                "price": "${:,.0f}",
                "sqft_living": "{:,.0f}",
                "sqft_lot": "{:,.0f}",
            }),
            use_container_width=True,
        )

        st.markdown("---")

        # --- D. Visualisations comparatives (matplotlib) ---
        st.subheader("📊 Comparaison visuelle")

        viz_col1, viz_col2 = st.columns(2)

        # Graphique en barres : prix
        with viz_col1:
            fig5, ax5 = plt.subplots(figsize=(8, 5))
            comp_ids = comps_display["id"].astype(str).tolist()
            comp_prices = comps_display["price"].tolist()

            all_ids = comp_ids + [str(int(prop["id"]))]
            all_prices = comp_prices + [prop["price"]]
            colors = ["#4fc3f7"] * len(comp_ids) + ["#e8c547"]

            ax5.bar(range(len(all_ids)), all_prices, color=colors, edgecolor="#0a1628")
            ax5.set_xticks(range(len(all_ids)))
            ax5.set_xticklabels(all_ids, rotation=45, ha="right", fontsize=7)
            ax5.set_title("Prix : Sélectionnée vs Comparables", fontsize=13, fontweight="bold")
            ax5.set_xlabel("ID de la propriété")
            ax5.set_ylabel("Prix ($)")
            ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
            ax5.grid(True, axis="y")

            ax5.annotate(
                "★ Sélectionnée",
                xy=(len(all_ids) - 1, prop["price"]),
                xytext=(len(all_ids) - 1, prop["price"] * 1.10),
                ha="center", fontsize=9, fontweight="bold", color="#e8c547",
                arrowprops=dict(arrowstyle="->", color="#e8c547"),
            )
            plt.tight_layout()
            st.pyplot(fig5)

        # Graphique radar (bonus)
        with viz_col2:
            categories = ["Prix / pi²", "Grade", "Condition", "Vue", "Superficie\n(norm.)"]
            max_price_sqft = df["price_per_sqft"].max()
            max_sqft = df["sqft_living"].max()

            prop_values = [
                prop["price_per_sqft"] / max_price_sqft,
                prop["grade"] / 13,
                prop["condition"] / 5,
                prop["view"] / 4,
                prop["sqft_living"] / max_sqft,
            ]
            comp_values = [
                comps["price_per_sqft"].mean() / max_price_sqft,
                comps["grade"].mean() / 13,
                comps["condition"].mean() / 5,
                comps["view"].mean() / 4,
                comps["sqft_living"].mean() / max_sqft,
            ]

            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
            prop_values += prop_values[:1]
            comp_values += comp_values[:1]
            angles += angles[:1]

            fig7, ax7 = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
            ax7.set_facecolor("#0a1628")
            fig7.set_facecolor("#112240")

            ax7.fill(angles, prop_values, color="#e8c547", alpha=0.25)
            ax7.plot(angles, prop_values, color="#e8c547", linewidth=2, label="Sélectionnée")

            ax7.fill(angles, comp_values, color="#4fc3f7", alpha=0.15)
            ax7.plot(angles, comp_values, color="#4fc3f7", linewidth=2, label="Moy. comparables")

            ax7.set_xticks(angles[:-1])
            ax7.set_xticklabels(categories, fontsize=9, color="#ccd6f6")
            ax7.set_yticklabels([])
            ax7.set_title("Radar comparatif", fontsize=13, fontweight="bold", color="#ccd6f6", pad=20)
            ax7.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)
            ax7.spines["polar"].set_color("#1e3a5f")
            ax7.grid(color="#1e3a5f")
            plt.tight_layout()
            st.pyplot(fig7)

        st.markdown("---")

        # --- E. Recommandation générée par LLM ---
        st.subheader("🤖 Recommandation d'investissement (IA)")
        if st.button("Générer une recommandation", key="btn_reco"):
            with st.spinner("L'IA analyse la propriété..."):
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

                response_reco = client.models.generate_content(
                    model=LLM_MODEL,
                    contents=prompt_reco,
                )
                st.markdown(response_reco.text)

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#4a5568; font-size:0.8rem;'>"
    "Analyseur Immobilier — King County | Données : kc_house_data.csv | "
    "Propulsé par Streamlit & Google Gemini"
    "</p>",
    unsafe_allow_html=True,
)
