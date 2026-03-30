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
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
LLM_MODEL = "gemini-3.1-flash-lite-preview"

st.set_page_config(page_title="Analyseur Immobilier - King County", layout="wide")
st.title("Analyseur Immobilier — King County (Seattle)")

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

# ---------------------------------------------------------------------------
# Onglets
# ---------------------------------------------------------------------------
tab1, tab2 = st.tabs(["Exploration du marché", "Analyse d'une propriété"])

# ═══════════════════════════════════════════════════════════════════════════
# ONGLET 1 — Exploration du marché
# ═══════════════════════════════════════════════════════════════════════════
with tab1:

    # --- A. Filtres interactifs (sidebar) ---
    st.sidebar.header("Filtres — Exploration")

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
    st.subheader("Métriques clés")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nombre de propriétés", f"{len(filtered):,}")
    col2.metric("Prix moyen", f"{filtered['price'].mean():,.0f} $")
    col3.metric("Prix médian", f"{filtered['price'].median():,.0f} $")
    col4.metric("Prix moyen / pi²", f"{filtered['price_per_sqft'].mean():,.0f} $")

    # --- C. Visualisations (matplotlib) ---
    st.subheader("Visualisations")

    # 1) Histogramme de la distribution des prix
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.hist(filtered["price"], bins=50, color="#4C9BE8", edgecolor="white")
    ax1.set_title("Distribution des prix")
    ax1.set_xlabel("Prix ($)")
    ax1.set_ylabel("Nombre de propriétés")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    st.pyplot(fig1)

    # 2) Scatter plot : prix vs superficie, coloré par grade
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    scatter = ax2.scatter(
        filtered["sqft_living"],
        filtered["price"],
        c=filtered["grade"],
        cmap="viridis",
        alpha=0.4,
        s=10,
    )
    ax2.set_title("Prix vs Superficie habitable (coloré par grade)")
    ax2.set_xlabel("Superficie habitable (pi²)")
    ax2.set_ylabel("Prix ($)")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.colorbar(scatter, ax=ax2, label="Grade")
    plt.tight_layout()
    st.pyplot(fig2)

    # 3) Matrice de corrélation (heatmap)
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    corr_cols = [
        "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
        "floors", "waterfront", "view", "condition", "grade",
        "sqft_above", "sqft_basement", "yr_built", "price_per_sqft", "age",
    ]
    corr = filtered[corr_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax3, square=True, linewidths=0.5)
    ax3.set_title("Matrice de corrélation")
    plt.tight_layout()
    st.pyplot(fig3)

    # 4) Diagramme en barres : prix moyen par nombre de chambres
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    avg_by_bed = filtered.groupby("bedrooms")["price"].mean().sort_index()
    ax4.bar(avg_by_bed.index.astype(str), avg_by_bed.values, color="#2ECC71", edgecolor="white")
    ax4.set_title("Prix moyen par nombre de chambres")
    ax4.set_xlabel("Nombre de chambres")
    ax4.set_ylabel("Prix moyen ($)")
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    plt.tight_layout()
    st.pyplot(fig4)

    # --- D. Résumé généré par LLM ---
    st.subheader("Résumé du marché (IA)")
    if st.button("Générer un résumé du marché", key="btn_resume"):
        with st.spinner("Analyse en cours..."):
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
    st.subheader("Sélection de la propriété")

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

    # --- B. Fiche descriptive ---
    st.subheader("Fiche descriptive")
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

    # --- C. Recherche de comparables ---
    st.subheader("Propriétés comparables")

    sqft_low = prop["sqft_living"] * 0.8
    sqft_high = prop["sqft_living"] * 1.2

    comps = df[
        (df["zipcode"] == prop["zipcode"])
        & (df["bedrooms"] == prop["bedrooms"])
        & (df["sqft_living"].between(sqft_low, sqft_high))
        & (df.index != selected_idx)
    ]

    if len(comps) == 0:
        st.info("Aucun comparable trouvé avec ces critères.")
    else:
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

        # --- D. Visualisation comparative (matplotlib) ---
        st.subheader("Comparaison visuelle")

        fig5, ax5 = plt.subplots(figsize=(10, 5))
        comp_ids = comps_display["id"].astype(str).tolist()
        comp_prices = comps_display["price"].tolist()

        all_ids = comp_ids + [str(int(prop["id"]))]
        all_prices = comp_prices + [prop["price"]]
        colors = ["#4C9BE8"] * len(comp_ids) + ["#E74C3C"]

        bars = ax5.bar(range(len(all_ids)), all_prices, color=colors, edgecolor="white")
        ax5.set_xticks(range(len(all_ids)))
        ax5.set_xticklabels(all_ids, rotation=45, ha="right", fontsize=8)
        ax5.set_title("Prix : Propriété sélectionnée vs Comparables")
        ax5.set_xlabel("ID de la propriété")
        ax5.set_ylabel("Prix ($)")
        ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

        # Annotation sur la propriété sélectionnée
        ax5.annotate(
            "Sélectionnée",
            xy=(len(all_ids) - 1, prop["price"]),
            xytext=(len(all_ids) - 1, prop["price"] * 1.08),
            ha="center",
            fontsize=9,
            fontweight="bold",
            color="#E74C3C",
            arrowprops=dict(arrowstyle="->", color="#E74C3C"),
        )
        plt.tight_layout()
        st.pyplot(fig5)

        # --- E. Recommandation générée par LLM ---
        st.subheader("Recommandation d'investissement (IA)")
        if st.button("Générer une recommandation", key="btn_reco"):
            with st.spinner("Analyse en cours..."):
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
