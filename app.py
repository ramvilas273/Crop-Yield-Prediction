import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Crop Yield Predictor", page_icon="🌾", layout="wide")

BASE = Path(__file__).parent
TARGET = "hg/ha_yield"
RAIN, PEST, TEMP = "average_rain_fall_mm_per_year", "pesticides_tonnes", "avg_temp"
FEATURES = ["Area", "Item", "Year", RAIN, PEST, TEMP]
GREEN, RED = "#2E7D32", "#D32F2F"

# Results of the model comparison in the project notebook (random 80/20 split)
NOTEBOOK_RESULTS = pd.DataFrame([
    ("KNN (k=2)",         0.9865, 4098,  9910,  0.081),
    ("Random Forest",     0.9842, 4021,  10687, 0.086),
    ("Gradient Boosting", 0.9493, 11888, 19178, 0.345),
    ("Decision Tree",     0.9224, 14029, 23713, 0.457),
    ("Linear Regression", 0.7487, 29921, 42681, 0.899),
    ("Lasso",             0.7486, 29906, 42686, 0.897),
    ("Ridge",             0.7485, 29885, 42692, 0.895),
], columns=["Model", "R²", "MAE (hg/ha)", "RMSE (hg/ha)", "MAPE"])


@st.cache_resource
def load_model():
    return joblib.load(BASE / "crop_yield_model.pkl")


@st.cache_data
def load_data():
    return pd.read_csv(BASE / "yield_clean.csv")


@st.cache_data
def load_meta():
    return json.loads((BASE / "model_meta.json").read_text())


model, df, meta = load_model(), load_data(), load_meta()
countries, crops = sorted(df["Area"].unique()), sorted(df["Item"].unique())
country_defaults = df.groupby("Area")[[RAIN, PEST, TEMP]].median()
known_pairs = set(zip(df["Area"], df["Item"]))
crop_avg = df.groupby("Item")[TARGET].mean()


def predict(country, crop, year, rain, pest, temp):
    row = pd.DataFrame([[country, crop, year, rain, pest, temp]], columns=FEATURES)
    return float(model.predict(row)[0])


def tree_range(country, crop, year, rain, pest, temp):
    """10th-90th percentile of the individual trees' predictions (tree agreement, not a confidence interval)."""
    row = pd.DataFrame([[country, crop, year, rain, pest, temp]], columns=FEATURES)
    x = model.named_steps["pre"].transform(row)
    per_tree = np.array([t.predict(x)[0] for t in model.named_steps["rf"].estimators_])
    return np.percentile(per_tree, [10, 90])


def t_ha(v):
    return f"{v / 10_000:,.2f} t/ha"


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🌾 Inputs")
    country = st.selectbox("Country", countries, index=countries.index("India"))
    crop = st.selectbox("Crop", crops, index=crops.index("Wheat"))
    year = st.slider("Year", int(df["Year"].min()), int(df["Year"].max()), int(df["Year"].max()))

    d = country_defaults.loc[country]
    rain = st.number_input("Annual rainfall (mm)", float(df[RAIN].min()), float(df[RAIN].max()),
                           float(d[RAIN]), step=10.0, key=f"rain_{country}")
    pest = st.number_input("Pesticides (tonnes)", float(df[PEST].min()), float(df[PEST].max()),
                           float(d[PEST]), step=100.0, key=f"pest_{country}")
    temp = st.number_input("Average temperature (°C)", float(df[TEMP].min()), float(df[TEMP].max()),
                           float(d[TEMP]), step=0.1, key=f"temp_{country}")
    st.caption(f"Rainfall, pesticides and temperature start at {country}'s historical median.")

pred = predict(country, crop, year, rain, pest, temp)
low, high = tree_range(country, crop, year, rain, pest, temp)
history = df[(df["Area"] == country) & (df["Item"] == crop)]
has_history = (country, crop) in known_pairs

st.title("Crop Yield Predictor")
st.caption("Predicts yield in hectograms per hectare (hg/ha) from country, crop, year, rainfall, "
           "pesticide use and temperature. FAO / Kaggle data: 101 countries, 10 crops, 1990–2013.")

tab_predict, tab_whatif, tab_data, tab_model = st.tabs(["Prediction", "What-if", "Data insights", "Model"])

# ------------------------------------------------------------------ prediction
with tab_predict:
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted yield", f"{pred:,.0f} hg/ha", t_ha(pred), delta_color="off", delta_arrow="off")
    c2.metric(f"vs. average {crop.lower()}", f"{(pred / crop_avg[crop] - 1) * 100:+.0f}%",
              help=f"Dataset-wide average for this crop: {crop_avg[crop]:,.0f} hg/ha")
    if has_history:
        hist_mean = history[TARGET].mean()
        c3.metric(f"vs. {country}'s history", f"{(pred / hist_mean - 1) * 100:+.0f}%",
                  help=f"{country}'s average {crop.lower()} yield in the data: {hist_mean:,.0f} hg/ha")
    else:
        c3.metric(f"vs. {country}'s history", "n/a")

    st.caption(f"Tree agreement range: {low:,.0f} to {high:,.0f} hg/ha "
               "(10th–90th percentile of the forest's trees; a rough guide, not a confidence interval).")

    if has_history:
        series = history.groupby("Year", as_index=False)[TARGET].mean().rename(columns={TARGET: "Yield (hg/ha)"})
        line = alt.Chart(series).mark_line(point=True, color=GREEN).encode(
            x=alt.X("Year:Q", axis=alt.Axis(format="d"), scale=alt.Scale(zero=False)),
            y=alt.Y("Yield (hg/ha):Q", scale=alt.Scale(zero=False)),
            tooltip=["Year", alt.Tooltip("Yield (hg/ha):Q", format=",.0f")])
        point = alt.Chart(pd.DataFrame({"Year": [year], "Yield (hg/ha)": [pred]})).mark_point(
            color=RED, size=180, filled=True).encode(x="Year:Q", y="Yield (hg/ha):Q",
                                                      tooltip=["Year", alt.Tooltip("Yield (hg/ha):Q", format=",.0f")])
        st.subheader(f"{crop} in {country}: history and your prediction")
        st.altair_chart((line + point).properties(height=320), width="stretch")
        st.caption("Green: yield recorded in the data. Red: the model's prediction for the inputs in the sidebar.")
    else:
        st.warning(f"The data has no record of **{crop}** in **{country}**. The prediction is an extrapolation "
                   "from other countries and crops, so treat it with extra caution.")

# ------------------------------------------------------------------ what-if
with tab_whatif:
    st.subheader("How does the prediction change?")
    vary = st.radio("Vary", ["Temperature", "Rainfall", "Pesticides"], horizontal=True)
    col = {"Temperature": TEMP, "Rainfall": RAIN, "Pesticides": PEST}[vary]
    grid = np.linspace(df[col].min(), df[col].max(), 60)
    sweep = pd.DataFrame([[country, crop, year, rain, pest, temp]] * len(grid), columns=FEATURES)
    sweep[col] = grid
    curve = pd.DataFrame({vary: grid, "Predicted yield (hg/ha)": model.predict(sweep)})
    chart = alt.Chart(curve).mark_line(color=GREEN).encode(
        x=alt.X(f"{vary}:Q", scale=alt.Scale(zero=False)),
        y=alt.Y("Predicted yield (hg/ha):Q", scale=alt.Scale(zero=False)),
        tooltip=[alt.Tooltip(f"{vary}:Q", format=",.1f"), alt.Tooltip("Predicted yield (hg/ha):Q", format=",.0f")])
    st.altair_chart(chart.properties(height=300), width="stretch")
    st.caption("Other inputs stay at the sidebar values. This shows the model's learned pattern, not a causal effect.")

    st.subheader("All crops under these conditions")
    comp = pd.DataFrame({"Crop": crops, "Predicted yield (hg/ha)": [predict(country, c, year, rain, pest, temp) for c in crops]})
    bar = alt.Chart(comp).mark_bar(color=GREEN).encode(
        x=alt.X("Predicted yield (hg/ha):Q"), y=alt.Y("Crop:N", sort="-x"),
        tooltip=["Crop", alt.Tooltip("Predicted yield (hg/ha):Q", format=",.0f")])
    st.altair_chart(bar.properties(height=300), width="stretch")

# ------------------------------------------------------------------ data insights
with tab_data:
    left, right = st.columns(2)
    with left:
        st.subheader("Typical yield by crop")
        med = df.groupby("Item", as_index=False)[TARGET].median().rename(columns={TARGET: "Median yield (hg/ha)"})
        st.altair_chart(alt.Chart(med).mark_bar(color=GREEN).encode(
            x="Median yield (hg/ha):Q", y=alt.Y("Item:N", sort="-x", title="Crop"),
            tooltip=["Item", alt.Tooltip("Median yield (hg/ha):Q", format=",.0f")]).properties(height=320),
            width="stretch")
        st.caption("Root crops and plantains yield several times more per hectare than cereals and oilseeds.")
    with right:
        st.subheader("Average yield over time")
        trend = df.groupby("Year", as_index=False)[TARGET].mean().rename(columns={TARGET: "Mean yield (hg/ha)"})
        st.altair_chart(alt.Chart(trend).mark_line(point=True, color=GREEN).encode(
            x=alt.X("Year:Q", axis=alt.Axis(format="d"), scale=alt.Scale(zero=False)),
            y=alt.Y("Mean yield (hg/ha):Q", scale=alt.Scale(zero=False)),
            tooltip=["Year", alt.Tooltip("Mean yield (hg/ha):Q", format=",.0f")]).properties(height=320),
            width="stretch")
        st.caption("Roughly +35% from 1990 to 2013. The year 2003 is missing from the data.")

    st.subheader(f"Top countries for {crop}")
    top = (df[df["Item"] == crop].groupby("Area").agg(yield_mean=(TARGET, "mean"), rows=(TARGET, "size"))
           .query("rows >= 10").sort_values("yield_mean", ascending=False).head(10).reset_index()
           .rename(columns={"yield_mean": "Mean yield (hg/ha)", "Area": "Country"}))
    st.altair_chart(alt.Chart(top).mark_bar(color=GREEN).encode(
        x="Mean yield (hg/ha):Q", y=alt.Y("Country:N", sort="-x"),
        tooltip=["Country", alt.Tooltip("Mean yield (hg/ha):Q", format=",.0f")]).properties(height=300),
        width="stretch")
    st.caption("Countries with at least 10 records for this crop.")

# ------------------------------------------------------------------ model
with tab_model:
    rs, un = meta["random_split"], meta["unseen_countries"]
    st.subheader("How reliable is it?")
    m1, m2, m3 = st.columns(3)
    m1.metric("Countries seen in training", f"R² {rs['r2']:.3f}", f"MAE {rs['mae']:,.0f} hg/ha", delta_color="off", delta_arrow="off",
              help="Random 80/20 split of rows, like the notebook.")
    m2.metric("Countries never seen", f"R² {un['r2']:.2f}", f"MAE {un['mae']:,.0f} hg/ha", delta_color="off", delta_arrow="off",
              help="5-fold GroupKFold: whole countries are held out.")
    m3.metric("Crop-average baseline (unseen)", f"R² {un['baseline_r2']:.2f}", f"MAE {un['baseline_mae']:,.0f} hg/ha",
              delta_color="off", delta_arrow="off", help="Always predict the crop's average yield.")
    st.info("The model is accurate for countries it has history on, and much less so for a new country, "
            "where it is only moderately better than simply using the crop's average yield.")

    st.subheader("What drives the predictions")
    imp = pd.DataFrame({"Feature": list(meta["importance"]), "Importance": list(meta["importance"].values())})
    st.altair_chart(alt.Chart(imp).mark_bar(color=GREEN).encode(
        x=alt.X("Importance:Q", axis=alt.Axis(format="%")), y=alt.Y("Feature:N", sort="-x"),
        tooltip=["Feature", alt.Tooltip("Importance:Q", format=".1%")]).properties(height=240), width="stretch")
    st.caption("Importance of the final model, with the one-hot country and crop columns grouped back together.")

    st.subheader("Models compared in the notebook")
    st.dataframe(NOTEBOOK_RESULTS.style.format({"R²": "{:.4f}", "MAE (hg/ha)": "{:,.0f}", "RMSE (hg/ha)": "{:,.0f}", "MAPE": "{:.3f}"}),
                 hide_index=True, width="stretch")
    st.caption(f"This app serves a Random Forest ({meta['n_trees']} trees, trained on all {meta['n_rows']:,} cleaned rows).")

    with st.expander("Limits to keep in mind"):
        st.markdown("""
- **Annual national figures.** Rainfall, temperature and pesticides are country-level values, not field-level data.
  Rainfall is a single fixed value per country. There is no soil, irrigation or fertiliser information.
- **Not causal.** Feature importance and what-if curves describe patterns the model learned, not cause and effect.
- **Training range.** Years outside 1990–2013 and values outside the training ranges are not meaningful.
- **No history, lower reliability.** A country or country-crop pair that is not in the data gets a much less reliable prediction.
""")
