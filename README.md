# Crop Yield Prediction

A beginner-level regression project that predicts crop yield (hectograms per hectare) across 101 countries and 10 crop types, using climate and agricultural input features from the FAO / Kaggle dataset. It includes the full analysis notebook and an interactive Streamlit app.

## Problem statement

Given **country, crop type, year, annual rainfall, pesticide usage and average temperature**, predict the continuous crop yield in hg/ha (`hg/ha_yield`). A model like this can support agricultural planning, food-security assessments and resource allocation decisions.

## Dataset

| | |
|---|---|
| Source | crop yield data (`yield_df.csv`) |
| Size | 28,242 rows, 25,932 after removing 2,310 exact duplicates |
| Coverage | 101 countries, 10 crops, 1990-2013 (2003 is missing) |
| Target | `hg/ha_yield` (range 50 to 501,412, strongly right-skewed) |
| Features | `Area`, `Item`, `Year`, `average_rain_fall_mm_per_year`, `pesticides_tonnes`, `avg_temp` |

## Approach

1. **Cleaning:** dropped the leftover index column and exact duplicate rows; no missing values.
2. **EDA:** yield by crop, country and year, correlations, distributions, pesticide and yield trends.
3. **Preprocessing:** a `ColumnTransformer` (median imputation and scaling for numeric columns, most-frequent imputation and one-hot encoding for `Area` and `Item`), fitted on the training split only.
4. **Models:** Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, KNN (k chosen by 3-fold CV on the training set) and Gradient Boosting, evaluated with MAE, RMSE, R² and MAPE.
5. **Validation:** 5-fold cross-validation, Random Forest hyperparameter tuning, residual analysis, and an unseen-country check.
6. **Export:** the notebook saves the trained model, cleaned data and metrics for the Streamlit app.

## Results

Hold-out test set (random 80/20 split):

| Model | R² | MAE (hg/ha) | RMSE (hg/ha) | MAPE |
|---|---|---|---|---|
| KNN (k=2) | 0.9865 | 4,098 | 9,910 | 0.081 |
| Random Forest | 0.9842 | 4,021 | 10,687 | 0.086 |
| Gradient Boosting | 0.9493 | 11,888 | 19,178 | 0.345 |
| Decision Tree | 0.9224 | 14,029 | 23,713 | 0.457 |
| Linear Regression | 0.7487 | 29,921 | 42,681 | 0.899 |
| Lasso | 0.7486 | 29,906 | 42,686 | 0.897 |
| Ridge | 0.7485 | 29,885 | 42,692 | 0.895 |

- **5-fold CV (R²):** Random Forest 0.9878 (±0.0022), KNN 0.9873 (±0.0016), Gradient Boosting 0.9534, Decision Tree 0.9266, linear models 0.7484.
- **Tuning:** a grid search on the Random Forest gave no real gain (R² 0.9844 vs 0.9842), so the default forest is the recommended final model.

### Key findings

- **Crop type is the dominant driver.** Root crops and plantains yield roughly 5 to 12 times more per hectare than cereals and oilseeds, and crop identity carries about 60% of the final model's feature importance (country about 17%, the four numeric features the rest).
- **Yield rose about 35% from 1990 to 2013**, while mean pesticide use rose about 1.85 times. Both move together over time, which does not prove a causal link.
- **Non-linear models beat linear ones by a wide margin** because yield depends on interactions between crop, country and climate.

### Honest evaluation

The scores above are for countries the model has already seen. When **whole countries are held out** (5-fold `GroupKFold` by country), the Random Forest drops from R² 0.98 to **0.64 (±0.04)**, against 0.55 for a simple crop-average baseline. For a country with no history the model is only moderately better than predicting the crop's average yield.

## Streamlit app

An interactive app with four tabs:

- **Prediction:** pick a country, crop, year and climate inputs and see the predicted yield, a comparison with the crop average and that country's history, and a rough tree-agreement range.
- **What-if:** sweep temperature, rainfall or pesticides, and compare all crops under the same conditions.
- **Data insights:** yield by crop, the yearly trend, and top countries for a crop.
- **Model:** the model comparison, seen vs unseen-country accuracy, feature importance and limitations.

## Project structure

```
crop-yield-prediction/
├── crop_yield.ipynb          # full analysis (EDA, models, CV, tuning, export)
├── yield_df.csv              # raw dataset
├── README.md
└── streamlit_app/
    ├── app.py                # the Streamlit app
    ├── requirements.txt
    ├── .streamlit/config.toml
    ├── crop_yield_model.pkl  # created by the notebook
    ├── yield_clean.csv       # created by the notebook
    └── model_meta.json       # created by the notebook
```

## How to run

**1. Notebook**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook crop_yield.ipynb
```
Use Restart and Run All. The last section, "Export for the Streamlit app", creates the three generated files in `streamlit_app/` and prints the exact `scikit-learn==...` line to put in `requirements.txt`. The saved model must be loaded with the same scikit-learn version that created it.

**2. App**
```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

## Limitations

- Rainfall, temperature and pesticide values are **annual national figures**, not field-level data. Rainfall is a single fixed value per country, so it acts more like a country label than a yearly measurement. There is no soil, irrigation or fertiliser information.
- Only 13,130 of the 25,932 rows are unique country-crop-year combinations. The rest repeat the same yield with a different temperature value.
- Random splits and shuffled cross-validation keep every country in training, which makes scores optimistic for new countries.
- Feature importance and the app's what-if curves describe patterns the model learned, not cause and effect.

## Future work

- Evaluate with a train-on-past / test-on-later-years split.
- Try a log-transformed target and lagged yield features.
- Add per-hectare pesticide use and other agronomic features.
- Deploy the app on Streamlit Community Cloud.

## Author

Ram Vilas Hasanapuram: [GitHub](https://github.com/ramvilas273) | [LinkedIn](https://linkedin.com/in/ram-vilas)
