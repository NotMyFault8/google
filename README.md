# NYC Taxi Fare Prediction

An end-to-end analytics and machine-learning project that explains the drivers of New York City taxi fares, develops practical regression models, and translates the results into pricing, product, and marketplace recommendations.

The repository combines a Python modeling notebook, a full written report, a rendered R analysis, and GitHub-ready static visualizations extracted from the report.

## Project at a Glance

| Item | Verified result |
|---|---:|
| Full Kaggle training file inspected in the R analysis | 55,423,856 rows, 8 columns |
| Phase 3 Python training sample | First 10,000,000 rows |
| Clean Phase 3 training set | 9,631,370 rows, 42 columns |
| Clean Phase 3 test set | 9,819 rows, 41 columns |
| Executed XGBoost validation result | RMSE **2.9383**, MAE **1.3864** |
| Executed CatBoost validation result | RMSE **2.9567**, MAE **1.4206** |
| R linear-model baseline | R² **0.7544**, adjusted R² **0.7543**, residual standard error **4.708** |
| Dominant interpretable driver | Haversine distance, estimated at **+$2.3166 per km** in the R baseline |

> **Metric note:** the R model reports in-sample explanatory statistics on an 800,000-row sample. The boosted-tree metrics are holdout prediction errors from the Python notebook. They are useful for different purposes and should not be treated as a strictly controlled head-to-head benchmark.

## Contents

- [Business Objective](#business-objective)
- [Data](#data)
- [Data Cleaning](#data-cleaning)
- [Feature Engineering](#feature-engineering)
- [Exploratory Data Analysis](#exploratory-data-analysis)
- [Visualization Gallery](#visualization-gallery)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Key Findings](#key-findings)
- [Business Recommendations](#business-recommendations)
- [Limitations](#limitations)
- [Repository Structure](#repository-structure)
- [Setup](#setup)
- [Reproducibility](#reproducibility)

## Business Objective

The project frames taxi-fare prediction as a mobility-marketplace problem rather than only a leaderboard exercise. A useful fare estimator should be accurate enough for customer expectations, interpretable enough to explain major price drivers, and operationally useful for identifying spatial and temporal patterns.

The analysis addresses three core questions:

1. **Distance:** How strongly does trip distance explain `fare_amount`, and where does the relationship become nonlinear because of base fares, flat-fare routes, or other pricing rules?
2. **Time:** How do pickup hour, day of week, weekend status, rush hour, and night travel relate to fare levels and trip composition?
3. **Location:** Do airport proximity, Manhattan location, pickup/dropoff zones, and spatial demand concentration improve fare estimation or reveal useful operating segments?

The technical success measures are:

- **RMSE** for sensitivity to larger fare-prediction errors.
- **MAE** for the typical absolute error in fare units.
- **R² and coefficients** for the interpretable statistical baseline.
- **Diagnostic plots** for nonlinearity, heteroskedasticity, heavy tails, and influential observations.
- **Spatial and temporal visualizations** for stakeholder communication and business interpretation.

## Data

### Source and schema

The project uses the Kaggle **New York City Taxi Fare Prediction** competition data.

The raw training schema contains:

| Column | Role |
|---|---|
| `key` | Unique trip identifier/time-like key in the source data |
| `fare_amount` | Continuous prediction target; training data only |
| `pickup_datetime` | Pickup timestamp |
| `pickup_longitude`, `pickup_latitude` | Pickup coordinates |
| `dropoff_longitude`, `dropoff_latitude` | Dropoff coordinates |
| `passenger_count` | Number of passengers |

The full R analysis loaded **55,423,856 training rows** and **9,914 test rows**. The Phase 3 Python notebook deliberately limits the training read to the first **10,000,000 rows** to keep feature engineering and boosted-tree training tractable.

### Analysis scopes

Several artifacts were produced at different stages, so their row counts should not be mixed:

| Artifact | Raw scope | Clean scope | Purpose |
|---|---:|---:|---|
| R analysis (`lm.html`) | 55,423,856 rows | 54,015,314 rows under its own cleaning rules | Full-data audit, EDA, statistical baseline, diagnostics |
| Written report | 2,000,000-row Python sample | 1,926,679 rows | Narrative report and visualization package |
| Phase 3 notebook | First 10,000,000 training rows | 9,631,370 rows | Advanced feature engineering, XGBoost, CatBoost |

The artifacts use related but not identical geographic and outlier thresholds. Comparisons should therefore be made within an artifact unless a common pipeline is rerun.

### Raw data-quality signals

The full-data R audit confirms why cleaning is necessary:

- `fare_amount` ranges from **-300.00** to **93,963.36** before filtering.
- `passenger_count` reaches **208**, far outside normal taxi capacity.
- Raw coordinates include physically impossible latitude/longitude values in the thousands.
- `dropoff_longitude` and `dropoff_latitude` each contain **376 missing values**.
- The fare distribution is strongly right-skewed, even after basic cleaning.

The raw CSV files are not included in this repository and should not be committed because of their size and competition-data terms.

## Data Cleaning

The Phase 3 notebook implements a reusable `cleaning_pipeline()` that applies each rule sequentially and records rows removed, step-level removal percentage, and rows remaining.

### Phase 3 cleaning rules

1. Drop rows missing essential time, coordinate, passenger, or target fields.
2. Keep `passenger_count` between 1 and 6.
3. Keep pickup and dropoff coordinates inside the configured NYC envelope: latitude 40–42 and longitude -75 to -72.
4. Remove non-positive fares from training data.
5. Trim fares above the training sample's 99.9th percentile.
6. Parse `pickup_datetime` as UTC and remove parse failures.
7. Remove zero-distance trips under both Haversine and Manhattan-style distance calculations.
8. Trim Haversine distance above its 99.9th percentile.

### Cleaning audit: 10-million-row Phase 3 sample

Percentages below are calculated against the rows entering each step, not the original 10 million.

| Step | Rule | Rows removed | Removed at step | Rows remaining |
|---:|---|---:|---:|---:|
| 0 | Essential fields are non-null | 69 | 0.000690% | 9,999,931 |
| 1 | Passenger count is 1–6 | 35,280 | 0.352802% | 9,964,651 |
| 2 | Pickup/dropoff inside NYC bounds | 210,081 | 2.108262% | 9,754,570 |
| 3 | Fare is positive | 599 | 0.006141% | 9,753,971 |
| 4 | Fare ≤ 99.9th percentile ($78.75) | 9,701 | 0.099457% | 9,744,270 |
| 5 | Datetime parsed successfully | 0 | 0.000000% | 9,744,270 |
| 6 | Haversine and Manhattan distances > 0 | 103,258 | 1.059679% | 9,641,012 |
| 7 | Haversine distance ≤ 99.9th percentile (23.27 km) | 9,642 | 0.100010% | **9,631,370** |

Overall, the pipeline removes **368,630 rows (3.6863%)** from the 10-million-row training sample. The largest single reduction comes from the geographic filter, followed by removal of degenerate zero-distance trips.

| Cleaning-rule removal | Cumulative rows retained |
|---|---|
| ![Percentage removed by each cleaning rule](assets/figures/cleaning-rule-removal.png) | ![Cumulative percentage of rows retained after cleaning](assets/figures/cleaning-cumulative-remaining.png) |

## Feature Engineering

The final notebook goes beyond the report's original Haversine-and-time feature set. It creates temporal, geometric, point-of-interest, spatial-grid, and interaction variables.

### Generated features

| Category | Features | Business or modeling purpose |
|---|---|---|
| Time | `pickup_hour`, `pickup_dow`, `pickup_month`, `pickup_year`, `year_index`, `year_month_index` | Captures hourly, weekly, seasonal, and long-term fare patterns |
| Time flags | `is_weekend`, `is_rush_hour`, `is_night` | Represents operationally meaningful travel periods |
| Cyclical time | `hour_sin`, `hour_cos` | Preserves the circular relationship between 23:00 and 00:00 |
| Distance | `haversine_km`, `manhattan_km`, `distance_ratio_manhattan_haversine` | Measures direct and road-like trip length and route geometry |
| Direction | `trip_bearing`, `bearing_sin`, `bearing_cos` | Captures travel direction without a discontinuity at ±180° |
| POI proximity | Pickup/dropoff distance to JFK, LGA, and Manhattan center | Represents airport and central-business-district effects |
| Location flags | `is_jfk_trip`, `is_lga_trip`, Manhattan-core pickup/dropoff flags | Creates interpretable trip segments |
| Trip regimes | `is_short_trip`, `is_long_trip` | Allows separate behavior for base-fare and long-distance trips |
| Interactions | Distance × JFK, LGA, rush hour, night, and long-trip indicators | Lets distance have different effects across trip scenarios |
| Spatial grid | Pickup/dropoff grid coordinates and cell IDs | Creates coarse location encodings for nonlinear models |

The airport flags use a **2 km radius** around JFK and LGA. The Manhattan-core flag uses a bounding box of longitude **-74.02 to -73.93** and latitude **40.70 to 40.82**.

### Features used by the boosted models

The notebook's `FULL` list contains **28 predictors**:

- Passenger, hour, day-of-week, month, year, weekend, night, and Haversine distance.
- Four pickup/dropoff grid-coordinate features.
- Manhattan distance and the Manhattan-to-Haversine distance ratio.
- Six distances to JFK, LGA, and Manhattan center.
- JFK and LGA trip flags.
- Distance interactions for JFK, LGA, rush hour, and night.
- Long-trip flag and long-trip distance interaction.

The notebook also computes pickup/dropoff cell-level fare means and counts using training-only statistics, then joins them to validation data. This is the correct leakage-aware direction; however, those four statistics are not included in the recorded `FULL` model feature list. Bearing, cyclical-hour, short-trip, and Manhattan-core variables are also generated but not used in the captured XGBoost/CatBoost run.

## Exploratory Data Analysis

### 1. Fare and distance

Fare rises strongly with Haversine distance for most trips. The relationship is approximately linear across the dense center of the data, but the scatter widens at longer distances and contains horizontal fare bands. Those bands are consistent with discrete pricing rules, flat-fare routes, or unobserved toll/surcharge components.

| Fare vs. distance | Average fare by distance bin |
|---|---|
| ![Fare amount versus Haversine distance](assets/figures/fare-vs-haversine-distance.png) | ![Average fare by Haversine-distance bin](assets/figures/average-fare-by-distance-bin.png) |

### 2. Base-fare effect and short-trip nonlinearity

Fare per kilometer is highest for very short trips and declines as distance increases. This pattern is expected when a fixed initial charge is spread over a short distance, and it motivates nonlinear distance terms or separate short-trip logic.

![Fare per kilometer versus trip distance](assets/figures/fare-per-km-vs-distance.png)

### 3. Time patterns

Average fare changes by pickup hour. The report's chart shows its highest average-fare bars in the early-morning period, especially around 04:00–06:00, while much of the daytime remains comparatively stable. This is descriptive rather than causal: the mix of airport, long-distance, and low-volume trips can change by hour.

| Average fare by hour | Hour × weekday/weekend heatmap |
|---|---|
| ![Average fare by pickup hour](assets/figures/average-fare-by-pickup-hour.png) | ![Average fare by pickup hour and weekend status](assets/figures/fare-hour-weekend-heatmap.png) |

### 4. Weekday and weekend trip mix

Weekday and weekend fare distributions have similar medians and long upper tails. Distance distributions are also strongly right-skewed in both groups. Weekend status alone is therefore unlikely to replace distance and location features, although it may still contribute through interactions.

| Fare distribution | Distance distribution |
|---|---|
| ![Weekday and weekend fare boxplots](assets/figures/fare-weekday-weekend-boxplot.png) | ![Weekday and weekend distance histograms](assets/figures/distance-weekday-weekend-histogram.png) |

### 5. Spatial concentration

Pickup points are highly concentrated in Manhattan, with smaller clusters around airports and major corridors. The 3D and multidimensional maps show that trip volume and average fare are not spatially uniform, supporting location features and zone-specific error analysis.

![Sampled NYC taxi pickup coordinates](assets/figures/pickup-locations-sample.png)

## Visualization Gallery

All images below are actual figures extracted from the uploaded final report and saved under relative repository paths for GitHub rendering.

### Spatial views

| 3D count and average fare | Multidimensional count and fare map |
|---|---|
| ![3D spatial columns encoding trip count and average fare](assets/figures/spatial-3d-count-average-fare.png) | ![Spatial map combining trip count and average fare](assets/figures/spatial-multidimensional-map.png) |

| Dropoff-density 3D hex map | Pickup-density 3D hex map |
|---|---|
| ![Three-dimensional dropoff density hexagons](assets/figures/dropoff-density-3d-hex.png) | ![Three-dimensional pickup density hexagons](assets/figures/pickup-density-3d-hex.png) |

### R-based distribution and diagnostic views

<details>
<summary>Open the R EDA and regression-diagnostic gallery</summary>

| Fare distribution | Fare vs. Haversine distance |
|---|---|
| ![R histogram of fare amount](assets/figures/r-fare-distribution.png) | ![R scatterplot of fare and Haversine distance](assets/figures/r-fare-vs-distance.png) |

| Fare by pickup hour | Box-Cox profile |
|---|---|
| ![R boxplot of fare by pickup hour](assets/figures/r-fare-by-pickup-hour.png) | ![Box-Cox log-likelihood profile](assets/figures/r-boxcox-profile.png) |

| Original residuals vs. fitted | Original normal Q-Q plot |
|---|---|
| ![Residuals versus fitted values for the original-scale model](assets/figures/r-residuals-vs-fitted-original.png) | ![Normal Q-Q plot for original-scale residuals](assets/figures/r-qq-original.png) |

| Transformed residuals vs. fitted | Transformed normal Q-Q plot |
|---|---|
| ![Residuals versus fitted values after transforming fare](assets/figures/r-residuals-vs-fitted-transformed.png) | ![Normal Q-Q plot after transforming fare](assets/figures/r-qq-transformed.png) |

</details>

### HTML visualizations

- [`lm.html`](sources/lm.html) is the uploaded standalone R analysis with code, model output, plots, ANOVA, and a supplementary high-fare classification analysis.
- The written report names four interactive spatial deliverables: `nyc_pickup_3d_hex.html`, `nyc_dropoff_3d_hex.html`, `nyc_3d_count_color_fare.html`, and `nyc_multidim_map.html`.
- Those four standalone map files were not included in the uploaded project package. If added later, recommended relative paths are `visualizations/nyc_pickup_3d_hex.html`, `visualizations/nyc_dropoff_3d_hex.html`, `visualizations/nyc_3d_count_color_fare.html`, and `visualizations/nyc_multidim_map.html`.

GitHub does not execute arbitrary HTML from the repository file view. Download `lm.html` and open it locally for the intended rendered experience.

## Modeling

### Interpretable R baseline

The R analysis samples **800,000 cleaned rows** with `set.seed(1)` and fits:

```text
fare_amount ~ haversine_km
            + passenger_count
            + pickup_hour
            + pickup_dow
            + pickup_month
            + is_night
            + is_weekend
            + rush_hour
```

Recorded performance and selected coefficients:

| Statistic | Value |
|---|---:|
| Residual standard error | 4.708 |
| Multiple R² | 0.7544 |
| Adjusted R² | 0.7543 |
| Haversine distance coefficient | +2.316600 per km |
| Haversine coefficient 95% CI | [2.313698, 2.319501] |
| Passenger-count coefficient | +0.042914 |
| Pickup-hour coefficient | +0.015570 |
| Night indicator coefficient | -1.179980 |
| Rush-hour indicator coefficient | -0.321681 |

These coefficients are conditional associations, not causal effects. `is_weekend` is undefined in the reported model because it is redundant with the included day-of-week factor coding. The overlap among hour, rush-hour, day-of-week, and weekend variables also makes coefficient-level interpretation sensitive to model specification.

The residual diagnostics show:

- A fan/triangular residual pattern, indicating heteroskedasticity.
- Strong Q-Q tail deviations, indicating heavy-tailed and non-normal residuals.
- A Box-Cox optimum close to zero, motivating a log-like target transformation.
- More stable residual spread and improved central Q-Q alignment after transformation.

The log-fare model reports R² **0.6673** on the transformed target. That value should not be compared directly with the original-scale R² or validation RMSE.

### XGBoost

The Phase 3 notebook uses an 80/20 split with `random_state=42`. Spatial cell statistics are derived from the training partition before being joined to validation data.

Recorded configuration:

| Hyperparameter | Value |
|---|---:|
| Estimators | 400 |
| Learning rate | 0.05 |
| Maximum depth | 16 |
| Row subsample | 0.8 |
| Column subsample | 0.8 |
| L2 regularization (`reg_lambda`) | 1.0 |
| Tree method | `hist` |
| Objective | `reg:squarederror` |

The executed notebook output is:

- **RMSE: 2.9383**
- **MAE: 1.3864**

A later markdown cell records RMSE **2.9252** and MAE **1.3877**, but those values are not attached to a captured execution output. For reproducibility, this README treats **2.9383 / 1.3864** as the authoritative recorded XGBoost result until the notebook is rerun top to bottom.

### CatBoost

The CatBoost run uses:

- 1,000 iterations.
- Learning rate 0.05.
- Tree depth 16.
- RMSE loss.

Its executed output is:

- **RMSE: 2.9567**
- **MAE: 1.4206**

The CatBoost function creates a second 80/20 split from `train_df` rather than using the same `val_df` as XGBoost. The recorded metrics therefore suggest that XGBoost performs slightly better, but the experiment should be rerun on one shared holdout before making a final model-selection decision.

### Supplementary high-fare model

The R artifact also defines a high-fare trip as `fare_amount > 30` and fits a logistic model. Its Haversine-distance odds ratio is **1.7801 per additional kilometer**. This supports the distance-first business conclusion, but the model is supplementary and is not the primary fare estimator.

## Evaluation

### Recorded model results

| Model | Evaluation data | RMSE | MAE | Other metric | Interpretation |
|---|---|---:|---:|---:|---|
| Linear regression (R) | 800,000-row modeling sample; in-sample fit | — | — | R² 0.7544; RSE 4.708 | Interpretable baseline and assumption diagnostics |
| XGBoost | Captured 20% Phase 3 holdout | **2.9383** | **1.3864** | — | Best executed predictive result in the notebook |
| XGBoost, markdown note | Unspecified rerun/note | 2.9252 | 1.3877 | — | Not tied to an executed output; requires confirmation |
| CatBoost | Secondary 20% split inside `train_df` | 2.9567 | 1.4206 | — | Competitive, but not evaluated on the identical holdout |

### How to read the metrics

- **RMSE** penalizes large misses more heavily and is appropriate when severe fare underestimation or overestimation matters disproportionately.
- **MAE** is easier to interpret as the average absolute fare error and is less dominated by rare large residuals.
- **R²** measures variance explained by the statistical baseline; it does not directly describe out-of-sample dollar error.
- **Residual standard error** is not a replacement for holdout RMSE, especially when the model and sample differ.

### Evaluation gaps

The uploaded artifacts do not contain:

- A shared cross-validation table for all models.
- A time-based holdout that tests future-period generalization.
- Persisted XGBoost or CatBoost feature-importance output.
- A top-five feature-importance table, despite that being requested in the original project brief.
- A Kaggle submission file or leaderboard score from the Phase 3 notebook.
- Segment-level error metrics for airports, short trips, long trips, hours, or spatial zones.

For that reason, no unsupported tree-model feature ranking or leaderboard claim is made here. The conclusion that distance is the dominant driver is supported by the EDA, the linear coefficient, and the supplementary high-fare analysis.

## Key Findings

1. **Distance is the dominant fare driver.** The R baseline estimates approximately **$2.32 per additional Haversine kilometer**, holding its other variables fixed, and the high-fare model reports a distance odds ratio of **1.78 per km**.
2. **Short trips operate in a different pricing regime.** Fare per kilometer is highest near zero distance, consistent with a fixed base charge and supporting nonlinear or segmented distance modeling.
3. **A linear model captures the central relationship but misses important structure.** Its R² is strong for an interpretable baseline, yet residual variance increases with predicted fare and the tails remain non-normal.
4. **A log-like transformation improves statistical behavior.** The Box-Cox profile peaks near zero, and the transformed diagnostics show more stable spread and better central normality.
5. **Time contributes signal, but less than distance and trip mix.** Hour, night, rush hour, day of week, and month are statistically associated with fare, but their coefficients are sensitive to overlapping encodings and should not be read causally.
6. **Passenger count has limited standalone pricing impact.** Its baseline coefficient is positive but small compared with distance.
7. **Location matters.** Manhattan dominates trip density, while airport and corridor patterns support dedicated spatial features and location-segmented monitoring.
8. **Boosted trees materially reduce predictive error relative to what the linear diagnostics suggest is achievable with a simple functional form.** The executed XGBoost holdout result is RMSE **2.9383** and MAE **1.3864**.

## Business Recommendations

| Recommendation | Evidence | Action | Expected impact | Risk or guardrail |
|---|---|---|---|---|
| Build a nonlinear, distance-first fare estimator | Strong fare-distance trend; high short-trip fare/km; heteroskedastic linear residuals | Use Haversine/Manhattan distance with short-trip and long-trip regimes, nonlinear splines, or boosted trees | Lower systematic error, especially near the base-fare region and at long distances | Straight-line proxies still miss the driven route, traffic, tolls, and detours |
| Calibrate airport and spatial segments separately | JFK/LGA proximity features and 3D maps show distinct corridors, density, and fare patterns | Monitor and, if justified, calibrate airport, Manhattan-core, and outer-zone estimates separately | Better accuracy for high-value routes and clearer operational visibility | Geofences can misclassify edge cases; flat-fare and toll rules must be represented explicitly |
| Use time signals for operations and transparent estimates, not isolated price decisions | Hour, night, rush-hour, and day-of-week terms add signal but are correlated and smaller than distance | Use these features to improve estimates, reposition supply, and explain contextual uncertainty | Better pickup availability, reduced surprise, and improved user trust | Observed associations are not causal; policy and fairness review should precede pricing changes |
| Monitor errors by scenario | Residual spread grows with fitted fare and tails remain heavy | Track RMSE, MAE, bias, and prediction intervals by distance band, hour, airport flag, and spatial zone | Detects systematic underestimation before it harms users or marketplace economics | Aggregate metrics can hide failures for rare or high-fare trips |

### Suggested operational dashboard

A production-facing dashboard should include:

- KPI cards for average fare, average distance, trip count, RMSE, MAE, and model version.
- Filters for date range, pickup hour, weekday/weekend, passenger count, airport flag, and spatial zone.
- Fare distribution and fare-vs-distance views.
- Average fare by hour/day and trip volume by hour/day.
- Pickup/dropoff density maps with average-fare overlays.
- Segment-level residual charts and an explanation panel listing the main estimate drivers.

## Limitations

- **Route approximation:** Haversine is straight-line distance and Manhattan distance is still only a coordinate-based proxy. Neither knows the actual road route.
- **Missing fare components:** The data does not explicitly provide traffic, weather, tolls, tips, route duration, payment type, or detailed fare-rule context.
- **Outlier policy:** Quantile trimming improves stability but removes rare trips that may be operationally important.
- **Geographic definitions:** The Python pipeline uses a broad NYC envelope, while grid features clip to a tighter rectangle. R and Python artifacts also use different bounds.
- **Multiple analysis scopes:** Full-data R results, the 2M report sample, and the 10M Phase 3 sample are not identical experiments.
- **Validation design:** A random split does not test temporal drift from 2009 through 2015. A chronological holdout would better represent future deployment.
- **Model comparability:** CatBoost and XGBoost are not evaluated on exactly the same captured validation frame.
- **Cached notebook state:** Some markdown values and displayed outputs do not fully match the latest source cells. Clear outputs and rerun top to bottom before publishing a definitive benchmark.
- **Interpretability gap:** No executed tree feature-importance or SHAP output is included, so the project cannot yet claim a verified top-five model feature ranking.
- **Interactive-file gap:** The report references four interactive map HTML files that were not included in the uploaded package; only their static screenshots are available here.
- **Reproducibility metadata:** No pinned `requirements.txt`, environment lockfile, serialized model, prediction export, or raw-data checksum is included.
- **Correlation versus causation:** Time and location effects describe associations in historical trips and should not automatically be interpreted as causal pricing effects.

## Repository Structure

```text
.
├── README.md
├── assets/
│   └── figures/
│       ├── cleaning-rule-removal.png
│       ├── fare-vs-haversine-distance.png
│       ├── average-fare-by-pickup-hour.png
│       ├── spatial-3d-count-average-fare.png
│       ├── r-residuals-vs-fitted-original.png
│       └── ... 17 additional report figures
└── sources/
    ├── phase 3 cleaned.ipynb
    ├── NYC_Taxi_Fare_Prediction_Report_v4.docx
    ├── Google Data Analytics New York City Taxi Fare Prediction.pdf
    ├── lm.html
    ├── [Project-Google-Data-Analytics-–-NYC-Taxi-Fare-Prediction-(Kaggle)].txt
    └── Answer-type.txt
```

Key artifacts:

- [Phase 3 Python notebook](sources/phase%203%20cleaned.ipynb)
- [Final project report](sources/NYC_Taxi_Fare_Prediction_Report_v4.docx)
- [Original project brief](sources/Google%20Data%20Analytics%20New%20York%20City%20Taxi%20Fare%20Prediction.pdf)
- [Rendered R analysis](sources/lm.html)

The Kaggle `train.csv`, `test.csv`, and `sample_submission.csv` files are intentionally absent.

## Setup

### Python environment

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install jupyter pandas numpy scikit-learn matplotlib seaborn xgboost catboost
```

On Windows PowerShell, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### Data placement

1. Download `train.csv` and `test.csv` from the Kaggle competition.
2. Keep the files outside version control.
3. The captured notebook currently uses `TRAIN_PATH = "train.csv"` and `TEST_PATH = "test.csv"`.
4. Either place the CSV files beside a working copy of the notebook or update those constants to a local data directory such as `data/raw/`.

Recommended local-only layout:

```text
data/
└── raw/
    ├── train.csv
    └── test.csv
```

If using that layout from a notebook stored in `notebooks/`, update the paths to `../data/raw/train.csv` and `../data/raw/test.csv`.

### Start Jupyter

```bash
jupyter lab
```

The 10-million-row workflow is memory intensive. To validate the pipeline on a laptop, first reduce `nrows` to 1–2 million, then scale up after the full notebook runs successfully.

## Reproducibility

For a defensible rerun:

1. Create a fresh environment and record exact package versions with `python -m pip freeze > requirements-lock.txt`.
2. Download the competition files and record file sizes or checksums locally.
3. Copy the notebook to a writable analysis directory if needed, update data paths, clear all cached outputs, and restart the kernel.
4. Run every cell from top to bottom without skipping feature-generation or cleaning cells.
5. Confirm the expected Phase 3 clean shapes: training `(9631370, 42)` and test `(9819, 41)` when using the same 10-million-row input slice and package behavior.
6. Preserve `random_state=42` for the Python split and XGBoost model. Preserve `set.seed(1)` for the R modeling sample if reconstructing the R workflow.
7. Refactor CatBoost to use the same `train_df`/`val_df` split as XGBoost before comparing models.
8. Save a model-comparison table, predictions, feature importance or SHAP values, and segment-level error metrics.
9. Add a chronological holdout and cross-validation before treating the recorded model as deployment-ready.

Because the repository stores the rendered `lm.html` rather than an `.Rmd` or `.qmd` source file, the R analysis is best treated as an auditable archival artifact unless its source is reconstructed.

## Next Modeling Steps

- Compare the current boosted models on one shared validation set.
- Add a Ridge baseline and a spline/GAM distance model for transparent nonlinear benchmarking.
- Evaluate log-target training with correct inverse transformation and original-scale RMSE/MAE.
- Add road-network distance, estimated duration, toll indicators, traffic, and weather.
- Use SHAP or permutation importance and publish a verified top-five feature table.
- Add spatial and temporal error slices, prediction intervals, and drift monitoring.
- Export a reproducible prediction file and, if appropriate, a Kaggle submission.

