# NYC Property Valuation AI: End-to-End Machine Learning Pipeline & Deployment

<img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue"> <img src="https://img.shields.io/badge/Python-3.10%2B-brightgreen"> <img src="https://img.shields.io/badge/Library-Scikit--Learn-orange"> <img src="https://img.shields.io/badge/License-MIT-yellow">

> **Live Interactive Demo & API:** [NYC Property Valuation App on Hugging Face Spaces](https://huggingface.co/spaces/abade1990/nyc-property-ai)  
> **Source Dataset:** [NYC Property Sales Dataset on Kaggle](https://www.kaggle.com/datasets/new-york-city/nyc-property-sales/data)

---

## 1. Executive Summary

This project implements an end-to-end data engineering and machine learning pipeline to value real estate transactions across the five boroughs of New York City (Manhattan, Bronx, Brooklyn, Queens, and Staten Island). 

Beginning with raw administrative records fraught with whitespace masking, non-market gift deeds, and data entry anomalies, the pipeline cleans, audits, and engineers domain-specific features. A baseline Linear Regression model is established and subsequently outperformed by an ensemble Random Forest model, which accounts for non-linear architectural trends and geographic price interactions. The trained model is deployed alongside an ensemble-based uncertainty estimation engine (80% prediction interval and confidence score) via an interactive Gradio UI and REST API on Hugging Face Spaces.

### Performance Highlights
* **Dataset Scale:** Cleaned from 84,548 raw transactions down to **56,476 valid market sales**.
* **Baseline Model (Linear Regression):** $R^2 = \mathbf{0.4684}$, $\text{MAE} = \mathbf{\$561,735}$.
* **Challenger Model (Random Forest Regressor):** $R^2 = \mathbf{0.6003}$, $\text{MAE} = \mathbf{\$469,615}$ (an error reduction of **\$92,120 per transaction** over the baseline).
* **Primary Valuation Driver:** Audited property age (`AGE`) emerged as the **#1 most important feature** across the ensemble.
* **Engineering Iteration (v1.1):** Replaced flat category imputation with **Hierarchical Multi-Level Imputation** (`[NEIGHBORHOOD, BUILDING CLASS]`), narrowing test prediction intervals by **$43,115** and increasing model consensus to **87.2%** without target leakage.
* **Production Serving:** Zero-downtime serving with real-time confidence intervals on [Hugging Face Spaces](https://huggingface.co/spaces/abade1990/nyc-property-ai).

---

## 2. Detailed Pipeline Stages

```
Raw CSV (84k rows) 
  ──> Ingestion & Masked Null Detection 
  ──> Type Casting & Outlier Trimming (56k rows) 
  ──> Exploratory Data Analysis & Diagnostics 
  ──> Hierarchical Domain Feature Engineering 
  ──> Model Training & Benchmarking (OLS vs. Random Forest) 
  ──> Uncertainty & Confidence Engine (87.2% Agreement) 
  ──> Cloud Deployment (Gradio + Hugging Face)
```

### Stage 1: Data Ingestion & Anomaly Auditing
* **Masked Missing Values:** Raw records indicated `84548 non-null` entries across all columns. Programmatic inspection revealed that missing values were encoded as whitespace (`' '`) or hyphens (`" -  "`), bypassing default null checks.
* **Zero-Variance Column Removal:** Column `EASE-MENT` contained 100% whitespace across all 84,548 records ($\sigma^2 = 0$). It was dropped alongside the redundant index column `Unnamed: 0`.

### Stage 2: Data Cleaning & Hierarchical Imputation
* **Forced Numeric Casting:** Parsed `SALE PRICE`, `LAND SQUARE FEET`, and `GROSS SQUARE FEET` using `pd.to_numeric(..., errors='coerce')`, exposing 14,561 missing target values and over 26,000 missing physical area values.
* **Target Integrity:** Discarded rows lacking a sale price to prevent training on synthetic or imputed target labels.
* **Zero-Area Conversion:** Physical areas recorded as `0.0` (common for condominiums where land is not held individually) were treated as `NaN`.
* **Hierarchical Multi-Level Imputation (v1.1 Breakthrough):** Imputing space solely by building category ignores urban density differences between Manhattan and Staten Island. Implemented a 3-tier cascading fallback:
  1. *Level 1 (Highest Precision):* Grouped median by `['NEIGHBORHOOD', 'BUILDING CLASS CATEGORY']`.
  2. *Level 2 (Macro Category Fallback):* Grouped median by `BUILDING CLASS CATEGORY`.
  3. *Level 3 (Safety Fallback):* Global column median.
  *Result:* Preserved micro-location physical integrity while maintaining zero target leakage.
* **De-duplication & Market Filtering:** Removed 380 duplicate records. Excluded non-market transfers (deeds transferred for family gifts, foreclosures, or nominal fees under \$100,000) and capped luxury outliers at the 99th percentile (\$14,300,000), leaving a clean sample of 56,476 transactions.

### Stage 3: Exploratory Data Analysis (EDA)
* **Comprehensive 47-Category Heatmap:** Constructed a 2D cross-tabulation of median prices across all 5 boroughs and 47 building types. Identified that single-family homes (`01 ONE FAMILY DWELLINGS`) are present across all boroughs, serving as an ideal control group.
* **The Non-Linear Era U-Curve:** Visualizing price against construction era demonstrated a non-linear relationship: prices drop from Modern (>\$878k) to Post-War mid-century (\$520k), then rise again for Pre-War historic properties (\$700k) due to heritage architectural premiums.
* **Heteroscedasticity Diagnosis:** Scatter plots of property area against raw sale price exhibited a pronounced fan-shaped dispersion. Applying a natural logarithmic transform, $y = \ln(\text{Price} + 1)$, stabilized error variance (homoscedasticity) and aligned distributions for regression.

### Stage 4: Feature Engineering & Auditing
* **Property Age Audit (`AGE`):** Calculated property age at transaction time (`SALE_YEAR - YEAR BUILT`). Inspection detected a data entry typo (`YEAR BUILT = 1111`, indicating an age of 906 years for a 7th Avenue commercial garage). Correcting the typo to `1911` stabilized the age distribution to a maximum of 217 years and a median of 68 years.
* **Architectural Era Binning (`BUILDING_ERA`):** Discretized building year into four distinct categories: `Pre-War (Historic)`, `Post-War (Mid-Century)`, `Late 20th Century`, and `Modern`.
* **Unit Density Metric (`UNIT_SIZE`):** Computed average unit area as $\frac{\text{GROSS SQUARE FEET}}{\max(1, \text{TOTAL UNITS})}$ with defensive zero-division safeguards.
* **Nominal Feature Encoding:** Applied One-Hot Encoding (`pd.get_dummies(..., drop_first=True)`) on nominal features (`BOROUGH`, `BUILDING CLASS CATEGORY`, `BUILDING_ERA`, `TAX CLASS AT TIME OF SALE`), yielding a 57-dimensional feature space while preventing the dummy variable trap.

### Stage 5: Model Training, Benchmarking & Interpretation
* **Validation Strategy:** Partitioned data into 80% training ($n = 45,180$) and 20% test ($n = 11,296$) using a fixed random seed (`random_state=42`).
* **Linear Baseline (Ordinary Least Squares):**
  * $R^2 = 0.4684$
  * $\text{RMSE}_{\text{log}} = 0.6291$
  * $\text{MAE} = \$561,735$
* **Ensemble Model (Random Forest Regressor):**
  * Configured with 100 estimators and a maximum depth of 15 to constrain overfitting.
  * $R^2 = 0.6003$
  * $\text{RMSE}_{\text{log}} = 0.5458$
  * $\text{MAE} = \$469,615$
* **Feature Importance Analysis:** Extracted Gini-based feature importances across all trees. The audited `AGE` feature ranked as the **#1 most influential predictor** (relative weight ~13.6%), followed by `TOTAL UNITS`, `GROSS SQUARE FEET`, and high-rise condominiums (`13 CONDOS - ELEVATOR APARTMENTS`).

### Stage 6: Uncertainty Estimation & Production Deployment
* **Ensemble Uncertainty Engine:** Rather than outputting a single point estimate, individual predictions are extracted from all 100 constituent decision trees:
  $$\hat{y}_i = \exp(\text{tree}_i(x)) - 1, \quad i \in \{1, \dots, 100\}$$
* **Evaluation Card Deliverables:**
  1. **Point Estimate:** Mean of the 100 predictions.
  2. **80% Prediction Interval:** 10th percentile to 90th percentile of the ensemble outputs.
  3. **Confidence Score (%):** Derived from the coefficient of variation ($CV = \frac{\sigma}{\mu}$):
     $$\text{Confidence Score} = \max\left(0, 1 - \frac{\sigma_{\text{trees}}}{\mu_{\text{trees}}}\right) \times 100\%$$

#### Evaluation Impact: Flat Imputation vs. Hierarchical Imputation
| Metric | Baseline Imputation (Category-only) | Hierarchical Imputation (Neighborhood + Category) | Impact |
| :--- | :---: | :---: | :---: |
| **Actual Test Price** | **$352,000** | **$352,000** | Ground Truth |
| **Model Point Estimate** | $341,029 | $377,498 | Consistent accuracy |
| **80% Prediction Interval** | $284,706 to $423,520 | **$327,275 to $422,974** | **Narrowed by $43,115** |
| **Interval Width (Uncertainty)** | $138,814 | **$95,699** | **31.1% tighter margin** |
| **Ensemble Confidence Score** | 84.9% | **87.2%** | **+2.3% agreement boost** |

* **Cloud Serving:** Packaged the serialized model (`nyc_rf_model.pkl`) and feature schema (`model_features.pkl`) into a Gradio interface running on Hugging Face Spaces.

---

## 3. Project File Structure

```text
├── nyc_rf_model.pkl          # Serialized Random Forest model artifact (100 trees)
├── model_features.pkl        # Ordered schema of the 57 encoded features
├── requirements.txt          # Server dependency specifications
├── app.py                    # Production Gradio application and inference logic
├── exploratory_analysis.ipynb # End-to-end development, EDA, and training notebook
└── README.md                 # Project documentation and architecture guide
```

---

## 4. Local Installation & Quickstart

To run the inference application locally:

```bash
# 1. Clone the repository
git clone [https://github.com/alsharef1550mm-commits/nyc-property-valuation-ml.git](https://github.com/alsharef1550mm-commits/nyc-property-valuation-ml.git)
cd nyc-property-valuation-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the Gradio UI and API
python app.py
```
Open your browser at `http://localhost:7860` to access the local valuation interface.

---

## 5. API Reference

The deployed application on Hugging Face Spaces exposes an automated API endpoint. Example usage via Python:

```python
from gradio_client import Client

client = Client("[https://huggingface.co/spaces/abade1990/nyc-property-ai](https://huggingface.co/spaces/abade1990/nyc-property-ai)")
result = client.predict(
    borough="Brooklyn",
    building_category="01 ONE FAMILY DWELLINGS",
    year_built=1950,
    gross_sqft=2000,
    land_sqft=2500,
    total_units=1,
    sale_month=6,
    tax_class="1",
    api_name="/predict",
)
print("Valuation Result:", result)
# Returns: ('$377,498', '$327,275 to $422,974', '87.2%')
```
