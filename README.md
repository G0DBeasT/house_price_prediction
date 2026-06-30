# California House Price Prediction

## Overview
This project involves building a machine learning pipeline to predict the median house values in Californian districts. It serves as a regression task utilizing historical housing data.

## Dataset
The project uses the **California Housing dataset**, fetched directly from `scikit-learn`.
- **Instances:** 20,640
- **Features:** 8 numerical features representing demographics and geographical data (e.g., `MedInc` (Median Income), `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`).
- **Target:** `MedHouseVal` (Median House Value in $100,000s).

## Machine Learning Pipeline

1. **Data Loading:**
   The dataset is loaded as a `pandas` DataFrame for straightforward analysis and modeling.

2. **Exploratory Data Analysis (EDA):**
   Summary statistics are generated, and the data is checked for missing values to understand the distributions and scale of the features.

3. **Data Preprocessing:**
   The dataset is split into training and testing sets. Feature scaling is typically performed using `StandardScaler` to ensure regression models converge quickly and perform optimally.

4. **Model Training:**
   The pipeline incorporates regression algorithms such as:
   - **Linear Regression:** As a foundational baseline model.
   - **Random Forest Regressor:** A powerful ensemble model to capture non-linear relationships and interactions among features.

5. **Model Evaluation:**
   The predictive models are evaluated on the test set using standard regression metrics:
   - **Mean Squared Error (MSE)**
   - **R² Score (Coefficient of Determination)**

## Libraries Used
- `pandas`, `numpy` for data manipulation
- `matplotlib`, `seaborn` for data visualization
- `scikit-learn` for data preprocessing, regression models, and evaluation metrics
