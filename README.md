### **Project Description: Power Consumption Forecasting**

This project focuses on analyzing and forecasting **power consumption** using time series data. The dataset, sourced from Kaggle, contains multiple variables, including **temperature, humidity, wind speed, and power consumption across different zones**. The project follows a structured approach covering data understanding, preprocessing, model implementation, fine-tuning, and evaluation.

---

## **1. Understanding the Dataset**
- **Key Features Analysis:**  
  The dataset includes weather-related variables (temperature, humidity, wind speed) and power consumption data across multiple zones. Unfortunately due to some limitation in SHAP installation, I use alternative tools to analyse data that can help determine the **feature importance** in forecasting energy demand.
  
- **Challenges in Time Series Forecasting:**  
  - **Seasonality & Trends:** Power consumption varies based on seasonal factors, requiring careful trend analysis.  
  - **Missing Values:** The dataset may have missing timestamps, requiring imputation techniques.  
  - **External Influences:** Factors like economic activities and holidays may impact energy usage.

- **Weather Impact on Power Consumption:**  
  - Scatter plots and correlation matrices are used to analyze the relationship between **temperature, humidity, and wind speed** with power consumption.  
  - Findings suggest that **higher temperatures may increase energy demand (e.g., air conditioning usage)**, whereas wind speed has a **lower correlation**.
The dataset contains 52,416 observations with the following key features: a. Datetime is a timestamp for each observation can be utilised for time series analysis and forecasting. By using this variable we can predict, identify trends, seasonality, and patterns over time (e.g., daily, weekly, or yearly cycles).

b. Temperature described in degrees Celsius (e.g., 6.559). It can be useful factor in power consumption, especially for heating and cooling systems. For example higher temperatures may increase cooling demand, while lower temperatures may increase heating demand. Descriptive statistics shows that its Mean = 18.81°C, Std Dev = 5.81°C, Range = 3.25°C to 40.01°C. Data seems pretty normal shape.

c. Humidity described in percentage (e.g., 73.8). It can be utilized to measure the efficiency of cooling systems. Additionally, high humidity levels may increase power consumption for dehumidification or cooling. Descriptive statistics shows that its Mean = 68.26%, Std Dev = 15.55%, Range = 11.34% to 94.8%. Data points are dispersed from the center point.

d. WindSpeed recorded in meters per second (e.g., 0.083). Wind speed can influence heating demand in colder climates. It can also be utilized for renewable energy generation (e.g., wind turbines). Wind Speed: Mean = 1.96 m/s, Std Dev = 2.35 m/s, Range = 0.05 m/s to 6.48 m/s, indicates data are normally distributed.

e. GeneralDiffuseFlows may represent distributed energy flows or grid-related metrics. [not sure] It could be relevant for understanding grid stability and power distribution.

f. DiffuseFlows Similar to GeneralDiffuseFlows.

g. PowerConsumption_Zone1, PowerConsumption_Zone2, PowerConsumption_Zone3 are basically three different zones of power consumption patterns. These could be useful for load management and resources allocation forecasting. Descriptive data of powe consumption zones indicated highly varied among each other.

Correlation matrix shows following fetures: Temperature positively correlates with power consumption (Zone 1 (0.44), Zone 2 (0.38), Zone 3 (0.49)). This indicates higher power consumption during higher temperatures, may be due to air conditioning use. Humidity negatively correlates with power consumption (Zone 1 (-0.29), Zone 2 (-0.29), Zone 3 (-0.23)). Higher humidity might be linked to lower energy demand, may be due to moderate temperature effects. Wind Speed has a weak positive correlation with power consumption (Zone 1 (0.17), Zone 2 (0.15), Zone 3 (0.28)). This suggests that wind speed may not significantly impact energy usage. General Diffuse Flows and Diffuse Flows (Solar Radiation) positively correlated with power consumption (0.17–0.19) but not strongly. More solar radiation may slightly increase power usage, possibly due to cooling needs. Power Consumption Across Zones strong correlations between zones (0.75 - 0.83), indicating a consistent pattern of electricity usage across areas.


Temperature               0
Humidity                  0
WindSpeed                 0
PowerConsumption_Zone1    0
dtype: int64
        Temperature      Humidity     WindSpeed  PowerConsumption_Zone1
count  52416.000000  52416.000000  52416.000000            52416.000000
mean      18.810024     68.259518      1.959489            32344.970564
std        5.815476     15.551177      2.348862             7130.562564
min        3.247000     11.340000      0.050000            13895.696200
25%       14.410000     58.310000      0.078000            26310.668692
50%       18.780000     69.860000      0.086000            32265.920340
75%       22.890000     81.400000      4.915000            37309.018185
max       40.010000     94.800000      6.483000            52204.395120
Temperature               0
Humidity                  0
WindSpeed                 0
PowerConsumption_Zone1    0
dtype: int64
        Temperature      Humidity     WindSpeed  PowerConsumption_Zone1
count  52327.000000  52327.000000  52327.000000            52327.000000
mean      18.777895     68.344047      1.958385            32334.316690
std        5.767739     15.427596      2.348607             7131.504052
min        3.247000     12.270000      0.050000            13895.696200
25%       14.410000     58.355000      0.078000            26300.146125
50%       18.760000     69.900000      0.086000            32252.164260
75%       22.860000     81.400000      4.915000            37289.933300
max       36.250000     94.800000      6.483000            52204.395120
---

## **2. Data Preprocessing**
- **Handling Missing & Inconsistent Data:**  
  - Used **linear interpolation and forward/backward filling** to handle missing values.
  - **Outlier Removal:** Applied **z-score filtering** to remove extreme values beyond 3 standard deviations.

- **Dataset Splitting for Training & Evaluation:**  
  - **Training (70%)**, **Validation (15%)**, **Testing (15%)** split was applied.
  Training set size: 35200
Validation set size: 7543
Test set size: 7544
  - Time-based splitting ensures that future values are predicted using past data.
Input sequences shape: (52392, 24, 4)
Target values shape: (52392, 4)
- **Tokenization for Transformer Models:**  
  - The dataset was **normalized using MinMaxScaler** and structured into **sequences (time windows)** for deep learning models.

---

## **3. Model Implementation**
- **Vanilla Transformer Model:**  
  - Implemented a **basic Transformer architecture** for time series forecasting, using **multi-head self-attention** to capture dependencies.

- **PatchTST Model:**  
  - Implemented **PatchTST**, an advanced time series transformer, which processes the input as **patches** to improve long-term forecasting.

- **Custom Model (Optional):**  
  - Users can modify or extend the architecture for improved results.

---

## **4. Model Fine-Tuning**
- **Hyperparameter Optimization:**  
  - Experimented with different values for **learning rate, batch size, sequence length, and attention heads**.
  - **AdamW optimizer** with weight decay was used for better convergence.

- **Regularization Techniques:**  
  - Adjusted **dropout rates** to prevent overfitting.
  - **Xavier weight initialization** was applied.

- **Early Stopping & Learning Rate Scheduling:**  
  - Implemented **ReduceLROnPlateau** to **reduce learning rate when validation loss stagnates**.
  - **Early stopping** prevents unnecessary training if performance stops improving.

---

## **5. Evaluation and Visualization**
- **Evaluation Metrics Used:**  
  - **MAE (Mean Absolute Error)**
  - **RMSE (Root Mean Squared Error)**
  - **MAPE (Mean Absolute Percentage Error)**
  - **R² Score**

- **Visualization:**  
  - **Actual vs. Predicted Power Consumption** plotted for each model.
  - **Comparison of model performance** across zones using bar charts.

- **Best Performing Model:**  
  - The model performance varied across different zones, with **PatchTST outperforming the Vanilla Transformer** in most cases.

---

## **Status of Tasks**
| Task | Status |
|------|--------|
| **Understanding the Dataset (SHAP Analysis, Challenges, Weather Impact)** | Partially Completed (SHAP analysis not explicitly performed) |
| **Data Preprocessing (Handling Missing Data, Splitting, Tokenization)** | Completed |
| **Model Implementation (Vanilla Transformer, PatchTST, Custom Model)** | Completed |
| **Hyperparameter Tuning & Regularization** | Completed |
| **Early Stopping & Learning Rate Scheduling** | Completed |
| **Evaluation & Visualization (Metrics, Plots, Model Comparison)** | Completed | 🚀