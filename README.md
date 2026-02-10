# 🏥 Medical Cost Prediction via Linear Regression

## 🌐 Dataset used:
-  https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv

## ✨ What does it do?
* **Problem:** Prediction of patient medical costs based on age, BMI and lifestyle choices.
* **Finding:** Smoking status is highest driver of cost, addition of est $23 grand to the bill regardless of age.
* **R2 results:** 0.77
* **RMSE results:** 5800

* ## 🚩 Issues & Solution
* **Low R2 Score:** 0.13 which meant terrible accuracy 
* **Fix:** Used one-hot encoding via `pd.get_dummies()` which showed that smoking status was a critical part, increasing accuracy

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Library:** 
  * `scikit-learn`
  * `pandas` and `numpy` (data manipulation)
  * `matplotlib` and `seaborn` (data visualization)
* **Algorithim:** Linear Regression

## 🚀 How to Run

### Prerequisites
Ensure you have Python installed. You can install the required libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```
  
## 📊 Visuals
<img width="292" height="241" alt="{BB1904EB-8438-44F0-8757-F27438789D6B}" src="https://github.com/user-attachments/assets/4de1d963-3acf-487e-a6eb-1962c2b5bef3" />

### 1. Actual vs. Predicted Costs
*The tight clustering in the bottom left shows high accuracy for non-smokers. The variance increases for higher-cost patients (smokers).*
<img width="1127" height="1154" alt="{2D9C0248-D485-4000-9367-8E2355367755}" src="https://github.com/user-attachments/assets/a62f1a3d-8f6a-4c44-9263-0ac71665008c" />

### 2. Feature Importance (Coefficients)
*This chart visualizes the dollar impact of each feature. Note the massive impact of `smoker_yes`.*
<img width="1127" height="1176" alt="{D8049FE1-8EC4-4C80-8EB8-259F5C9DB126}" src="https://github.com/user-attachments/assets/80d344ba-4248-42ed-a71e-1b9a5a41fbe8" />


