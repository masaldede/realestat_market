# 🏠 Real Estate Price Prediction – SGD Regressor with Grid Search

## 🚀 Purpose  
This project aims to predict housing prices using **Stochastic Gradient Descent (SGD) Regression**, enhanced with **hyperparameter optimization via GridSearchCV**. It showcases how machine learning models can be scaled and tuned for real-world applications.

## 🧠 Concepts Covered  
- Scikit-learn’s `SGDRegressor`  
- Feature scaling using `MinMaxScaler`  
- Grid search for hyperparameter optimization  
- `r²` and `MSE` performance evaluation  
- Use of real estate dataset with numeric features

## 🛠️ Technologies Used  
- Python 3.x  
- pandas, numpy  
- matplotlib  
- scikit-learn  
- Dataset: Turkish real estate data from [Dropbox XLSX file](https://www.dropbox.com/s/luoopt5biecb04g/SATILIK_EVI.xlsx?dl=1)

## 📋 Features Used for Prediction  
- `Oda_Sayısı` (Room Count)  
- `Net_m2` (Net Area in m²)  
- `Katı` (Floor Number)  
- `Yaşı` (Building Age)  
- Target: `Fiyat` (Price in Turkish Lira)

## 📦 How to Run
```bash
pip install pandas numpy matplotlib scikit-learn openpyxl
python real_estate_market_v2.py

Note: Internet connection is required to load the dataset from Dropbox.

⚙️ Model Pipeline Summary
	1.	Load and clean dataset
	2.	Scale features and target using MinMaxScaler
	3.	Split into train/test sets
	4.	Define SGDRegressor with elasticnet penalty
	5.	Use GridSearchCV to find optimal hyperparameters
	6.	Evaluate with r² and mean_squared_error
	7.	Visualize prediction performance

📊 Sample Output
	•	Best parameters from grid search
	•	Final r² scores for both training and test sets
	•	Mean squared error
	•	Scatter plots of actual vs. predicted prices

🧠 Learning Outcomes
	•	How to apply SGD regression for large datasets
	•	How to prepare and scale real-world housing data
	•	How to optimize model performance with grid search
	•	How to analyze and visualize prediction success

🔄 Future Improvements
	•	Add categorical features (location, building type)
	•	Try other regression methods (e.g., Random Forest, XGBoost)
	•	Deploy model via Streamlit or Flask
	•	Extend dataset with more samples for generalization

📌 Educational Context

This project was created to deepen understanding of scalable regression techniques and hyperparameter tuning. It reflects practical machine learning workflows for real estate or finance domains.
