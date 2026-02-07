# Crop Yield Prediction using Machine Learning

## Problem Understanding
The objective of this project is to build a machine learning model that predicts crop yield (hg/ha) using historical agricultural and climatic data. Accurate crop yield prediction plays an important role in agricultural planning, resource management, and food security. By analyzing past data, machine learning models can help understand how environmental and farming factors influence crop productivity.

This problem is formulated as a **supervised regression task** because the target variable, crop yield, is a continuous numerical value.


## Dataset Description
The dataset used in this project is a publicly available crop yield prediction dataset. It contains historical data related to agricultural production along with climatic factors.

**Key features in the dataset include:**
- Area (Country)
- Item (Crop type)
- Year
- Average rainfall
- Pesticide usage
- Average temperature

**Target variable:**
- `hg/ha_yield` – Crop yield measured in hectograms per hectare

Categorical features such as country and crop type are encoded before training the model.


## Model Pipeline
The machine learning pipeline consists of the following steps:

1. Loading and inspecting the dataset  
2. Handling missing values using mean imputation  
3. Encoding categorical variables using one-hot encoding  
4. Splitting the dataset into training and testing sets  
5. Feature scaling for linear regression  
6. Training machine learning models  
7. Evaluating model performance using standard regression metrics  

Two models were trained and compared:
- **Linear Regression** – Used as a baseline model
- **Random Forest Regressor** – Used to capture non-linear relationships


## Evaluation Metrics
The models were evaluated using the following regression metrics:
- **Mean Absolute Error (MAE)**
- **Root Mean Squared Error (RMSE)**
- **R² Score**

Lower RMSE and higher R² values indicate better model performance.


## Results and Discussion
The Random Forest Regressor outperformed the Linear Regression model by achieving lower RMSE and higher R² score. This indicates that Random Forest was better able to capture the complex, non-linear relationships present in agricultural and climatic data.

Feature importance analysis revealed that **rainfall, pesticide usage, and average temperature** are among the most influential factors affecting crop yield.


## Conclusion
This project demonstrates the application of supervised machine learning techniques to solve a real-world agricultural prediction problem. By comparing a baseline linear model with a more advanced ensemble model, the project highlights the importance of model selection and evaluation.

**Future improvements may include:**
- Hyperparameter tuning
- Adding more environmental features
- Deploying the trained model as a REST API for real-time predictions


## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Google Colab


## How to Run
1. Clone the repository  
2. Install required Python libraries  
3. Open `crop_yield_prediction.ipynb` in Jupyter Notebook or Google Colab  
4. Run all cells sequentially  


## Author
Developed as part of an assignment.
