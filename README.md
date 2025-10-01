# Hyperparameter Tuning of Regression Models


This project is part of a machine learning course and extends the work from **Lab 2**, where four regression models were trained and evaluated:

- Linear Regression (Ridge)
- Decision Tree Regression
- Random Forest Regression
- Support Vector Regression (SVR with linear kernel)

## Objective
The aim of this lab is to fine-tune the models using **Grid Search** and **Random Search** with 5-fold cross-validation, compare their performance, and select the best model for final evaluation on the test set.

## Results
- **Random Forest** achieved the lowest RMSE on both cross-validation and test sets, making it the best-performing model.  
- **Decision Tree** showed improved performance compared to Lab 2, thanks to hyperparameter tuning.  
- **Ridge Regression** and **SVR** produced higher errors, indicating limited suitability for this dataset.  

## Visualizations
The repository includes plots such as:
- Predicted vs Actual values (Random Forest test results)
- Residual plots
- Error distribution histograms
- Feature importance (Random Forest)

## How to Run
1. Clone this repository  
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd Fine_Tuning-ML
