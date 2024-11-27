Sales Prediction Using Linear Regression
This repository contains code for predicting sales based on TV advertising expenditure using linear regression. 
The task is part of my CodSoft Data Science Internship, where I applied machine learning techniques to analyze and predict business outcomes.

Project Overview
In this project, I built a linear regression model to predict sales based on the amount spent on TV advertising. 
The dataset used for this task contains three advertising channels (TV, Radio, and Newspaper) and the corresponding sales. 
For this particular task, I focused on predicting sales based only on the TV advertising expenditure.

Key Features
Data Exploration and Preprocessing: Visualizing and preparing the data for modeling.
Linear Regression Model: Built and trained a linear regression model to predict sales.
Model Evaluation: Used metrics like Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared (R²) to evaluate the model's performance.
Error Analysis: Visualized the residuals and compared actual vs. predicted values.

Requirements
Python 3.x
pandas
numpy
matplotlib
seaborn
scikit-learn

File Structure
salesPrediction.py: Python script containing the main code for data analysis, model building, and evaluation.
advertising (1).csv: The dataset used for predicting sales based on advertising expenditure.

Steps
Data Preprocessing:
Import and load the dataset.
Visualize the relationship between TV advertising and sales.
Model Training:
Split the data into training and testing sets.
Train a linear regression model on the training set.
Model Evaluation:
Evaluate the model using MAE, MSE, and R².
Visualize the actual vs. predicted sales.
Error Analysis:
Visualize residuals to assess model performance.

How to Use
Download or clone this repository.
Ensure the dataset file (advertising (1).csv) is available in the same directory as the script.
Install the required libraries
Run the script in a Python environment:
python salesPrediction.py
The script will load the data, build a linear regression model, evaluate it, and display visualizations.

What I Learned
Data cleaning and exploration techniques.
How to build and train a linear regression model.
Evaluating model performance using various metrics.
Visualizing errors and understanding the model's behavior.
Gained deeper insights into using Python for predictive analytics.

This README provides an overview of my project, the requirements, how to use the code, and what I’ve learned from the task. 
