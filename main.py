import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(r"C:\Users\User\PycharmProjects\diabetes-data-analysis\diabetes_prediction_dataset.csv")
data.info()

# breaking into numerical and categorical values
data_num = data[["age", "bmi", "HbA1c_level", "blood_glucose_level"]]
data_cat = data[["gender", "hypertension", "heart_disease", "smoking_history", "diabetes"]]

print(data_num.describe())

# making histograms to understand distributions
"""for i in data_num.columns:
    plt.hist(data_num[i])
    plt.title(i)
    plt.show()

print(data_num.corr())
sns.heatmap(data_num.corr())
plt.show() """

# compare diabetes across age, bmi, HbA1c_level and blood_glucose_level
print(pd.pivot_table(data, index = 'diabetes', values = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]))
"""
for i in data_cat.columns:
    sns.barplot(x=data_cat[i].value_counts().index, y=data_cat[i].value_counts()).set_title(i)
    plt.show() """

