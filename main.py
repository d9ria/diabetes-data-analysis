import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pycaret
from pycaret.classification import *

data = pd.read_csv(r"C:\Users\User\PycharmProjects\diabetes-data-analysis\diabetes_prediction_dataset.csv")
data.info()

# breaking into numerical and categorical values
data_num = data[["age", "bmi", "HbA1c_level", "blood_glucose_level"]]
data_cat = data[["gender", "hypertension", "heart_disease", "smoking_history", "diabetes"]]

print(data_num.describe())

custom_palette = ['pink', 'blue']

# making histograms to understand distributions
"""for i in data_num.columns:
    plt.hist(data_num[i])
    plt.title(i)
    plt.show() """

data_corr = data[["age", "bmi", "HbA1c_level", "blood_glucose_level", "hypertension", "heart_disease", "diabetes"]]
"""
print(data_corr.corr())
sns.heatmap(data_corr.corr())
plt.show() """

# compare diabetes across age, bmi, HbA1c_level and blood_glucose_level
print(pd.pivot_table(data, index = 'diabetes', values = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]))
"""
for i in data_cat.columns:
    sns.barplot(x=data_cat[i].value_counts().index, y=data_cat[i].value_counts()).set_title(i)
    plt.show() """


"""
plt.subplot(1, 1, 1)
sns.countplot( x=data['diabetes'], hue=data['gender'], palette=custom_palette, linewidth=0.7, alpha=0.8, edgecolor='k', saturation=1)
plt.title("Стать VS. Діабет")

plt.subplot(1, 1, 1)
sns.barplot(x='gender', y='heart_disease', hue='diabetes', data=data,  palette=custom_palette, linewidth=0.7, edgecolor='k', alpha=0.8, saturation=1)
plt.title("Стать VS. Серцеві хвороби & hue = Діабет")

plt.subplot(1, 1, 1)
sns.barplot(x='gender', y='hypertension', hue='diabetes', data=data,  palette=custom_palette, linewidth=0.7, edgecolor='k', alpha=0.8, saturation=1)
plt.title("Стать VS. Гіпертонія & hue = Діабет") """

plt.show()
"""
fig = plt.figure(figsize=(14, 8))

ax = plt.subplot(2, 3, 1)
ax.text(0.5, 0.5, "Вік\n VS. \n Кількісні змінні", fontdict={'fontsize': 22, 'fontweight': 'bold', 'color': 'black', 'ha': 'center', 'va': 'center'})
ax.set_facecolor("black")
ax.axis('off')

plt.subplot(2, 3, 2)
sns.lineplot(x=data['age'], y=data['diabetes'], color="black")

plt.subplot(2, 3, 3)
sns.lineplot(x=data['age'], y=data['heart_disease'], hue=data['diabetes'], palette=custom_palette)


plt.subplot(2, 3, 4)
sns.lineplot(x=data['age'], y=data['hypertension'], hue=data['diabetes'], palette=custom_palette)

plt.subplot(2, 3, 5)
sns.lineplot(x=data['age'],y=data['bmi'], hue=data['diabetes'], palette=custom_palette)

plt.subplot(2, 3, 6)
sns.lineplot(x=data['age'], y=data['blood_glucose_level'], hue=data['diabetes'], palette=custom_palette )

plt.tight_layout()
plt.show() """



