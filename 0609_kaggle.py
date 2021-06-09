import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv('heart_failure_clinical_records_dataset.csv')
data
data.describe(include = 'all')
data.shape
data.info()

data['age'] = data['age'].apply(int)
data

import plotly.express as px
fig = px.imshow(data.corr())
fig.show()

sns.histplot(x='age', data = data)
sns.histplot(x='age', data = data, hue = 'DEATH_EVENT', kde = True, palette='muted')

data.columns
sns.histplot(x='creatinine_phosphokinase', data = data)
sns.histplot(data = data.loc[data['creatinine_phosphokinase']<3000, 'creatinine_phosphokinase'])

sns.histplot(data = data[data['creatinine_phosphokinase']<3000])

data2 = data[data['creatinine_phosphokinase']<3000]

sns.histplot(x = 'creatinine_phosphokinase', data = data2)


sns.histplot(x = 'ejection_fraction', data = data, bins = 14, hue = 'DEATH_EVENT', kde = True)

data.columns
sns.histplot(x = 'platelets', data = data, hue = 'DEATH_EVENT')

sns.boxplot(x = 'DEATH_EVENT', y = 'ejection_fraction', data=data)

sns.boxplot(x = 'smoking', y = 'ejection_fraction', data = data)

sns.violinplot(x = 'DEATH_EVENT', y = 'ejection_fraction', data = data)



######
from sklearn.preprocessing import StandardScaler
data.columns
'''
Index(,
'''
X_num_list = ['age', 'creatinine_phosphokinase', 'ejection_fraction','platelets','serum_creatinine', 'serum_sodium','time',]
X_cat_list = ['anaemia' , 'diabetes', 'high_blood_pressure', 'sex', 'smoking']
X_num = data[X_num_list]
X_cat = data[X_cat_list]
y = data['DEATH_EVENT']

scaler = StandardScaler()
scaler.fit(X_num)
X_scaled = scaler.transform(X_num)
X_scaled = pd.DataFrame(X_scaled)
X_scaled.columns = X_num_list
X_scaled

X = pd.concat([X_scaled, X_cat], axis = 1)
X

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=11)

from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression(max_iter=100)
model_lr.fit(X_train, y_train)
# 학습 부족 에러가 뜨면 max_iter를 늘려주면 됨.

from sklearn.metrics import classification_report
pred = model_lr.predict(X_test)

print(classification_report(y_test, pred))


from xgboost import XGBClassifier
model_xgb = XGBClassifier()
model_xgb.fit(X_train, y_train)

model_xgb.feature_importances_
plt.plot(model_xgb.feature_importances_)
plt.show()
plt.barh(X.columns, model_xgb.feature_importances_)
plt.bar(X.columns, model_xgb.feature_importances_)
plt.xticks(rotation = 90)
plt.show()


sns.jointplot(x = 'ejection_fraction', y = 'serum_creatinine', data = data, hue = 'DEATH_EVENT')
data.columns
data.ejection_fraction





