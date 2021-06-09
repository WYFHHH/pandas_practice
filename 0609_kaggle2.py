import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from seaborn import palettes
from seaborn.miscplot import palplot


df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
df

df.isna().sum()
df.isnull().sum()

plt.figure(figsize = (10,10))
plt.xticks(rotation = 60)
sns.heatmap(df.corr(), annot = True)
plt.show()

sns.histplot(x = 'age', data = df, hue = 'DEATH_EVENT', kde = True)
sns.histplot(x = df['age'])

sns.kdeplot(x = df['creatinine_phosphokinase'], shade = True, hue = 'DEATH_EVENT', data = df)

sns.kdeplot(x = df['creatinine_phosphokinase'], hue = 'DEATH_EVENT', data = df, fill = True, palette='crest', linewidth = 0, alpha = .3)

from scipy.stats import skew
print(skew(df['age']))
print(skew(df['serum_sodium']))
print(skew(df['serum_creatinine'])) #
print(skew(df['platelets']))  #
print(skew(df['time'])) 
print(skew(df['creatinine_phosphokinase']))  #
print(skew(df['ejection_fraction']))
df.columns

sns.kdeplot(x = df['serum_creatinine2'])
sns.kdeplot(x = df['serum_sodium2'])

df['serum_creatinine2'] = np.log(df['serum_creatinine'])
df['serum_sodium2'] = np.log(df['serum_sodium'])
print(skew(df['serum_creatinine2'])) #
print(skew(df['serum_sodium2'])) #


sns.countplot(df['DEATH_EVENT'])

sns.catplot(x = 'diabetes', y = 'age', hue = 'DEATH_EVENT', kind = 'box', data = df)

features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
       'ejection_fraction', 'high_blood_pressure', 'platelets',
       'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']
 = ['DEATH_EVENT']





 from sklearn.preprocessing import StandardScaler
 sc = StandardScaler()
 X_train = sc.fit_transform(X_train[features1])
 X_test = sc.transform(X_test[features1])

 from sklearn.ensemble import RandomForestClassifier
 rfc = RandomForestClassifier()
 rfc.fit(X_train, y_train)
 rfc = rfc.predict(X_test)

 

 from sklearn.metrics import confusion_matrix

 cm = 