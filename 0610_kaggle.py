import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import folium

df = pd.read_csv('AB_NYC_2019.csv')
df.columns
df.info()
df.shape
df.describe(include = 'all')
df.name.unique()
df.corr()

plt.figure(figsize = (10, 10))
sns.heatmap(df.corr(), annot = True)
plt.show()

df.groupby(['name']).count()


## 지도에 찍어보기
df

df_location = df[['latitude', 'longitude']]
df_location.info()
df_location.iloc[1][0]
df_location

m = folium.Map(
    location = [40.64749, 40.64749],
    zoom_start = 8,
    tiles = 'Cartodb Positron'
)

for i in range(df.shape[0]):
    folium.Circle(
        location = [df_location.iloc[i][0],df_location.iloc[i][1]],
        radius = 50,
        color = '#000000',
        fill = 'crimson',
    ).add_to(m)
    
m.save('map.html')


df.isna().sum()

df.columns
df.reviews_per_month.isna()
df.last_review.isna()
(df.reviews_per_month.isna() & df.last_review.isna()).sum()

df.availability_365.hist(bins=30)
(df['availability_365']==0).sum()
df[df['availability_365']==0]

df.columns
df.drop(['id', 'name', 'latitude', 'longitude'], axis = 1, inplace=True)
df

sns.jointplot(x = 'host_id', y = 'price', data = df)
sns.jointplot(x = 'host_id', y = 'price', data = df, kind = 'hex')

sns.jointplot(x = 'reviews_per_month', y = 'price', data = df)

plt.figure(figsize = (10, 10))
sns.heatmap(df.corr(), annot = True, cmap = 'YlOrRd')
plt.show()

df.columns

sns.boxplot(x = 'neighbourhood_group', y = 'price', data = df)

df.neighbourhood.unique()
df.neighbourhood.value_counts()

sns.barplot(x = df.neighbourhood.unique(), y = df.neighbourhood.value_counts(), )

neigh = df['neighbourhood'].value_counts()
plt.plot(range(len(neigh)), neigh)

df['neighbourhood'] = df['neighbourhood'].apply(lambda x :x if str(x) not in neigh[50:] else 'others')
sns.barplot(x = df.neighbourhood.unique(), y = df.neighbourhood.value_counts())
df['neighbourhood'].value_counts()

sns.rugplot(x = 'price', data = df, height = 1)

df['price'].quantile(0.95)
df['price'].quantile(0.005)

low_price = df['price'].quantile(0.005)
high_price = df['price'].quantile(0.95)

df_price = df[(df['price']<high_price) & (df['price']>low_price)]
df_price
df_price['price'].hist()

df_price['review_exists'] = df_price['reviews_per_month'].isna().apply(lambda x: 'No' if x is True else 'Yes')
df_price['review_exists']

df_price.isna().sum()

df.isna()


X_cat = df_price[['neighbourhood_group', 'neighbourhood', 'room_type', 'review_exists']]
X_cat.head()
X_cat = pd.get_dummies(X_cat)
X_cat

from sklearn.preprocessing import StandardScaler
df_price.info()
X_num = df_price.drop(['neighbourhood_group', 'neighbourhood', 'room_type', 'review_exists'], axis = 1)
X_num = X_num.drop(['host_name', 'last_review'], axis = 1,)
X_num
X_num.info()
X_num.columns

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_num)
X_scaled = scaler.transform(X_num)
X_scaled = pd.DataFrame(X_scaled, index=X_num.index, columns=X_num.columns)

X_scaled

X = X_scaled[['host_id', 'minimum_nights', 'number_of_reviews',
       'reviews_per_month', 'calculated_host_listings_count',
       'availability_365']]
X
y = X_scaled['price']
y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

from xgboost import XGBRegressor
model_reg = XGBRegressor()
model_reg.fit(X_train, y_train)

from lightgbm import LGBMRegressor
model_lgbm = LGBMRegressor()
model_lgbm.fit(X_train, y_train)

y_pred = model_reg.predict(X_train)
y_pred = model_lgbm.predict(X_test)

plt.scatter(x = y_test, y = y_pred)