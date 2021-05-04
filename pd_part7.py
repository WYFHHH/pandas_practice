import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/pandas/data/part7/auto-mpg.csv', header = None)

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

df.head()

df.describe()

pd.set_option('display.max_columns', 10)

print(df.info())

print(df.describe())

print(df['horsepower'].unique())
sr = df['horsepower'].unique().astype('object')
list(sorted(sr))

df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset = ['horsepower'], axis = 0, inplace = True)
df['horsepower'] = df['horsepower'].astype('float')

print(df.describe())

ndf = df[['mpg', 'cylinders', 'horsepower', 'weight']]
ndf

ndf.plot(kind = 'scatter', x='weight', y = 'mpg', c = 'coral', s = 10, figsize = (10, 5))

plt.show()
plt.close()

fig = plt.figure(figsize = (10, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

sns.regplot(x='weight', y='mpg', data=ndf, ax=ax1)
sns.regplot(x='weight', y='mpg', data=ndf, ax=ax2, fit_reg = False, color = 'coral')
plt.show()
plt.close()

df.head()
df.describe()
df.info()
df['origin'] = df['origin'].astype('object')
df['medel year'] = df['model year'].astype('object')

df.info()

df.describe()

ndf

sns.jointplot(x='weight', y='mpg', data = ndf)
sns.jointplot(x='weight', y='mpg', data = ndf, kind='reg')
plt.show()
plt.close()

grid_ndf = sns.pairplot(ndf)
plt.show()
plt.close()
ndf
ndf['weight']
X = ndf[['weight']]
X
y = ndf[['mpg']]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=10)

print( 'train data 개수 : ', len(X_train))
print('test data 개수 : ', len(X_test))


from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

r_square = lr.score(X_test, y_test)
r_square

print('기울기 a : ', lr.coef_)

print('y 절편 : ', lr.intercept_)

y_hat = lr.predict(X)
plt.figure(figsize=(10, 5))
ax1 = sns.distplot(y, hist=False, label='y')
ax2 = sns.distplot(y_hat, hist=False, label='y_hat', ax = ax1, color='red')
plt.show()
plt.close()

y_hat = lr.predict(X)
plt.figure(figsize = (10, 5))
ax1 = sns.distplot(y, hist=False, label='y')
ax2 = sns.distplot(y_hat, hist=False, label='y_hat')
plt.legend(loc='best')
plt.show()
plt.close()



##########
# 다항회귀분석

# 기본 라이브러리 불러오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/pandas/data/part7/auto-mpg.csv', header = None)

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']
df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset=['horsepower'], axis = 0, inplace = True)
df['horsepower'] = df['horsepower'].astype('float')

ndf = df[['mpg', 'cylinders', 'horsepower', 'weight']]

X = ndf[['weight']]
y = ndf[['mpg']]

# 사이킷런 활용, 데이터셋 나누기(7:3)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)

print(X_train.shape)
print(X_test.shape)

# 사이킷런에서 필요한 모듈 가져오기
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 다항식 변환
poly = PolynomialFeatures(degree=2)             # 2차항 적용
# type(poly)
X_train_poly = poly.fit_transform(X_train)      # X_train 데이터를 2차항으로 변형
X_train
X_train_poly

print('원 데이터 : ', X_train.shape)
print('2차항 변환 데이터 : ', X_train_poly.shape)

# train data를 가지고 모형 학습
pr = LinearRegression()
pr.fit(X_train_poly, y_train)

# 학습을 마친 모형에 test data를 적용하여 결정계수 (R-제곱) 계산
X_test_poly = poly.fit_transform(X_test)
r_square = pr.score(X_test_poly, y_test)
print(r_square)

# train data의 산점도와 test data로 예측한 회귀선을 그래프로 출력
y_hat_test = pr.predict(X_test_poly)

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111)
ax.plot(X_train, y_train, 'o', label='Train Data')
ax.plot(X_test, y_hat_test, 'r+', label='Predicted Value')
ax.legend(loc='best')
plt.xlabel('weight')
plt.ylabel('mpg')
plt.show()
plt.close()

#####

X_poly = poly.fit_transform(X)
y_hat = pr.predict(X_poly)

plt.figure(figsize=(10,5))
ax1 = sns.distplot(y, hist=False, label='y')
ax2 = sns.distplot(y_hat, hist=False, label='y_hat', ax = ax1)
plt.show()
plt.close()

#####
# 다중회귀분석

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/pandas/data/part7/auto-mpg.csv', header = None)

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

df['horsepower'].replace('?', np.nan, inplace = True)
df.dropna(subset=['horsepower'], axis = 0, inplace = True)
df['horsepower'] = df['horsepower'].astype('float')

ndf = df[['mpg', 'weight', 'cylinders', 'horsepower']]

# 속성값 선택
X = ndf[['cylinders', 'weight', 'horsepower']]
y = ndf[['mpg']]

# train data 와 test data로 구분 (7:3)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, t_test = train_test_split(X, y, test_size = 0.3, random_state = 10)

print('훈련 데이터 : ', X_train.shape)
print('테스트 데이터 : ', X_test.shape)

# sklearn에서 선형회귀분석 모듈 가져오기
from sklearn.linear_model import LinearRegression

# 단순회귀분석 모형 객체 생성
lr = LinearRegression()

# train data를 가지고 모형 학습
lr.fit(X_train, y_train)

# 학습을 마친 모형에 test data를 적용하여 결정계수(R-제곱) 계산
r_square = lr.score(X_test, y_test)
print(r_square)

# 회귀식의 기울기
print('X 변수의 계수 a : ', lr.coef_)

# 회귀식의 y 절편
print('상수항 b : ', lr.intercept_)

# train data의 산점도와 test data로 예측한 회귀선을 그래프로 출력
y_hat = lr.predict(X_test)

plt.figure(figsize = (10, 5))
ax1 = sns.distplot(y_test, hist=False, label='y_test')
ax2 = sns.distplot(y_hat, hist=False, label='y_hat', ax = ax1)
plt.show()
plt.close()



#############

import pandas as pd
import seaborn as sns

df = sns.load_dataset('titanic')
df.head()

pd.set_option('display.max_columns', 15)

df.info()

rdf = df.drop(['deck', 'embark_town'], axis = 1)

print(rdf.columns.values)

rdf = rdf.dropna(subset = ['age'], how='any', axis = 0)
print(len(rdf))

most_freq = rdf['embarked'].value_counts(dropna=True).idxmax()
print(most_freq)

print(rdf.describe(include = 'all'))

rdf['embarked'].fillna(most_freq, inplace=True)

ndf=rdf[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked']]

# 원핫인코딩 - 범주형 데이터를 모형이 인식 할 수 있도록 숫자형으로 변환
onehot_sex = pd.get_dummies(ndf['sex'])
ndf = pd.concat([ndf, onehot_sex], axis =1)
ndf
onehot_embarked = pd.get_dummies(ndf['embarked'], prefix='town')
ndf = pd.concat([ndf, onehot_embarked], axis=1)
ndf

ndf.drop(['sex', 'embarked'], axis = 1, inplace = True)
ndf.head()


##
# 훈련 / 검증데이터 분할

X = ndf[['pclass', 'age', 'sibsp', 'parch', 'female', 'male', 'town_C', 'town_Q', 'town_S']] # 설명변수
y = ndf[['survived']]   # 예측변수

# 설명변수 데이터를 정규화
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

# train data 와 test data로 구분(7:3)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=10)

print('train data 개수 : ', X_train.shape)
print('test data 개수 : ', X_test.shape)


##
# 모형 학습 및 검증

# sklearn 라이브러리에서 KNN 분류 모형 가져오기
from sklearn.neighbors import KNeighborsClassifier

# 모형 객체 생성(k=5로 설정)
knn = KNeighborsClassifier(n_neighbors=5)

# train data를 가지고 모형 학습
knn.fit(X_train, y_train)

# test data를 가지고 y_hat을 예측(분류)
y_hat = knn.predict(X_test)

print(y_hat[0:10])
print(y_test.values[0:10])

# 모형 성능 평가 - Confusion Matrix 계산
from sklearn import metrics
knn_matrix = metrics.confusion_matrix(y_test, y_hat)
print(knn_matrix)

#####

import pandas as pd
import seaborn as sns

df = sns.load_dataset('titanic')
rdf = df.drop(['deck', 'embark_town'], axis=1)
print(rdf.columns.values)

# age 열에 나이 데이터가 없는 모든 행 삭제 - age 열(891개 중 177개의 NaN 값)
rdf = rdf.dropna(subset=['age'], how = 'any', axis = 0)
len(rdf)

# embarked 열의 NaN 값을 승선도시 중에서 가장 많이 출현한 값으로 치환
most_freq = rdf['embarked'].value_counts(dropna=True).idxmax()
print(most_freq)

print(rdf.describe(include='all'))

rdf['embarked'].fillna(most_freq, inplace=True)

ndf = rdf[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch','embarked']]
ndf.head()
ndf.info()

df.info()

onehot_sex = pd.get_dummies(ndf['sex'])
ndf = pd.concat([ndf, onehot_sex], axis=1)

onehot_embarked = pd.get_dummies(ndf['embarked'], prefix = 'town')
ndf = pd.concat([ndf, onehot_embarked], axis = 1)

ndf.drop(['sex', 'embarked'], axis = 1, inplace =True)
ndf.head()

X = ndf[['pclass', 'age', 'sibsp', 'parch', 'female','male','town_C','town_Q','town_S']]
y = ndf['survived']|

# 설명 변수 데이터를 정규화(normalization)
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

# train data와 test data로 구분 (7:3 비율)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 10)

print('train_data의 개수 : ', X_train.shape)
print('test_data의 개수 : ', X_test.shape)


# sklearn 라이브러리에서 KNN 분류모형 가져오가

from sklearn.neighbors import KNeighborsClassifier

# 모형 객체 생성 (k=5로 설정)
knn = KNeighborsClassifier(n_neighbors=5)

# train data를 가지고 모형 학습
knn.fit(X_train, y_train)

# test data를 가지고 y_hat을 예측(분류)
y_hat = knn.predict(X_test)

print(y_hat[0:10])
print(y_test.values[0:10])

from sklearn import metrics
knn_matrix = metrics.confusion_matrix(y_test, y_hat)
print(knn_matrix)

knn_report = metrics.classification_report(y_test, y_hat)
print(knn_report)


############
# SVM 모형

import pandas as pd
import seaborn as sns

df=sns.load_dataset('titanic')

pd.set_option('display.max_columns', 15)

rdf
rdf = df.drop(['deck', 'embark_town'])

rdf = rdf.dropna(subset=['age'], how='any', axis = 0)

most_freq = rdf['embarked'].value_counts(dropna=True).idxmax()
rdf['embarked'].fillna(most_freq, inplace=True)

ndf = rdf[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked']]

onehot_sex = pd.get_dummies(ndf['sex'])
ndf = pd.concat([ndf, onehot_sex], axis=1)
ndf
onehot_embarked = pd.get_dummies(ndf['embarked'], prefix='town')
ndf = pd.concat([ndf, onehot_embarked], axis = 1)
ndf.drop(['sex', 'embarked'], axis = 1, inplace = True)

X = ndf[['pclass', 'age', 'sibsp', 'parch', 'female', 'male', 'town_C', 'town_Q', 'town_S']]
y = ndf['survived']

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

print('train data 개수 : ', X_train.shape)
print('test data 개수 : ', X_test.shape)


# sklearn 라이브러리에서 SVM분류 모형 가져오기
from sklearn import svm

# 모형 객체 생성 (kernel='rbf' 적용)
svm_model = svm.SVC(kernel='rbf')

# train data를 가지고 학습
svm_model.fit(X_train, y_train)

# test data를 가지고 y_hat 예측(분류)
y_hat = svm_model.predict(X_test)

print(y_hat[1:10])
print(y_test.values[1:10])

from sklearn import metrics
svm_matrix = metrics.confusion_matrix(y_test, y_hat)
print(svm_matrix)

svm_report = metrics.classification_report(y_test, y_hat)
print(svm_report)

rdf.info()


df = sns.load_dataset('titanic')
df.info()
df.head()
removed_df = df.drop(['deck', 'embark_town'], axis=1)

removed_df.info()

removed_df = removed_df.dropna(subset=['age'], how = 'any', axis = 0)
removed_df.head()
removed_df.info()

most_freq = removed_df['embarked'].value_counts(dropna = True).idxmax()
print(most_freq)

removed_df['embarked'].fillna(most_freq, inplace = True)
removed_df.info()

ndf = removed_df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'embarked']]

onehot_sex = pd.get_dummies(ndf['sex'])
onehot_sex
ndf = pd.concat([ndf, onehot_sex], axis = 1)
ndf

onehot_embarked = pd.get_dummies(ndf['embarked'], prefix = 'town')
onehot_embarked
ndf = pd.concat([ndf, onehot_embarked], axis = 1)
ndf
ndf.drop(['sex', 'embarked'], axis = 1, inplace = True)
ndf

X = ndf[['pclass', 'age', 'sibsp', 'parch', 'female', 'male', 'town_C', 'town_Q', 'town_S']]
y = ndf['survived']

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

print('train data 개수 : ', X_train.shape)
print('test data 개수 : ', X_test.shape)

from sklearn import svm

svm_model = svm.SVC(kernel='rbf')

svm_model.fit(X_train, y_train)

y_hat = svm_model.predict(X_test)

print(y_hat[0:10])
print(y_test.values[0:10])

from sklearn import metrics
svm_matrix = metrics.confusion_matrix(y_test, y_hat)
print(svm_matrix)

svm_report = metrics.classification_report(y_test, y_hat)
print(svm_report)


#################
# Decision Tree

import pandas as pd
import numpy as np

# Breast Cancer 데이터셋 가져오기(출처 : UCL ML Repository)
uci_path = 'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
df = pd.read_csv(uci_path, header = None)
df.head()
df.columns = ['id', 'clump', 'cell_size', 'cell_shape', 'adhesion', 'epithlial', 'bare_nuclei', 'chromatin', 'normal_nucleoli', 'mitoses', 'class']

pd.set_option('display.max_columns', 15)


df.head()

df.info()

df.bare_nuclei

df.describe()

df.bare_nuclei.unique()

df.size

print(df['bare_nuclei'].unique())

df['bare_nuclei'].replace('?', np.nan, inplace=True)
df.info()
df.dropna(subset=['bare_nuclei'], axis = 0, inplace = True)
df.info()

df['bare_nuclei'] = df['bare_nuclei'].astype('int')
df.info()

df.describe()

# 속성(변수) 선택
X = df[['clump', 'cell_size', 'cell_shape', 'adhesion', 'epithlial', 'bare_nuclei', 'chromatin', 'normal_nucleoli', 'mitoses']]
y = df['class']

# 설명 변수 데이터 정규화
from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

# train data와 test data로 구분
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=10)

print('train data 개수 : ', X_train.shape)
print('test data 개수 : ', X_test.shape)


# sklearn 라이브러리에서 Decision Tree 분류 모형 가져오기
from sklearn import tree

# 모형 객체 생성
tree_model = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=5)

# train data를 가지고 모형 학습
tree_model.fit(X_train, y_train)

# test data 를 가지고 y_hat 예측(분류)
y_hat = tree_model.predict(X_test)

print(y_hat[0:10])
print(y_test.values[0:10])

from sklearn import metrics
tree_matrix = metrics.confusion_matrix(y_test, y_hat)
print(tree_matrix)

tree_report = metrics.classification_report(y_test, y_hat)
print(tree_report)


import pandas as pd
import matplotlib.pyplot as plt

# wholesale customers 데이터셋 가져오기 (출처 : UCI ML Repository)
uci_path = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv'
df = pd.read_csv(uci_path, header = 0)
df.head()

df.info()

df.describe()

X = df.iloc[:,:]
print(X[:5])

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)

print(X[:5])

# sklearn 라이브러리에서 cluster 군집 모형 가져오기
from sklearn import cluster

# 모형 객체 생성
kmeans = cluster.KMeans(init='k-means++', n_clusters=5, n_init=10)

# 모형 학습
kmeans.fit(X)

# 예측(군집)
cluster_label = kmeans.labels_
print(cluster_label)

# 예측 결과를 데이터 프레임에 추가
df['Cluster'] = cluster_label
print(df.head())

fig = plt.figure(figsize = (10, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
df.plot(kind='scatter', x ='Grocery', y='Frozen', c='Cluster', cmap='Set1', colorbar='False', figsize=(10, 10),ax = ax1)
df.plot(kind='scatter', x='Milk', y='Delicassen', c = 'Cluster', cmap='Set1', colorbar='False', figsize=(10, 10), ax = ax2)

plt.show()
plt.close()

mask = (df['Cluster'] ==0) | (df['Cluster'] == 4)
ndf = df[~mask]

fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ndf.plot(kind='scatter', x='Grocery', y = 'Frozen', c = 'Cluster', cmap= 'Set1', colorbar = False, figsize=(10, 10), ax = ax1)
ndf.plot(kind='scatter', x='Milk', y='Delicassen', c = 'Cluster', cmap='Set1', colorbar=True, figsize=(10,10), ax=ax2)
plt.show()
plt.close()

df

#######
# DBSCAN

import pandas as pd
import folium

file_path = '/Users/heechankang/projects/pythonworkspace/pandas/data/part7/2016_middle_shcool_graduates_report.xlsx'

df = pd.read_excel(file_path, header = 0, index_col=0)

pd.set_option('display.width', None)
pd.set_option('display.max_row', 100)
pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 20)
pd.set_option('display.unicode.east_asian_width',True)

print(df.columns.values)

print(df.head())

print(df.info())
print(df.describe())

mschool_map = folium.Map(location=[37.55, 126.98], tiles='Stamen Terrain', zoom_start=12)

for name, lat, lng in zip(df.학교명, df.위도, df.경도):
    folium.CircleMarker([lat, lng],
                         radius = 5,
                         color='brown',
                         fill=True,
                         fill_color ='coral',
                         fill_opacity = 0.7,
                         popup=name).add_to(mschool_map)

# 지도를 html 파일로 저장하기
mschool_map.save('./seoul_mschool_location.html')


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()
onehot_encoder = preprocessing.OneHotEncoder()

onehot_location = label_encoder.fit_transform(df['지역'])
onehot_code = label_encoder.fit_transform(df['코드'])
onehot_type = label_encoder.fit_transform(df['유형'])
onehot_day = label_encoder.fit_transform(df['주야'])

df['location'] = onehot_location
df['code'] = onehot_code
df['type'] = onehot_type
df['day'] = onehot_day

print(df.head())


from sklearn import cluster

# 분석에 사용할 속성 선택(과학고, 외고국제고, 자사고 진학률)
columns_list = [9, 10, 13]
X = df.iloc[:, columns_list]
print(X[:5])

# 설명 변수 데이터 정규화
X = preprocessing.StandardScaler().fit(X).transform(X)

# DBSCAN 모형 객체 생성
dbm = cluster.DBSCAN(eps=0.2, min_samples=5)

# 모형 학습
dbm.fit(X)

# 예측(군집)
cluster_label = dbm.labels_
print(cluster_label)

df['Cluster'] = cluster_label
print(df.head())


grouped_cols = [0, 1, 3] + columns_list
grouped = df.groupby('Cluster')
for key, group in grouped:
    print('key : ', key)
    print('number : ', len(group))
    print(group.iloc[:, grouped_cols].head())

colors = {-1:'gray', 0:'coral', 1:'blue', 2:'green', 3:'red', 4:'purple', 5:'orange', 6:'brown', 7:'brick', 8:'yellow', 9:'magenta', 10:'cyan'}
cluster_map = folium.Map(location = [37.55, 126.98], tiles = 'Stamen Terrain', zoom_start = 12)

for name, lat, lng, clus in zip(df.학교명, df.위도, df.경도, df.Cluster):
    folium.CircleMarker([lat, lng],
                         radius = 5,
                         color = colors[clus],
                         fill=True,
                         fill_color = colors[clus],
                         fill_opacity=0.7,
                         popup=name).add_to(cluster_map)

cluster_map.save('./seoul_mschool_cluster.html')


# X2 데이터셋에 대하여 위의 과정을 반복(과학고, 외고국제고, 자사고 진학률 + 유형)
columns_list2 = [9, 10, 13,22]
X2 = df.iloc[:, columns_list2]
print(X2[:5])

X2 = preprocessing.StandardScaler().fit(X2).transform(X2)
dbm2 = cluster.DBSCAN(eps=0.2, min_samples=5)
dbm2.fit(X2)
df['Cluster2'] = dbm2.labels_

grouped2_cols = [0, 1, 3] + columns_list2
grouped2 = df.groupby('Cluster2')
for key, group in grouped2:
    print('key : ', key)
    print('number : ', len(group))
    print(group.iloc[:, grouped2_cols].head())
    print()

cluster2_map = folium.Map(location=[37.55, 126.98], tiles = 'Stamen Terrain', zoom_start=12)

for name, lat, lng, clus in zip(df.학교명, df.위도, df.경도, df.Cluster2):
    folium.CircleMarker([lat, lng],
                         radius = 5,
                         color=colors[clus],
                         fill=True,
                         fill_color=colors[clus],
                         fill_opacity = 0.7,
                         popup = name).add_to(cluster2_map)

# 지도를 html 파일로 저장하기
cluster2_map.save('./seoul_mschool_cluster2.html')

#########

columns_list3 = [9, 10]
X3 = df.iloc[:, columns_list3]
print(X3[:5])
print()

X3 = preprocessing.StandardScaler().fit(X3).transform(X3)
dbm3 = cluster.DBSCAN(eps=0.2, min_samples=5)
dbm3.fit(X3)
df['Cluster3'] = dbm3.labels_
grouped3_cols = [0, 1, 3] + columns_list3

grouped3 = df.groupby('Cluster3')
for key, group in grouped3:
    print('key : ', key)
    print('number : ', len(group))
    print(group.iloc[:, grouped3_cols].head())
    print()

cluster3_map = folium.Map(location=[37.55, 126.98], tiles = 'Stamen Terrain', zoom_start = 12)

for name, lat, lng, clus in zip(df.학교명, df.위도, df.경도, df.Cluster3):
    folium.CircleMarker([lat, lng],
                         radius=5,
                         color = colors[clus],
                         fill = True,
                         fill_color=colors[clus],
                         fill_opacity=0.7,
                         popup=name).add_to(cluster3_map)

cluster3_map.save('./seoul_mschool_cluster3.html')

df.head()