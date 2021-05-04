# import seaborn as sns

# df = sns.load_dataset('titanic')

# nan_deck = df['deck'].value_counts(dropna=False)
# print(nan_deck)

# df.describe()

# print(df.head().isnull())

# print(df.head().notnull())

# print(df.head().isnull().sum(axis=0))

# print(df.head().notnull().sum())


# ##########
# # 판다스 part5, 예제 5-2 p176

# # 라이브러리 불러오기
# import seaborn as sns

# # titanic 데이터셋 가져오기
# df = sns.load_dataset('titanic')

# # for 반복문으로 각 열의 NaN 개수 계산하기
# missing_df = df
# for col in missing_df:
#     missing_count = missing_df[col].value_counts()  # 각 열의 개수 파악
#     try:
#         print(col, ': ', missing_count[True]) # NaN값이 있면 개수 출력
#     except:
#         print(col, ': ', 0) # NaN이 없으면 0개 출력

# ##########################
# # 예제 5-2 p.177

# # NaN 값이 500개 이상인 열을 모두 삭제 - deck 열 (891개 중 688개의 NaN값)
# df_thresh = df.dropna(axis=1, thresh=500)
# print(df_thresh.columns)
# #################

# # age 열에 나이 데이터가 없는 모든 행 삭제 - age 열 (891개 중 177개의 NaN 값)
# df_age = df.dropna(subset=['age'], how= 'any', axis = 0)
# print(len(df_age))

# df_age

# ########################
# # 예제 5-3


# import seaborn as sns

# df = sns.load_dataset('titanic')

# print(df['age'].head(10))
# print()

# mean_age = df['age'].mean(axis=0)
# df['age'].fillna(mean_age, inplace=True)

# print(df['age'].head(10))


# ##############################
# # 예제 5-4

# import seaborn as sns

# df = sns.load_dataset('titanic')

# # embark_town  열의 NaN 값을 승선도시 중에서 가장 많이 출현한 값으로 치환하기
# most_freq = df['embark_town'].value_counts(dropna=True).idxmax()
# print(most_freq)

# df['embark_town'].fillna('most_freq', inplace=True)
# print(df['embark_town'][825:830])

# ###################################

# import seaborn as sns

# df = sns.load_dataset('titanic')

# print(df['embark_town'][825:830])
# print()

# most_freq = df['embark_town'].value_counts(dropna = True).idxmax()
# print(most_freq)

# df['embark_town'].fillna(most_freq, inplace=True)

# print(df['embark_town'][825:830])

# ###################################

# import seaborn as sns
# df = sns.load_dataset('titanic')

# print(df['embark_town'][825:830])

# df['embark_town'].fillna(method='ffill', inplace=True)
# print(df['embark_town'][825:830])

# ####################################

# import pandas as pd

# df = pd.DataFrame({'c1':['a','a','b','a','b'],
#                    'c2':[1,1,1,2,2],
#                    'c3':[1,1,2,2,2]
#                    }
#                  )
# print(df)

# col_dup = df.duplicated()
# print(col_dup)

# col_dup

# ###############

# import pandas as pd

# df

# df2 = df.drop_duplicates()
# df2

# df3 = df.drop_duplicates(['c2', 'c3'])
# df3

# #############################

# import pandas as pd

# df = pd.read_csv('/Users/heechankang/Desktop/pythonworkspace/pandas/data/5674-833/part3/auto-mpg.csv', header=None)

# # 열 이름 지정
# df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']
# df.head(3)

# # mpg를 kpl로 바꿔구지
# mpg_to_kpl = 1.60934/3.78514

# # mpg 를 kpl로 변환한 열을 새로 추가
# df['kpl'] = df['mpg']*mpg_to_kpl
# df.head(3)

# # kpl 열을 소수점 아래 둘째 자리에서 반올림

# df['kpl'] = df['kpl'].round(2)
# df.head(3)

# #######################################

# import pandas as pd

# df = pd.read_csv('/Users/heechankang/Desktop/pythonworkspace/pandas/data/5674-833/part3/auto-mpg.csv', header = None)

# df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

# print(df.dtypes)

# print(df['horsepower'].unique())


# ###############################
# import numpy as np
# df['horsepower'].replace('?', np.nan, inplace=True)
# df.dropna(subset=['horsepower'], axis=0, inplace=True)
# df['horsepower'] = df['horsepower'].astype('float')

# print(df['horsepower'].dtypes)


# print(df['origin'].unique())
# df['origin'].replace({1:'USA', 2:'EU', 3:'JPN'}, inplace=True)
# print(df['origin'].unique())
# print(df['origin'].dtypes)


# df['origin'] = df['origin'].astype('category')
# print(df['origin'].dtype)

# df['origin'] = df['origin'].astype('str')
# print(df['origin'].dtypes)


# print(df['model year'].sample(3))
# df['model year'] = df['model year'].astype('category')
# print(df['model year'].sample(3))

# ################################################################################

# import pandas as pd
# import numpy as np

# df = pd.read_csv('/Users/heechankang/Desktop/pythonworkspace/pandas/data/5674-833/part3/auto-mpg.csv', header = None)


# # 열 이름 지정
# df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

# # horsepower 열의 누락 데이터('?')를 삭제하고 실수형으로 변환
# df['horsepower'].replace('?', np.nan, inplace=True)
# df.dropna(subset=['horsepower'], axis=0, inplace=True)
# df['horsepower']=df['horsepower'].astype('float')

# count, bin_dividers = np.histogram(df['horsepower'], bins=3)
# print(bin_dividers)

# bin_names = ['저출력', '보통출력', '고출력']

# # pd.cut 함수로 각 데이터를 3개의 bin에 할당

# df['hp_bin'] = pd.cut(x=df['horsepower'],
#                       bins = bin_dividers,
#                       labels=bin_names,
#                       include_lowest=True)

# # horsepower를 열, hp_bin 열의 첫 15행 출력
# print(df[['horsepower', 'hp_bin']].head(15))

# #######################################
# df['horsepower'].replace('?', np.nan, include=True)
# df.dropna(subset=['horsepower'], axis = 0, inplace=True)
# df['horsepower']=df['horsepower'].astype('float')

# count, bin_dividers=np.histogram(df['horsepower'], bins=3)
# print(bin_dividers)

# ##########################

# import pandas as pd
# import numpy as np

# # read_csv 함수로 df 생성
# df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/pandas/data/5674-833/part4/auto-mpg.csv', header=0)

# # 열 이름 지정
# df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

# # horsepower 열의 누락 데이터('?')를 삭제하고 실수형으로 변환
# df['horsepower'].replace('?', np.nan, inplace=True)
# df.dropna(subset = ['horsepower'], axis=0, inplace = True)
# df['horsepower'] =df['horsepower'].astype('float')

# # np.histogram 함수로 3개의 bin으로 구분할 경계값의 리스트 구하기
# count, bin_dividers = np.histogram(df['horsepower'], bins=3)
# print(bin_dividers)

# import pandas as pd
# import numpy as np
# import missingno
# import matplotlib
# import matplotlib.pyplot as plt

# %matplotlib inline

# missingno.bar(df, figsize = (10, 5), fontsize=12)

# # 3개의 bin에 이름 지정
# bin_names = ['저출력', '보통출력', '고출력']

# # pd.cut 함수로 각 데이터를 3개의 bin에 할당
# df['hp_bin'] = pd.cut(x=df['horsepower'],
#                      bins=bin_dividers,
#                      labels=bin_names,
#                      include_lowest=True)

# # horsepower 열, hp_bin 열의 첫 15행 출력
# print(df[['horsepower', 'hp_bin']].head())

# # np.histogram 함수로 3개의 bin 으로 구분할 경계값의 리스트 확인
# count, bin_dividers = np.histogram(df['horsepwer'], bins = 3)

# # 3개의 bin 에 이름 저장
# bin_names =


#######################################
# 예제 5-10

# import pandas as pd
# import numpy as np
# import missingno

# df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/pandas/data/5674-833/part4/auto-mpg.csv', header = None)
# df.head()

# df.columns=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']
# df.head()

# df.isnull()

# missingno.bar(df, figsize =(10, 5), fontsize=12)

# df['horsepower'].sum()

# df['horsepower'].replace('?', np.nan, inplace=True)
# df.dropna(subset=['horsepower'], axis = 0, inplace=True)
# df['horsepower'] = df['horsepower'].astype('float')

# count, bin_dividers = np.histogram(df['horsepower'], bins=3)
# print(bin_dividers)

# # 3개의 bin에 이름 지정
# bin_names = ['저출력', '보통출력', '고출력']

# # pd.cut 함수로 각 데이터를 3개의 bin에 할당
# df['hp_bin'] = pd.cut(x=df['horsepower'],
#                       bins=bin_dividers,
#                       labels = bin_names,
#                       include_lowest=True)

# # horsepower 열, hp_bin 열의 첫 15행 출력
# print(df[['horsepower', 'hp_bin']].head(15))

# ############################
# # 예제 5-11

# # np.histogram 함수로 3개의 bin으로 구분할 경계값의 리스트 구하기
# count, bin_dividers = np.histogram(df['horsepower'], bins=3)

# # 3개의 bin에 이름 지정
# bin_names = ['저출력', '보통출력', '고출력']

# # pd.cut 으로 각 데이터를 3개의 bin에 할당
# df['hp_bin'] = pd.cut(x=df['horsepower'],
#                       bins=bin_dividers,
#                       labels=bin_names,
#                       include_lowest=True)

# # hp_bin 열의 범주형 데이터를 더미 변수로 변환
# horsepower_dummies = pd.get_dummies(df['hp_bin'])
# print(horsepower_dummies.head(15))


# ###############
# # 예제 5-12

# # sklearn 라이브러리 불러오기
# from sklearn import preprocessing

# # 전처리를 위한 encoder 객체 만들기
# label_encoder = preprocessing.LabelEncoder()
# onehot_encoder = preprocessing.OneHotEncoder()

# # label encoder로 문자열 범주를 숫자형 범주로 변환
# onehot_labeled = label_encoder.fit_transform(df['hp_bin'].head(15))
# print(onehot_labeled)
# print(type(onehot_labeled))

# # 2차원 행렬로 변경
# onehot_reshaped = onehot_labeled.reshape(len(onehot_labeled), 1)
# print(onehot_reshaped)
# print(type(onehot_reshaped))

# # 희소행렬로 변환
# onehot_fitted = onehot_encoder.fit_transform(onehot_reshaped)
# print(onehot_fitted)
# print(type(onehot_fitted))



# #############################
# # 예제 5-13

# # 라이브러리 불러오기
# import pandas as pd
# import numpy as np

# # read_csv() 함수로 df 생성
# df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/pandas/data/5674-833/part4/auto-mpg.csv', header=None)

# # 열 이름 지정
# df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

# # horsepower 열의 누락 데이터 (?)를 삭제하고 실수형으로 변환
# df['horsepower'].replace('?', np.nan, inplace=True)

# df.dropna(subset = ['horsepower'], axis=0, inplace=True)

# df['horsepower'] = df['horsepower'].astype('float')

# # horsepower 열의 통계 요약 정보로 최대값(max) 확인
# print(df.horsepower.describe())
# print()

# # horsepower 열의 최대값이 절대값으로 모든 데이터를 나눠서 저장
# df.horsepower = df.horsepower/abs(df.horsepower.max())
# print(df.horsepower.head())

# print(df.horsepower.describe())


# ##########################################
# # 예제 5-14

# # horsepower 열의 통계 요약 정보로 최대값(max) 과 최소값(min) 확인
# print(df.horsepower.describe())
# print(())

# # horsepower 열의 최대값의 절대값으로 모든 데이터를 나눠서 저장
# min_x = df.horsepower - df.horsepower.min()
# min_max = df.horsepower.max() - df.horsepower.min()
# df.horsepower = min_x/min_max

# print(df.horsepower.head())
# print()
# print(df.horsepower.describe())

# ################################
# # 예제 5-15

# import pandas as pd

# df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/pandas/data/5674-833/part5/stock-data.csv')

# df.describe()

# df['new_date'] = pd.to_datetime(df['Date'])

# print(df.head())


# df.set_index('new_date', inplace=True)
# df.drop('Date', axis=1, inplace=True)

# df

# #################################
# # 예제 5-16


# import pandas as pd

# dates = ['2019-01-01', '2020-03-01', '2021-06-01']

# ts_dates = pd.to_datetime(dates)
# print(ts_dates)


# pr_day = ts_dates.to_period(freq='D')
# print(pr_day)
# pr_month = ts_dates.to_period(freq='M')
# print(pr_month)
# pr_year = ts_dates.to_period(freq='A')
# print(pr_year)

# ########################
# # 예제 5-17

# import pandas as pd

# ts_ms = pd.date_range(start='2019-01-01',
#                       end=None,
#                       periods= 6,
#                       freq='MS',
#                       tz='Asia/Seoul')
# print(ts_ms)
# #####
# ts_me = pd.date_range('2019-01-01', periods = 6,
#                       freq = 'M',
#                       tz = 'Asia/Seoul')
# print(ts_me)
# #####
# ts_3m = pd.date_range('2019-01-01', periods=6,
#                       freq = '3M',
#                       tz = 'Asia/Seoul')

# print(ts_3m)

# ###################################
# # 예제 5-18

# import pandas as pd

# pr_m = pd.period_range(start = '2019-01-01',
#                        end = None,
#                        periods=3,
#                        freq = 'M')
# print(pr_m)

# ######

# pr_h = pd.period_range(start = '2019-01-01',
#                        end = None,
#                        periods=3,
#                        freq='H')
# print(pr_h)

# pr_2h = pd.period_range(start='2019-01-01',
#                         end = None,
#                         periods=3,
#                         freq = '2H')
# print(pr_2h)
# ########################################
# # 예제 5-19
# import pandas as pd

# df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/pandas/data/5674-833/part5/stock-data.csv', header = 0)
# df.head()

# df['new_date'] = pd.to_datetime(df['Date'])
# print(df.new_date)

# df['Year'] = df['new_date'].dt.year
# df['Month'] = df['new_date'].dt.month
# df['Day'] = df['new_date'].dt.day
# print(df.head())

# #########
# df['Date_yr'] = df['new_date'].dt.to_period(freq='A')
# df['Date_m'] = df['new_date'].dt.to_period(freq='M')
# print(df.head())

# ##########

# df.set_index('Date_m', inplace=True)
# print(df.head())

# #############################################
# # 예제 5-20

# import pandas as pd

# df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/pandas/data/5674-833/part5/stock-data.csv', header = 0)

# df['new_date'] = pd.to_datetime(df['Date'])
# df.set_index('new_date', inplace=True)
# df

# df.index

# df_y = df['2018']
# print(df_y.head())

# df_ym = df.loc['2018-07']

# df_ym_cols = df.loc['2018-07', 'Start':'High']

# df_ymd = df['2018-07-02']
# df_ymd

# df_ymd_range = df['2018-06-25':'2018-06-20']
# df_ymd_range

# df

# today = pd.to_datetime('2018-12-25')
# today
# df['time_delta'] = today - df.index
# df['time_delta']
# df.set_index('time_delta', inplace=True)
# df_180 = df['180 days':'189 days']
# print(df_180)

# df.loc['2018-06']



#################################################
# 복습

import pandas as pd
import numpy as np

df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/pandas/data/5674-833/part5/auto-mpg.csv', header = None)

df.head()

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin' , 'name']

df['horsepower'].replace('?', np.nan, inplace=True)
df.dropna(subset = ['horsepower'], axis = 0, inplace = True)
df['horsepower'] = df['horsepower'].astype('float')

df
bin_names = ['저출력', '중간출력', '고출력']
count, bin_dividers = np.histogram(df['horsepower'], bins=3)
print(count)
print(bin_dividers)

df['hp_bin'] = pd.cut(x=df['horsepower'],
                      bins = bin_dividers,
                      labels = bin_names,
                      include_lowest=True)

print(df[['horsepower', 'hp_bin']])



count, bin_dividers = np.histogram(df['horsepower'], bins=3)

bin_names = ['저출력', '중간출력', '고출력']

df['hp_bin'] = pd.cut(x=df['horsepower'],
                     labels=bin_names,
                     bins = bin_dividers,
                     include_lowest = True)
                     

horsepower_dummies = pd.get_dummies(df['hp_bin'])
print(horsepower_dummies.head(15))

# from sklearn import preprocessing

# label_encoder = preprocessing.LabelEncoder()
# onehot_encoder = preprocessing.OneHotEncoder()

# onehot_labeled = label_encoder.fit_transform(df['hp_bin'].head(15))
# print(onehot_labeled)

# onehot_reshaped = onehot_labeled.reshape(len(onehot_labeled), 1)
# onehot_reshaped

# onehot_fitted = onehot_encoder.fit_transform(onehot_reshaped)

from sklearn import preprocessing

label_encoder = label_encoder.fit_transform(df['hp_bin'].head(15))

label_reshape = onehot_labeled.reshape(len(onehot_labeled),1)

onehot_fitted = onehot_encoder.fit_transform(onehot_reshaped)

onehot_fitted


##############################

import pandas as pd
import numpy as np

df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/pandas/data/5674-833/part5/auto-mpg.csv', header = None)

df

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year' , 'origin', 'name']

df['horsepower'].replace('?', np.nan, inplace = True)
df.dropna(subset=['horsepower'], axis = 0, inplace = True)
df['horsepower'] = df['horsepower'].astype('float')

df['horsepower'].max()

df.horsepower = df.horsepower/abs(df.horsepower.max())
df.horsepower.describe()


min_x = df.horsepower - df.horsepower.min()
min_max = df.horsepower.max() - df.horsepower.min()
df.horsepower = min_x / min_max

df.horsepower.describe()

##############################
import pandas as pd

df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/pandas/data/5674-833/part5/stock-data.csv', header =0)
df
df['new_date'] = pd.to_datetime(df['Date'])
df.head()
df.set_index('new_date', inplace = True)

df.drop('Date', axis = 1, inplace= True)
df

########

import pandas as pd

dates = ['2019-01-01', '2020-03-01', '2021-06-01']

ts_dates = pd.to_datetime(dates)

ts_dates

pr_day = ts_dates.to_period(freq='D')
pr_day

pr_month = ts_dates.to_period(freq = 'M')
pr_month

pr_year = ts_dates.to_period(freq='Y')
pr_year
#############

import pandas as pd

ts_ms = pd.date_range(start='2019-01-01',
                      end = None, 
                      periods = 6,
                      freq = 'MS',
                      tz = 'Asia/Seoul')
ts_ms

ts_me = pd.date_range(start='2019-01-01',
                      periods = 6, 
                      freq = 'M',
                      tz = 'Asia/Seoul')

ts_me

ts_3m = pd.date_range(start = '2019-01-01',
                      periods = 6, 
                      freq='3M',
                      tz = 'Asia/Seoul')

ts_3m

import pandas as pd

pr_m = pd.period_range(start = '2019-01-01',
                       end = None,
                       periods = 30,
                       freq = 'M',)

pr_m

pr_h = pd.period_range(start = '2019-01-01',
                       end = None,
                       periods=30,
                       freq = 'H')

pr_h

pr_2h = pd.period_range(start = '2019-01-01',
                        end = None,
                        periods = 30,
                        freq = '2H')

pr_2h

import pandas as pd

df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/pandas/data/5674-833/part5/stock-data.csv')
df

df.head()

df['new_date'] = pd.to_datetime(df['Date'])
df.head()

df.set_index('new_date', inplace=True)
df.head()

df.drop('Date', axis=1, inplace = True)

df.head()



df['year'] = df['new_date'].dt.year
df.head()

df['month'] = df['new_date'].dt.month
df['day'] = df['new_date'].dt.day

df.head()