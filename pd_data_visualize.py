# # ######################################
# # # pandas 실습 1

# # import pandas as pd

# # df = pd.DataFrame(
# #         {
# #                 'a': [4,5,6],
# #                 'b':[7,8,9],
# #                 'c':[10,11,12],
# #         },
# #                 index=[1,2,3]
# # )
# # # list 형태의 데이터 : 판다스에서는 Series라고 함.
# # # dictionary 형태로 지정

# # df["a"]
# # df['b']
# # df['c']
# # df['C'] #대소문자 구분해야함.

# # df[['a', 'b']]
# # # 두 가지를 부를 땐 리스트로 묶어서 부르기
# # # 따로 부를 땐 쉼표, 이어서 부를땐 :

# # df.loc[1] # row 데이터 뽑아보기

# # df.loc[2]

# # df.loc[3, 'a']
# # # 3행 a열 원소 뽑기

# # df.loc[[1,2], ['a', 'b']]
# # # 1,2행 a,b열

# # df = pd.DataFrame(
# #         [[4,7,10],
# #         [5,8,11],
# #         [6,9,12]],
# #         index=[1,2,3],
# #         columns=['a', 'b','c']
# # )
# # df[['b'] < 8]
# # # Series를 불러서 한번 에 DF만들기

# # import numpy as np

# # df = pd.DataFrame(
# #         {
# #         "a":[4,5,6,6,np.nan],
# #         'b':[7,8,np.nan,9,9],
# #         'c':[10,11,12,np.nan, 12]
# #         },
# #         index=pd.MultiIndex.from_tuples(
# #         [('d',1), ('d',2), ('e',2), ('e',3), ('e',4)],
# #         names=['n','v'])
# # )
# # # index 를 tuple 형태로 multi-index로 만들어줌.
# # # 뭔가했더니,, 그냥 인덱스 열 두개 넣어버리고 키값으로 묶어버리는 게 나을듯

# # ############################################
# # # pandas 실습 2

# # # 딕셔너리에서 시리즈로 변환

# # #불러오기
# # import pandas as pd

# # # key:values 구조를 갖는 dict 를 만들고 변수 dict_data로 저장
# # dict_data = {'a':1, 'b':2, 'c':3}
# # dict_data

# # # 판다스 Series() 함수로 딕셔너리(dict_data)를 시리즈로 변환. 변수 sr에 저장
# # sr = pd.Series(dict_data)
# # sr

# # # 변수 sr의 자료형 출력
# # print(type(sr))

# # # 변수 sr에 저장되어있는 시리즈 객체를 출력
# # print(sr)

# # # 시리즈 인덱스
# # import pandas as pd

# # # 리스트를 시리즈로 변환하여 변수 sr에 저장
# # list_data = ['2019-01-02', 3.14, 'ABC', 100, True]
# # sr = pd.Series(list_data)
# # print(sr)

# # # 인덱스 배열은 변수 idx에 저장, 데이터 값 배열은 변수 val에 저장
# # idx = sr.index
# # val = sr.values
# # print(idx)
# # print(val)

# # # tuple을 시리즈로 변환(index 옵션에 인덱스 이름을 지정)
# # tup_data = ('영인', '2010-05-01', '여', True)
# # sr = pd.Series(tup_data, index = ['이름', '생년월일', '성별', '학생여부'])
# # print(sr)

# # # 원소를 1개 서낵
# # print(sr[0])  # sr의 1번째 원소를 선택(정수형 위치인덱스 활용)
# # print(sr['이름']) # '이름' 라벨을 가진 원소를 선택(인덱스 이름을 활용)

# # # 여러 개의 원소를 선택 (인덱스 리스트 활용)
# # print(sr[[1,2]])
# # print(sr[['생년월일', '성별']])

# # # 여러 개의 원소를 선택 (인덱스 범위 지정)
# # print(sr[1:2])
# # print(sr['생년월일':'성별'])

# # ############################################
# # # pandas 실습 3
# # import pandas as pd
# # import numpy as np
# # df = pd.DataFrame(
# #     {'a': [4,5,6],
# #     'b': [7,8,9],
# #     'c':[10,11,12],
# #     }, index = pd.MultiIndex.from_tuples(
# #         [('d', 1), ('d', 2), ('e',2)],
# #         names=['n', 'v']
# #     )
# # )

# # df[df>7]

# # df[df.b>7]

# # df.b>7

# # df[df.a<7]

# # df[df.c >7]

# # df[df['c']>7]

# # df[df['a']<7]

# # df[df['c']>=7]

# # df

# # df.drop_duplicates()
# # # 현재 중복된 값 없음.

# # # 임의로 중복된 값 생성
# # df2 = pd.DataFrame(
# #     {'a':[4,5,6,6,],
# #     'b':[7,8,9,9],
# #     'c':[10,11,12,12]},
# #     index = pd.MultiIndex.from_tuples(
# #         [('d',1), ('d',2), ('e',2),('e',3)],
# #         names = ['n', 'v']
# #     )
# # )
# # df2

# # df2.drop_duplicates()

# # df2

# # df2.drop_duplicates(inplace=True)
# # # in place of noun: ~을 대신하다
# # # 파이썬에서 권장하지 않는 방식

# # df2 = pd.DataFrame(
# #     {'a':[4,5,6,6,],
# #     'b':[7,8,9,9],
# #     'c':[10,11,12,12]},
# #     index = pd.MultiIndex.from_tuples(
# #         [('d',1), ('d',2), ('e',2),('e',3)],
# #         names = ['n', 'v']
# #     )
# # )

# # df2 = df2.drop_duplicates()
# # df2


# # df2 = pd.DataFrame(
# #     {'a':[4,5,6,6,],
# #     'b':[7,8,9,9],
# #     'c':[10,11,12,12]},
# #     index = pd.MultiIndex.from_tuples(
# #         [('d',1), ('d',2), ('e',2),('e',3)],
# #         names = ['n', 'v']
# #     )
# # )

# # df2 = df2.drop_duplicates(keep='last')
# # df2


# # ############################################
# # # pandas 실습 4
# # df
# # df['a'] != 7

# # df['b'] !=7

# # df[df['b'] !=7] # 7이 아닌 값만 가지고 오고 싶을 때

# # # df.column.isin(values)
# # df.['a'].isin?
# # # Use a list of one element instead: values는 리스트 형태

# # df['a'].isin([5]) # 특히 한글 사용 시 유용

# # df.a.isin([5])
# # # a columns에 원소 5가 포함되어 있는지 확인

# # pd.isnull(df)


# # df3 = pd.DataFrame(
# #         {
# #         "a":[4,5,6,6,np.nan],
# #         'b':[7,8,np.nan,9,9],
# #         'c':[10,11,12,np.nan, 12]
# #         },
# #         index=pd.MultiIndex.from_tuples(
# #         [('d',1), ('d',2), ('e',2), ('e',3), ('e',4)],
# #         names=['n','v'])
# # )
# # # 임의로 null 값 생성

# # pd.isnull(df3)

# # df3['a'].isnull()

# # df3['a'].isnull().sum()

# # pd.notnull(df3)

# # df3.notnull()

# # df3.a.notnull()

# # df3.notnull().sum()
# # # null 이 아닌 총 개수

# # df3.any()

# # ~df3.a.notnull()
# # df3.a.notnull()

# # 1 and 1

# # True and False

# # df3[df3.b ==7] | df3[df3.a==5]

# # df3[df3.b == 7] | df3[df3.a == 5]



# # ############################################
# # # pandas 실습 5

# # df3.head()

# # df3. head(2)

# # df3.tail(2)

# # df3.sample(frac=0.5)

# # df3.sample(frac=0.7)

# # df3.sample(frac=0.3)

# # df3.sample(n=10)
# # # 현재 데이터 5개뿐이라 에러

# # df3.sample(n=5)

# # df3
# # # 비율로 하려면 ? frac =0.X
# # # 지정된 횟수, 개수로 샘플링 하고싶다며? n=X

# # df3.iloc[10:20]
# # # 범위에 값이 없음.

# # df3.iloc[:2]
# # # 0, 1 가지고옴.

# # df3.iloc[:3]

# # df3.iloc[3:]
# # # 3부터 끝까지.

# # df3.iloc[-2:]

# # df3.iloc[-2:-1]

# # df4 = pd.DataFrame({'a':[1,10,8,11,-1],
# #                     'b':list('abdce'),
# #                     'c': [1.0, 2.0, np.nan, 3.0, 4.0],})
# # df4.nlargest(1, 'a') # a컬럼에서 가장 큰 값만 가지고오기

# # df4.nlargest(1, 'b')
# # # Type Error 발생, 숫자로 구성된 컬럼에서만 가져 올 수 있음.

# # df4.nlargest(2, 'c')
# # df4

# # df4.nsmallest(3, 'a')

# # ############################################
# # # pandas 실습 6
# # import pandas as pd
# # import seaborn as sns

# # df = sns.load_dataset('iris')
# # df.head()

# # df[['sepal_width', 'sepal_length', 'species']]

# # columns = ['sepal_width', 'sepal_length', 'species']
# # df[columns]

# # df[columns].head()

# # df['sepal_width'].head()

# # df.sepal_width.head()

# # df.filter(regex='regex')

# # df.filter(regex='\.')

# # df.filter(regex='^sepal')
# # df

# # df.filter(regex='^se')

# # df.filter(regex='es$')

# # df.filter(regex='^(?!species$).*')
# # # Matches strings except the string 'sprecies'

# # df.loc[2:4]
# # df.loc[2:5, 'sepal_width':'petal_width']

# # df.iloc[:3, [1,3]]

# # df.loc[df['sepal_length'] > 5, ['sepal_length', 'sepal_width']]

# # df.loc[df['sepal_length'] >5, ['sepal_length', 'sepal_width']].head()


# # ############################################
# # # pandas 실습  3주차 1일
# # import pandas as pd
# # import numpy as np
# # import seaborn as sns

# # # iris data 사용
# # df = sns.load_dataset('iris')

# # df.shape

# # df.head(2)

# # df['species'].value_counts()

# # df['petal_width'].value_counts()

# # pd.DataFrame(df['petal_width'].value_counts())

# # len(df)

# # df.shape[0] # 행만 가지고 오고 싶을 때

# # df.shape[1] # 열만 가지고 오고 싶을 때

# # len(df) == df.shape[0]

# # df['species'].nunique()
# # # unique 한 종류가 3개임을 확인

# # df.describe(include='all')

# # df.describe(include=[np.object])

# # df.describe(exclude=[np.object])

# # df.describe(include=[np.number])
# # # 숫자형 타입만 보여짐

# # df['petal_width'].sum() # 합계

# # df['petal_width'].count() # 개수

# # df['petal_width'].median() # 특정 컬럼 지정해서 중위수, 중간값 지정

# # df.median()

# # df['petal_width'].mean()

# # df.mean()

# # pd.DataFrame(df.mean())

# # df['petal_width'].quantile([0.25, 0.75])

# # df.max()   #최대값

# # df.min()    # 최소값

# # df.var() # 분산

# # df.std() # 표준편차

# # df.corr() # 상관관계


# # ############################################
# # # pandas 실습  3주차 2일

# # df.apply?

# # df.apply(lambda x: x[0])

# # df.apply(lambda x: x[1])

# # df['species'].apply(lambda x : x[0])
# # # species 에서 첫 번째 글자만 가지고 옴.

# # df['species'].apply(lambda x : x[:3])
# # # species 에서 세 번째 글자까지 가지고 옴.

# # df['species_3'] = df['species'].apply(lambda x:x[:3])
# # # species 에서 3글자만 뽑아다가 컬럼 하나를 새로 만듦.

# # df

# # df.head()

# # def smp(x):
# #     # 뒤에서 세번째 까지 문자를 가져오는 함수
# #     x = x[-3:]
# #     return x

# # df['species_3'] = df['species'].apply(lambda x:x[:3])

# # df

# # df['species-3']=df['species'].apply(smp)

# # df

# # ############################################
# # # pandas 실습  3주차 3일
# # # 결측치 다루기

# # import pandas as pd
# # import numpy as np
# # df = pd.DataFrame([])
# # df.dropna?

# # df = pd.DataFrame([[np.nan, 2, np.nan, 0], 
# #                    [3,4,np.nan, 1],
# #                    [np.nan,np.nan,np.nan,5]],
# #                    columns = list('ABCD'))
# # df

# # df.dropna(axis=0, how='all')
# # # axis = 0: 행, axis = 1: 열, how = 'all': 전부 다 null이면 drop

# # df.dropna(axis=1, how='all')

# # df.dropna(axis=0, how = 'any')

# # df.dropna(axis=1, how = 'any')

# # df.fillna(0)

# # values = {"A":0, "B":1, "C":2, "D":3}
# # df.fillna(value = values)

# # df['A'].mean()

# # df['D'].mean()

# # df.fillna(df['D'].mean())

# # df['D'].median()

# # fill_na_value = df['D'].max()
# # fill_na_value

# # df.fillna(fill_na_value)

# # df.isnull()

# # df.isnull().sum()

# # df.notnull().sum()

# # ############################################
# # # pandas 실습  3주차 4일
# # # assign으로 새로운 컬럼 만들기

# # import pandas as pd
# # import numpy as np

# # df = pd.DataFrame({
# #     "A":range(1, 11), "B":np.random.randn(10)
# #     })

# # df

# # df.assign?

# # df.assign(ln_A = lambda x:np.log(x.A))

# # df.assign(ln_A= lambda x: np.log(x.A)).head()

# # df['ln_A']=np.log(df.A)
# # df['ln_A']

# # df.ln_A = np.log(df.A)
# # df

# # ############################################
# # # pandas 실습  3주차 5일
# # # qcut 으로 binning, bucketing 하기

# # pd.qcut?
# # df.A
# # pd.qcut(df.A, 3, labels = ['good', 'median', 'bad'])

# # pd.qcut(df.B, 2, labels= ['good', 'bad'])
# # df.B

# # df.max(axis=0)

# # df.min(axis=0)

# # df['A'].clip(lower = 3, upper = 7)

# # df['B'].clip(lower= 0, upper =1)
# # df

# # df['B']

# # df['B'].abs()


# # ############################################
# # # pandas 실습  4주차 1일
# # # 데이터 재형성_Reshaing Data

# # import pandas as pd
# # import seaborn as sns

# # df = sns.load_dataset('mpg')

# # df.shape
# # df.info()
# # df.describe()

# # df.head()

# # df.sort_values?

# # df.sort_values('mpg').head()

# # df.sort_values('cylinders').head()

# # df.sort_values('cylinders', ascending=False).head()

# # df.rename(columns = {'model_year':'year'})

# # df.rename({'year':'model_year'}, axis=0)

# # df.sort_index().head()

# # df.reset_index?

# # df2 = df.drop(columns=['mpg', 'model_year'])
# # df2


# # ############################################
# # # pandas 실습  4주차 2일
# # # 깔끔한 데이터 만들기(melt, pivot)

# # pd.melt?

# # df = pd.DataFrame({'A':{0:'a', 1:'b', 2:'c'},
# #                    'B':{0:1, 1:3, 2:5},
# #                    'C':{0:2, 1:4, 2:6}})

# # df

# # pd.melt(df, id_vars=['A'], value_vars=['B'])

# # df2 = pd.melt(df, id_vars=['A'], value_vars=['B', 'C'])
# # df2
# # df

# # df3 = pd.melt(df, value_vars = ['A', 'B', 'C'])
# # df3

# # df.pivot?

# # df4 = pd.DataFrame(
# #     {
# #         'bar':['A','B','C','A','B','C'],
# #         'baz':[1,2,3,4,5,6],
# #         'foo':['one', 'one','one','two','two','two']
# #     }
# # )

# # df4
# # df4.pivot(index='foo', columns='bar', values='baz')

# # df4.pivot(index='foo', columns='bar', values='baz').reset_index()

# # df5 = df4.pivot(index='foo', columns='bar', values='baz').reset_index()
# # df5

# # df5.melt(id_vars='foo', value_vars=['A', 'B', 'C'])
# # df5

# # df5.melt(id_vars=['foo'], value_vars=['A', 'B', 'C']).sort_values(by=['bar'])

# # df5.melt(id_vars=['foo'], value_vars=['A', 'B', 'C']).sort_values(by=['foo', 'bar'])

# # df5.melt(id_vars=['foo'], value_vars=['A', 'B', 'C']).sort_values(by=['foo', 'bar']).rename(columns={'value':'baz'})
# # df4

# # ############################################
# # # pandas 실습  4주차 3일
# # # 데이터 합치기 Pandas concat
# # import pandas as pd

# # pd.concat?

# # s1 = pd.Series(['a', 'b'])
# # s1

# # s2 = pd.Series(['c', 'd'])
# # s2

# # pd.concat([s1, s2])

# # pd.concat([s1, s2], keys=['s1' ,'s2'])

# # pd.concat([s1, s2], keys=['s1','s2'], names=['Series name', 'Row ID'])

# # df1 = pd.DataFrame([['a', 1], ['b',2]], columns=['letter', 'number'])
# # df1

# # df2 = pd.DataFrame([['c',3], ['d',4]], columns=['letter', 'number'])
# # df2

# # pd. concat([df1, df2])

# # df3 = pd.DataFrame([['c',3,'cat'], ['d', 4,'dog']], columns=['letter', 'number', 'animal'])
# # df3
# # pd.concat([df1, df3])

# # pd. concat([df1, df3], join="inner")

# # pd.concat([df1, df3], sort=False)

# # df4 = pd.DataFrame([['bird', 'polly'], ['monkey', 'george']], columns=['animal', 'name'])
# # df4

# # df5 = pd. DataFrame([1], index=['a'])
# # df5

# # df6 = pd.DataFrame([2], index=['a'])
# # df6

# # pd.concat([df5, df6], verify_integrity=True)

# # ############################################
# # # pandas 실습  4주차 4일
# # # 데이터 합치기_pandas merge

# # import pandas as pd

# # adf = pd.DataFrame(
# #     {
# #         'x1':['A', 'B', 'C'],
# #         'x2':[1,2,3]
# #     }
# # )

# # bdf = pd.DataFrame(
# #     {
# #         'x1':['A', 'B', 'D'],
# #         'x3':['T','F','T']
# #     }
# # )

# # adf
# # bdf

# # pd.merge(adf, bdf, how = 'left', on ='x1')

# # pd.merge(adf, bdf, how='right', on='x1')

# # pd.merge(adf, bdf, how = 'inner', on='x1')

# # pd.merge(adf, bdf, how = 'outer', on='x1')

# # pd.merge?

# # adf.x1

# # bdf.x1

# # adf.x1.isin(bdf.x1)

# # adf[adf.x1.isin(bdf.x1)]

# # adf[~adf.x1.isin(bdf.x1)]

# # ydf = pd.DataFrame(
# #     {
# #         'x1': ['A', 'B', 'C'],
# #         'x2': [1,2,3]
# #     }
# # )

# # ydf

# # zdf = pd.DataFrame(
# #     {
# #         'x1':['B', 'C', 'D'],
# #         'x2':[2,3,4],
# #     }
# # )

# # pd.merge(ydf, zdf)

# # pd.merge(ydf, zdf, how = 'outer', indicator = True)

# # pd.merge(ydf, zdf, how='outer', indicator=True).query('_merge=="left_only"')

# # pd.merge(ydf, zdf, how = 'outer', indicator = True).query('_merge=="left_only"').drop(columns=['_merge'])

# # ############################################
# # # pandas 실습  4주차 5일
# # # 데이터 집계활용_groupby

# # import pandas as pd
# # import seaborn as sns

# # df = sns.load_dataset('mpg')

# # df.head(3)

# # df.groupby(by='origin').size()

# # df['origin'].value_counts()

# # df.groupby(by='origin').min()

# # df.groupby(by='origin').max()

# # df.groupby(by='origin').mean()

# # df.groupby(by='origin')['cylinders'].size()
# # df.groupby(by='origin')['cylinders'].mean()

# # df.groupby(by='origin')['cylinders'].median()

# # df.groupby(['model_year', 'origin'])['cylinders'].mean()

# # pd.DataFrame(df.groupby(['model_year', 'origin'])['cylinders'].mean())

# # df2 = pd.DataFrame(
# #     [[4,7,10],
# #     [5,11,8],
# #     [6,9,12]],
# #     index=[1,2,3],
# #     columns=['a','b','c']
# # )

# # df2

# # df2.shift(1)

# # df2.shift(-1)

# # df2['b'].shift(-1)

# # df2['b'].shift(2)

# # df['model_year'].rank(method='min')

# # df['model_year'].rank(method='min').value_counts()

# # df['model_year']

# # df['model_year'].rank(method='min').value_counts()

# # df['model_year'].rank(method='max').value_counts()

# # df['model_year'].rank(pct=True)

# # df['model_year'].rank(method='first').head()

# # df2

# # df2.cumsum()

# # df2.cummax()

# # df2.cummin()

# # df2.cumprod()

# # df2

# ##############################################################################

# import pandas as pd
# import matplotlib.pyplot as plt
# plt.style.use('classic') # 스타일 서식 지정

# # read_csv로 df 생성
# df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/pandas/data/part4/auto-mpg.csv', header = None)
# df.head()

# df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']
# df.head()

# df['mpg'].plot(kind = 'hist', bins = 10, color = 'coral', figsize = (10, 5))

# plt.title("HIstogram")
# plt.xlabel('mpg')
# plt.show()

# #######

# import pandas as pd
# import matplotlib.pyplot as plt

# plt.style.use('default')

# df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/pandas/data/part4/auto-mpg.csv', header = None)

# df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

# df.plot(kind = 'scatter', x='weight', y = 'mpg', c = 'coral', s=10, figsize = (10,5))
# plt.title('Scatter Plot - mpg vs weight')
# plt.show()
# #########
# df.plot(kind='scatter', x= 'mpg', y='weight', c='blue', s = 5, figsize=(10, 5))
# ##############
# cylinders_size = df.cylinders/df.cylinders.max() *300

# # 3개의 변수로 산점도 그리기
# df.plot(kind = 'scatter', x = 'weight', y='mpg', c='coral', figsize=(10, 5), s = cylinders_size, alpha = 0.3)
# plt.title('Scatter Plot : mpg-weight-cylinders')
# plt.show()
# ###################
# cylinders_size = df.cylinders/df.cylinders.max() * 300

# # 3개의 변수로 산점도 그리기
# df.plot(kind = 'scatter', x = 'weight', y='mpg', marker = '+', figsize=(10,5),
#         cmap = 'viridis', c = cylinders_size, s = 50, alpha = 0.3)
        
# plt.title('Scatter Plot : mpg-weight-cylinders')
# plt.show()

# #################

# import pandas as pd
# import matplotlib.pyplot as plt

# plt.style.use('default')

# df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/pandas/data/part4/auto-mpg.csv', header = None)

# df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

# df['count'] = 1
# df_origin = df.groupby('origin').sum()
# print(df_origin)

# df_origin.index = ['USA', 'EU', 'JPN']

# df_origin['count'].plot(kind='pie',
#                         figsize=(7,5),
#                         autopct='%1.9f%%',
#                         startangle = 10,
#                         colors = ['chocolate', 'bisque', 'cadetblue']
#                         )

# plt.title('Model Origin', size=20)
# plt.axis('equal')
# plt.legend(labels=df_origin.index, loc='upper right')
# plt.show()


# df_origin['count'].plot(kind = 'pie',
#                         figsize=(7,5),
#                         autopct = '%1.1f%%',
#                         startangle = 10,
#                         colors = ['chocolate', 'bisque', 'cadetblue']
#                         )

# ###################

# import pandas as pd
# import matplotlib.pyplot as plt

# from matplotlib import rc
# rc('font', family = 'AppleGothic')

# plt.style.use('seaborn-poster')
# plt.rcParams['axes.unicode_minus']=False
# df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/pandas/data/part4/auto-mpg.csv', header = None)

# df.columns = ['mpg', 'cylinders', 'difplacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

# fig = plt.figure(figsize = (15, 5))
# ax1 = fig.add_subplot(1,2,1)
# ax2 = fig.add_subplot(1,2,2)

# ax1.boxplot(x=[df[df['origin']==1]['mpg'],
#                df[df['origin']==2]['mpg'],
#                df[df['origin']==3]['mpg']],
#             labels = ['USA', 'EU', 'JPN'])
# ax2.boxplot(x=[df[df['origin']==1]['mpg'],
#                df[df['origin']==2]['mpg'],
#                df[df['origin']==3]['mpg']],
#             labels = ['USA', 'EU', 'JPN'],
#             vert = False)

# ax1.set_title('제조국가별 연비 분포(수직 박스 플롯)')
# ax2.set_title('제조국가별 연비 분포(수평 박스 플롯)')

# plt.show()

# ##############

# import seaborn as sns

# titanic = sns.load_dataset('titanic')

# print(titanic.head())

# titanic.info()


# import matplotlib.pyplot as plt
# import seaborn as sns

# titanic = sns.load_dataset('titanic')

# # 스타일 테마 설정 (5가지 : darkgrid, whitegrid, dark, white, ticks)
# sns.set_style('darkgrid')

# # 그래프 객체 생성 (figure에 2개의 서브 플롯 생성)
# fig = plt.figure(figsize=(15, 5))
# ax1 = fig.add_subplot(1,2,1)
# ax2 = fig.add_subplot(1,2,2)

# # 그래프 그리기 - 선형회귀선 표시(fit_reg=True)
# sns.regplot(x = 'age', 
#             y = 'fare',
#             data = titanic,
#             ax = ax1
#             )
# sns.regplot(x='age',
#             y='fare',
#             data = titanic,
#             ax = ax2, 
#             color = 'coral',
#             fit_reg = False)

# plt.show()


# fig = plt.figure(figsize = (15,5))
# ax1 = fig.add_subplot(1,3,1)
# ax2 = fig.add_subplot(1,3,2)
# ax3 = fig.add_subplot(1,3,3)

# sns.distplot(titanic['fare'], ax = ax1)

# sns.distplot(titanic['fare'], hist = False, ax=ax2)

# sns.distplot(titanic['fare'], kde=False, ax = ax3)

# ax1.set_title('titanic fare - hist/ked')
# ax2.set_title('titanic fare - ked')
# ax3.set_title('titanic fare - hist')

# plt.show()

# ######################

# table = titanic.pivot_table(index=['sex'], columns = ['class'], aggfunc = 'size')

# sns.heatmap(table,
#             annot = True, fmt = 'd',
#             cmap = 'YlGnBu',
#             linewidth = .5,
#             cbar = False)

# plt.show()
# #########


# import pandas as pd
# import numpy as np
# import seaborn as sns

# titanic = sns.load_dataset('titanic')

# sns.set_style('whitegrid')

# fig = plt.figure(figsize=(15,5))
# ax1 = fig.add_subplot(1,2,1)
# ax2 = fig.add_subplot(1,2,2)

# sns.stripplot(x = 'class',
#               y = 'age',
#               data = titanic,
#               ax = ax1)

# sns.swarmplot(x = 'class',
#               y = 'age',
#               data = titanic,
#               ax = ax2)

# ax1.set_title('Strip Plot')
# ax2.set_title('Strip Plot')

# plt.show()

# ##########

# fig = plt.figure(figsize = (15,5))
# ax1 = fig.add_subplot(1,3,1)
# ax2 = fig.add_subplot(1,3,2)
# ax3 = fig.add_subplot(1,3,3)

# sns.barplot(x= 'sex', y='survived', data = titanic, ax = ax1)

# sns.barplot(x= 'sex', y='survived', hue='class', data=titanic, ax=ax2)

# sns.barplot(x= 'sex', y='survived', hue = 'class', dodge = False, data = titanic, ax = ax3)

# ax1.set_title('titanic survived - sex')
# ax2.set_title('titanic survived - sex/class')
# ax3.set_title('titanic survived - sex/class(stacked)')

# plt.show()

# ################


# fig = plt.figure(figsize = (15,5))
# ax1 = fig.add_subplot(1,3,1)
# ax2 = fig.add_subplot(1,3,2)
# ax3 = fig.add_subplot(1,3,3)

# sns.countplot(x='class', palette='Set1', data = titanic, ax=ax1)

# sns.countplot(x='class', hue = 'who', palette = 'Set2', data = titanic, ax = ax2)

# sns.countplot(x='class', hue = 'who', palette = 'Set3', dodge = False, data = titanic, ax = ax3)

# ax1.set_title('titanic class')
# ax2.set_title('titanic class')
# ax3.set_title('titanic class')

# ###############################

# import pandas as pd
# import seaborn as sns

# titanic = sns.load_dataset('titanic')

# sns.set_style('whitegrid')
# s
# fig = plt.figure(figsize = (15,10))
# ax1 = fig.add_subplot(2,2,1)
# ax2 = fig.add_subplot(2,2,2)
# ax3 = fig.add_subplot(2,2,3)
# ax4 = fig.add_subplot(2,2,4)

# sns.boxplot(x = 'alive', y ='age', data=titanic, ax=ax1)
# sns.boxplot(x='alive', y='age', data=titanic, ax=ax2)
# sns.violinplot(x='alive', y='age', data=titanic, ax=ax3)
# sns.violinplot(x='alive', y='age',hue='sex', data = titanic, ax=ax4)

# plt.show()

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# titanic = sns.load_dataset('titanic')

# sns.set_style('whitegrid')

# j1 = sns.jointplot(x='fare', y='age', data=titanic)

# j2 = sns.jointplot(x='fare', y='age', kind='reg', data = titanic)

# j3 = sns.jointplot(x='fare', y='age', kind='hex', data = titanic)

# j4 = sns.jointplot(x='fare', y='age', kind='kde', data = titanic)

# j1.fig.suptitle('titanic fare - scatter', size=15)
# j2.fig.suptitle('titanic fare - reg', size=15)
# j3.fig.suptitle('titanic fare - hex', size=15)
# j4.fig.suptitle('titanic fare - ked', size=15)

# plt.show()

# ##########

# import matplotlib.pyplot as plt
# import seaborn as sns

# titanic = sns.load_dataset('titanic')

# sns.set_style('whitegrid')

# g = sns.FacetGrid(data = titanic, col = 'who', row = 'survived')
# g = g.map(plt.hist,'age')

# titanic_pair = titanic[['age', 'pclass', 'fare']]

# g = sns.pairplot(titanic_pair)


# import folium

# seoul_map = folium.Map(location = [37.55, 126.98], zoom_start=12)

# seoul_map.save('./seoul.html')


# seoul_map2 = folium.Map(location=[37.66, 126.98], tiles = 'Stamen Terrain', zoom_start = 12)
# seoul_map3 = folium.Map(location=[37.66, 126.98], tiles = 'Stamen Toner', zoom_start = 15)

# seoul_map2.save('./seoul2.html')
# seoul_map3.save('./seoul3.html')

# import pandas as pd
# import folium

# # 대학교 리스트를 데이터프레임으로 변환
# df = pd.read_excel('/Users/heechankang/projects/pythonworkspace/pandas/data/part4/서울지역 대학교 위치.xlsx', index_col=0)

# # 서울 지도 만들기
# seoul_map = folium.Map(location = [37.5, 127], tiles = 'Stamen Terrain', zoom_start=12)

# for name, lat,lng in zip(df.index, df.위도, df.경도):
#         folium.Marker([lat, lng], popup=name).add_to(seoul_map)

# seoul_map.save('./seoul_colleges.html')


# for name, lat, lng in zip(df.index, df.위도, df.경도):
#         folium.CircleMarker([lat, lng],
#                              radius = 10,
#                              color = 'brown',
#                              fill = True,
#                              fill_color = 'coral',
#                              fill_opacity = 0.7,
#                              popup = name).add_to(seoul_map)

# seoul_map.save('./seoul_colleges2.html')

# import pandas as pd
# import folium
# import json

# # 경기도 인구변화 데이터를 불러와서 데이터프레임으로 변환
# file_path = '/Users/heechankang/projects/pythonworkspace/pandas/data/part4/경기도인구데이터.xlsx'
# df = pd.read_excel(file_path, index_col = '구분')
# df.columns = df.columns.map(str)

# # 경기도 시군구 경계 정보를 가진 geo-json 파일 불러오기
# geo_path = '/Users/heechankang/projects/pythonworkspace/pandas/data/part4/경기도행정구역경계.json'
# try:
#         geo_data = json.load(open(geo_path, encoding='utf-8'))
# except:
#         geo_data = json.load(open(geo_path, encoding='utf-8-sig'))

# # 경기도 지도 만들기
# g_map = folium.Map(location = [37.5, 127], tiles = 'Stamen Terrain', zoom_start=9)

# # 출력할 연도 선택
# year = '2017'

# # Chropleth 클래스로 단계 구분도 표시하기
# folium.Choropleth(geo_data = geo_data,
#                   data = df[year],
#                   columns = [df.index, df[year]],
#                   fill_color = 'YlOrRd', fill_opacity = 0.7, line_opacity = 0.3,
#                   threshold_scale = [10000, 100000, 300000, 500000, 700000],
#                   key_on = 'feature.properties.name',).add_to(g_map)

# # 지도를 HTML 파일로 저장하기
# g_map.save('gyonggi_population_' + year + '.html')

import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age', 'sex', 'class', 'fare', 'survived']]

grouped = df.groupby(['class'])
titanic.head()
grouped.head()

grouped_age = df.groupby(['age'])


grouped.age.mean()

age_std = grouped.age.std()
age_std
age_mean = grouped.age.mean()
age_mean

df.groupby('class').sum()

for k, g in grouped.age:
        group_zscore = (g - grouped.age.mean().loc[k]) / grouped.age.std().loc[k]
        print(k)
        print(group_zscore)

# 혹은
for k, g in grouped.age:
        group_zscore = (g - age_mean.loc[k]) / age_std.loc[k]
        print(k)
        print(group_zscore)

def z_score(x):
        return (x-x.mean())/x.std()

grouped.age.transform(z_score)
transformed_age = grouped.age.transform(z_score)

transformed_age.loc[[1,9,0]]


######

for key, group in grouped.age:
        group_zscore = (group - grouped.age.mean().loc[key]) / grouped.age.std().loc[key]
        print(key)
        print(group_zscore)

transformed_age.loc[0:9]

grouped.get_group('First')

grouped.get_group('Second')
grouped.filter(lambda x: len(x) >= 0)
grouped.filter(lambda x: len(x) >= 200)
grouped.filter(lambda x: x.age.mean() <30)

def z_score(x):
        return( x - x.mean()) / x.std()

age_zscore = grouped.age.transform(z_score)
print(age_zscore.loc[[1,9,0]])
print()
print(len(age_zscore))
print(age_zscore.loc[0:9])
print(type(age_zscore))
age_zscore

grouped_filter = grouped.filter(lambda x: len(x) >=200)
grouped_filter
grouped.filter(lambda x: len(x) <200)


agg_grouped = grouped.apply(lambda x: x.describe())
print(agg_grouped)


def z_score(x):
        return(x-x.mean()) / x.std()

age_zscore = grouped.age.apply(z_score)
print(age_zscore.head())

age_filter = grouped.apply(lambda x: x.age.mean() < 30)
print(age_filter)
print()
for x in age_filter.index:
        if age_filter[x] == True:
                age_filter_df = grouped.get_group(x)
                print(age_filter_df.head())
                print()

titanic = sns.load_dataset('titanic')
df = titanic.loc[:,['age', 'sex', 'class', 'fare', 'survived']]
grouped = df.groupby('class')
grouped

grouped.apply(lambda x : x.describe())

def z_score(x):
        return (x-x.mean()) / x.std()

grouped.age.apply(z_score)

age_filter = grouped.apply(lambda x:x.age.mean() < 30)
print( age_filter)
age_filter.index

for x in age_filter.index:
        if age_filter[x] ==True:
                age_filter_df = grouped.get_group(x)
                print(age_filter_df.head())
                print()


grouped2 = df.groupby(['class', 'sex'])
grouped2

grouped3 = grouped2.mean()

grouped2.mean().loc['First']
grouped2.mean().iloc[0:2]


gdf = grouped.mean()
gdf

grouped3.xs('First', level='class')

titanic
df

group = df.groupby(['class', 'sex'])
gdf = group.mean()
gdf
gdf.xs('Second', level='class')


#####

import pandas as pd
import seaborn as sns

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 20)

titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age', 'sex', 'class','fare', 'survived']]

df.head()


pdf1 = pd.pivot_table(df,
                      index = 'class',
                      columns = 'sex',
                      values = 'age',
                      aggfunc = 'mean')

pdf1

pdf2 = pd.pivot_table(df,
                      index = 'class',
                      columns = 'sex',
                      values = 'survived',
                      aggfunc = ['count', 'sum'])

pdf2

pdf3 = pd.pivot_table(df,
                      index = ['class', 'sex'],
                      columns = 'survived',
                      values = ['age', 'fare'],
                      aggfunc = ['mean', 'max'])

pdf3

pdf3.xs('First')

pdf3.xs(('First', 'male'))

pdf3.xs('male', level='sex')

pdf3.xs('male', level='sex')

pdf3.columns
pdf3.index

pdf3.xs('male', level='sex')

pdf3.xs(('Second', 'male'), level=(0, 'sex'))

pdf3

pdf3.xs(1, level='survived', axis=1)
pdf3.xs('age',level = 1, axis = 1)
pdf3.xs('male', level=1)
pdf3.xs('First', level=0)