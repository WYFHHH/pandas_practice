# # import pandas as pd


# # # 예제 1-1 딕셔너리 -> 시리즈 변환

# # dict_data = {'a':1, 'b': 2, 'c' : 3}

# # sr = pd.Series(dict_data)

# # print(type(sr))
# # print('\n')
# # print(sr)


# # # 예제 1-2 시리즈 인덱스

# # list_data = ['2019-01-02', 3.14, 'ABC', 100, True]
# # sr = pd.Series(list_data)
# # print(sr)


# # # 예제 1-3 시리즈 원소 선택

# # tup_data = ('영인', '2010-05-01', '여', True)
# # sr = pd.Series(tup_data, index = ['이름', '생년월일', '성별', '학생여부'])
# # print(sr)

# # print(sr[0])
# # print(sr['이름'])

# # print(sr[[1, 2]])
# # print()
# # print(sr[['생년월일', '성별']])

# # print(sr[1:2])
# # print()
# # print(sr['생년월일':'성별'])


# # # 예제 1-4 딕셔너리 -> 데이터프레임 전환

# # dict_data = {'c0' : [1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], 'c3':[10,11,12], 'c4':[13,14,15]}

# # df = pd.DataFrame(dict_data)
# # print(type(df))
# # print()
# # print(df)


# # # 예제 1-5 행 인덱스 / 열 이름 설정

# # df = pd.DataFrame(
# #     [[15, '남', '덕영중'], [17, '여', '수리증']],
# #     index = ['준서', '예은'],
# #     columns = ['나이', '성별', '학교']
# # )

# # print(df)
# # print()
# # print(df.index)
# # print()
# # print(df.columns)

# # df.index = ['학생1', '학생2']
# # df.columns = ['연력', '남녀', '소속']

# # print(df)
# # print()
# # print(df.index)
# # print()
# # print(df.columns)


# # # 예제 1-6 행 인덱스 / 열 이름변경

# # df = pd.DataFrame(
# #     [[15, '남', '덕영중'], [17, '여', '수리중']],
# #     index = ['준서', '예은'],
# #     columns=['나이', '성별', '학교']
# # )

# # print(df)
# # print()

# # df.rename(columns={'나이':'연령', '성별':'남녀', '학교':'소속'}, inplace = True)

# # df.rename(index={'준서': '학생1', '예은':'학생2'}, inplace=True)

# # print(df)



# # # 예제 1-7 행 삭제

# # exam_data = {'수학':[90, 80, 70],'영어':[98, 89,95], '음악' : [85, 95, 100], '체육' : [100, 90, 90]}

# # df = pd.DataFrame(exam_data, index = ['서준', '우현', '인아'])
# # print(df)
# # print()

# # df2 = df[:]
# # df2.drop('우현', inplace = True)
# # print(df2)
# # print()

# # df3 = df[:]
# # #df3.drop(['우현', '인아'], axis=0, inplace = True)
# # df3 = df3.drop(['우현', '인아'])
# # print(df3)


# # # 예제 1-8 열 삭제

# # df4 = df[:]
# # df5 = df[:]

# # df4.drop('수학', axis = 1, inplace = True)
# # print(df4)
# # print()

# # df5.drop(['영어','음악'], axis = 1, inplace=True)
# # print(df5)
# # print()


# # # 예제  1-9 행 선택

# # df = pd.DataFrame(exam_data, index = ['서준', '우현', '인아'])
# # print()
# # print(df)

# # label1 = df.loc['서준']
# # position1 = df.iloc[0]
# # print(label1)
# # print()
# # print(position1)


# # label2 = df.loc[['서준', '우현']]
# # position2 = df.iloc[[0, 1]]
# # print(label2)
# # print()
# # print(position2)

# # label3 = df.loc['서준':'우현']
# # position3 = df.iloc[0:1]
# # print(label3)
# # print()
# # print(position3)



# # #--------------------------------

# # #21.04.11(일) Pandas 자습

# # import numpy as np
# # import pandas as pd
# # import random

# # df = pd.DataFrame(np.random.rand(5,2), columns=['A', "B"])
# # print(df)
# # print(df['A']<0.5)

# # print(df[(df["A"] <0.5) & (df["B"] > 0.3)])
# # df.query("A < 0.5 and B >0.3")
# # df.query("A>0.9 and B<0.1")

# # df = pd.DataFrame(columns = ["Animal", 'Name'])
# # print(df)
# # df["Animal"] = ['Dog', 'Cat', 'Cat', 'Pig', 'Cat']
# # df["Name"] = ['Happy', 'Sam', 'Toby', 'Mini', 'Rocky']
# # print(df)

# # df["Animal"].str.contains("at")
# # df.Animal.str.match("Ca")

# # df = pd.DataFrame(np.arange(5), columns = ["Num"])
# # def square(x):
# #     return x**2
# # df["Num"].apply(square)
# # df["Square"] = df.Num.apply(lambda x:x**2)
# # print(df)

# # df = pd.DataFrame(columns = ["phone"])
# # df.loc[0] = '010-1234-1235'
# # df.loc[1] = '공일공-일이삼사-1235'
# # df.loc[2] = '010.1234.일이삼오'
# # df.loc[3] = '공1공-1234.1이3오'
# # df['preprocess_phone'] = ''
# # print(df)

# # def get_preprocessed_phone(phone):
# #     mapping_dict = {
# #         '공' : '0',
# #         '일' : '1',
# #         '이' : '2',
# #         '삼' : '3',
# #         '사' : '4', 
# #         '오' : '5', 
# #         '육' : '6',
# #         '-' : '',
# #         '.' : '',
# #     }
# #     for key, value in mapping_dict.items():
# #         phone = phone.replace(key, value)
# #     return phone

# # df['preprocessed_phone'] = df['phone'].apply(get_preprocessed_phone)
# # print(df)


# # df = pd.DataFrame(columns = ["sex"])
# # df['sex'] = ['Male', 'Male', 'Female', 'Female', 'Male']
# # print(df)

# # df['sex'].replace({'Male' : 0, 'Female' : 1})
# # print(df)
# # df.sex.replace({'Male' : 0, 'Female':1}, inplace = True)
# # print(df)

# # df = pd.DataFrame({'key' : ['A', "B", "C", "A", "B", "C"], 'data' : range(6)})
# # print(df.groupby('key'))
# # df.groupby('key').sum()
# # df.groupby('key', 'data').sum()
# # print(df)
# # df["data2"] = [4,4,6,0,6,1]
# # print(df)

# # df.groupby(['key', 'data']).sum()
# # df.groupby('key').sum()

# # df.groupby('key').aggregate(['min', np.median, max])
# # df.groupby('key').aggregate({'data': min, 'data2' : np.sum})

# # #################################
# # # '21. 04. 12(월)
# # import numpy as np
# # import pandas as pd
# # import random

# # df = pd.DataFrame(columns = ['key', 'data1' ,'data2'])
# # df.key = ['A', "B", "C", 'A', "B", "C"]
# # df['data1'] = range(6)
# # df['data2'] = [4,4,6,0,6,1]

# # df.data1 = [1,2,1,1,2,2]

# # print(df)

# # df.groupby('key').sum()
# # df.groupby(['key', 'data1']).sum()

# # df.groupby('key').aggregate([{'data1': min, 'data2': max}])

# # def filter_by_mean(x):
# #     return x['data2'].mean() >3

# # df.groupby('key').mean()
# # df.groupby('key').filter(filter_by_mean)

# # df.groupby('key').get_group('A')

# # df.groupby('key').apply(lambda x : x.max() - x.min())


# ##########################
# #'21.04.12(월)

# # 예제 1-10

# import numpy as np
# import pandas as pd

# # DataFrame() 함수로 데이터프레임 변환, 변수 df에 저장

# exam_data = {'이름':['서준', '우현', '인아'],
#              '수학':[90, 80, 70],
#              '영어':[98, 89, 95],
#              '음악':[85, 95, 100],
#              '체육':[100, 90, 90]}

# df = pd.DataFrame(exam_data)
# print(df)

# print(type(df))
# print()

# # 수학 점수만 선택, 변수 math1에 저장
# math1 = df['수학']
# print(math1)
# print(type(math1))
# print()

# # 영어 점수만 선택, english에 저장
# english = df['영어']
# print(english)
# print(type(english))

# music_gym = df[['음악' ,'체육']]
# print(music_gym)
# print(type(music_gym))

# math2 = df['수학']
# print(type(math2))
# print(math2)

# print(df)

# df.iloc[::2]
# df.iloc[:3:2]
# df.iloc[::-1]

# ########################
# # 예제 1-11

# import pandas as pd

# # DataFrame() 함수로 데이터프레임 변환, 변수 df 에 저장

# exam_data
# df = pd.DataFrame(exam_data)
# print(df)

# df.set_index('이름', inplace =True)
# print(df)

# a=df.loc['서준', '음악']
# print(a)
# b = df.iloc[0, 2]
# print(b)

# # 데이터프레임 df의 특정 원소 2개 이상 선택('서준'의 '음악, '체육' 점수)
# c = df.loc['서준', ['음악', '체육']]
# print(c)

# d = df.iloc[0, 2:3]
# print(d)

# e = df.loc['서준', '음악':'체육']
# print(e)

# ########################
# #잠시 연습
# # df = pd.DataFrame({'A' : range(6)})
# # print(df)
# # df['square'] = df['A'].apply(lambda x : x**2)
# # print(df)
# ########################

# # 2개 이상의 행과 열에 속하는 원소들 선택('서준', '우현'의 '음악', '체육' 점수)

# print(df)
# df.set_index('이름', inplace=True)
# print(df)
# g = df.loc[['서준', '우현'], ['음악','체육']]
# print(g)

# h = df.iloc[[0,1], [2,3]]
# print(h)

# i = df.loc['서준':'우현', '음악':'체육']
# print(i)

# j = df.iloc[0:2, 2:4]
# print(j)

# #################################
# # 연습문제 1-12

# # #################################

# import numpy as np  
# import pandas as pd


# exam_data = {
#     '수학' :[90, 80, 70], '영어':[98, 89, 95],
#     '음악':[85, 95, 100], '체육':[100, 90, 90]
#              }
# df = pd.DataFrame(exam_data, index=['서준', '우현', '인아'])
# print(df)
# print()

# df2 = df[:]
# print(df2)

# ###################

# import pandas as pd

# exam_data = {
#     '이름' : ['서준', '우현', '인아'],
#     '수학' : [90, 80, 70],
#     '영어' : [98, 89, 95],
#     '음악' : [85, 95, 100],
#     '체육' : [100, 90, 90],
# }

# df_ori = pd.DataFrame(exam_data)

# df['국어'] = 80
# df.loc[3] = 0
# df2 = df.iloc[:,:5]x
# df2
# df2.loc[4] = ['동규', 90, 80, 70, 60]
# df2.loc['행5'] = df.loc[3]
# df2.iloc[3,0] = '현준'
# df2.iloc[5,0] = '민환'
# df2.set_index('이름', inplace = True)
# df2.loc['현준'] = [50, 60, 70, 80]
# df2.loc['민환'] = [55, 66, 77, 88]
# df2.loc['동규']['체육'] = 0
# df2.loc['우현', '수학'] = 0
# df2

# df2.loc['서준', ['음악', '체육']] = 7
# df2.loc['우현':'동규', '음악'] = 9
# df2.loc['인아':'민환', '영어':'체육'] = 11
# df2.iloc[1:5, 1:3] = 99
# df2.iloc[:,1] = [90,80,70,60,50,40]
# df2

# exam_data = {
#     '이름' : ['서준', '우현', '인아'],
#     '수학' : [90, 80, 70],
#     '영어' : [98, 89, 95],
#     '음악' : [85, 95, 100],
#     '체육' : [100, 90, 90],
# }
# df = pd.DataFrame(exam_data)
# df = df.set_index('이름')
# df3 = df
# df3.drop('체육', axis=1, inplace = True)
# df3
# df.set_index('이름', inplace=True)
# # df2 = df[:]
# # df2.transpose()
# # df2 = df2.transpose()
# # df2


# import pandas as pd

# dict_data = {
#     'c0': [1,2,3,],
#     'c1': [4,5,6],
#     'c2': [7,8,9], 
#     'c3': [10,11,12],
#     'c4': [13,14,15],
# }

# df_ori = pd.DataFrame(dict_data, index = ['r1' , 'r2', 'r3'])
# df_ori

# df = df_ori[:]
# new_index = ['r1' ,'r2', 'r3', 'r4', 'r5']

# ndf2 = df.reindex(new_index, fill_value = 0)
# ndf2


# df = df.reset_index()
# df.set_index('index', inplace = True)
# df

# df.sort_index(ascending=False)


# import pandas as pd


# dict_data = {
#     'c0': [1,2,3,],
#     'c1': [4,5,6],
#     'c2': [7,8,9], 
#     'c3': [10,11,12],
#     'c4': [13,14,15],
# }

# df = pd.DataFrame(dict_data, index = ['r2', 'r1' ,'r0'])
# print(df)
# df.sort_values(by='c1', ascending=False)



# import pandas as pd
# student1 = pd.Series(
#     {
#         '국어':100,
#         '영어':90,
#         '수학':80,
#     }
# )
# print(student1)

# percentage = student1/200

# print(percentage)

# print(type(percentage))


# import pandas as pd

# student1 = pd.Series(
#     {
#         '국어': 100,
#         '영어': 80,
#         '수학':90,
#     }
# )
# student2 = pd.Series(
#     {
#         '수학':80,
#         '국어':90,
#         '영어':80,
#     },
# )
# print(student1)
# print(student2)

# addition = student1 + student2
# subtraction = student1 -   student2
# multiplication = student1 * student2
# division = student1 / student2
# print(type(division))
# print()

# result = pd.DataFrame([addition, subtraction, multiplication, division], index=['덧셈', '뺄셈', '곱셈', '나눗셈'])
# result


# sr1 = pd.Series(
#     {
#         'a':10,
#         'b':20,
#         'c':30,
#         'd':40
#     }
# )
# sr2 = pd.Series(
#     {
#         'b':5,
#         'c':15,
#         'd':25,
#         'e':35,
#     }
# )

# add = sr1+sr2
# sub = sr1-sr2
# result = pd.DataFrame([add, sub], index = ['덧셈', '뺄셈'])

# result

# import numpy as np

# student1 = pd.Series(
#     {
#         '국어':np.nan,
#         '영어':80,
#         '수학':90
#     }
# )
# student2 = pd.Series(
#     {
#         '수학':80,
#         '국어':90,
#     }
# )

# print(student1)
# print(student2)

# addition = student1 + student2
# subtraction = student1 - student2
# multiplication = student1* student2
# division = student1 / student2

# print(type(division))

# result = pd.DataFrame([addition, subtraction, multiplication, division], index = ['덧셈', '뺄셈', '곱셈','나눗셈'])

# result

# ####################

# import pandas as pd
# import seaborn as sns
# import numpy as np

# titanic = sns.load_dataset('titanic')
# df = titianic.loc[:, ['age', 'fare']]
# # print(df.head())

# import pandas as pd
# import numpy as np

# exam_data = {
#     '이름' : ['서준' ,'우현', '인아'],
#     '수학' : [90, 80, 70],
#     '영어' : [98, 89, 95],
#     '음악' : [85, 95, 100],
#     '체육' : [100, 90, 90]
# }
# df = pd.DataFrame(exam_data)
# df.set_index('이름')
# df = df.set_index('이름')
# df

# a = df.loc['서준', '음악']
# print(a)
# b = df.iloc[0, 2]
# b

# c = df.loc['서준', ['음악', '체육']]
# c

# import pandas as pd
# import seaborn as sns

# df=pd.read_csv('pandas/data/titanic.csv')
# df
# df1 = df.loc[:,['Age', 'Fare']]
# print(df1.head())

# print()
# print(type(df1))
# print()

# addition = df1 + 10
# print(addition.head())
# print()
# print(type(addition))

# df1.T.T.T.T

###########

# import pandas as pd
# import seaborn as sns

# titanic = sns.load_dataset('titanic')
# df = titanic.loc[:, ['age', 'fare']]
# print(df.tail())

# print()
# print(type(df))
# print()

# addition = df + 10
# print(addition.tail())
# print()
# print(type(addition))
# print()

# subtraction = addition - df
# print( subtraction.tail())

# import pandas as pd
# import numpy as np

# df = pd.DataFrame(
#     {
#         'c0':[1,2,3],
#         'c1':[4,5,6],
#         'c2':[7,8,9],
#         'c3':[10,11,12],
#         'c4':[13,14,15],
#     }, index=['r0', 'r1', 'r2']
# )
# new_index = ['r0','r1', 'r2', 'r3', 'r4']
# df2 = df.reindex(new_index)
# df3 = df.reindex(new_index, fill_value=0)
# dd = df3.reset_index

# import seaborn as sns

# titanic = sns.load_dataset('titanic')
# print(titanic.head())

# df = pd.DataFrame(titanic)
# print(df.head())

# df.sort_values(by='age', ascending=True)
# df.sort_values(by='fare', ascending=True)
# df_cal=df[:]
# df_cal.dropna()
# df_cal = df_cal.dropna()
# df_cal.sort_values(by='fare', ascending=True)
# df_cal.sort_index(ascending=1)



# anscombe = sns.load_dataset('anscombe')
# anscombe

# attention = sns.load_dataset('attention')
# attention

# brain_networks = sns.load_dataset('brain_networks')
# brain_networks

# student1 = pd.Series({
#     '국어':100,
#     '영어':80,
#     '수학':90,
# })

# percentage = student1 / 200

# print(percentage)

# car_crashes = sns.load_dataset('car_crashes')
# car_crashes

# diamonds = sns.load_dataset('diamonds')
# diamonds

# dots = sns.load_dataset('dots')
# dots

# exercise = sns.load_dataset('exercise')
# exercise

# flights = sns.load_dataset('flights')
# flights

# fmri = sns.load_dataset('fmri')
# fmri

# gammas = sns.load_dataset('gammas')
# gammas

# iris = sns.load_dataset('iris')
# iris

# mpg = sns.load_dataset('mpg')
# mpg

# planets = sns.load_dataset('planets')
# planets

# tips = sns.load_dataset('tips')
# tips

# student1 = pd.Series({
#     '국어':100,
#     '영어':80,
#     '수학':90
# })
# student2 = pd.Series({
#     '수학':80,
#     '국어':90,
#     '과학':80,
# })

# print(student1)
# print()
# print(student2)

# sr_add = student1.add(student2, fill_value=0)

# sr_add

# df = pd.DataFrame(columns = ['a', 'b', 'c', 'd', 'e'])
# df.loc[0] = [1,2,3,4,5]
# df.loc[1] = [6,7,8,9,0]
# df.loc[2] = [1,3,5,7,9]
# df.loc[3] = [2,4,6,8,0]
# n_index = ['가', '나', '다', '라']
# df['name'] = n_index
# df = df.set_index('name')
# df.loc['나':'라', 'c']

# import seaborn as sns
# import numpy as np
# import pandas as pd

# titanic = sns.load_dataset('titanic')
# df = pd.DataFrame(titanic)
# df = df.loc[:, ['age', 'fare']]
# print(df.head())
# addition = df.add(10, fill_value = 0)
# print(addition)

# cal = addition.sub(df, fill_value=0)
# print(cal)

# import random

# dff = pd.DataFrame(columns = ['age', 'fare'])
# dff

# dff['age'] = random.random()
# dff

###############################
# 판다스 파트3

# import pandas as pd

# df = pd.read_csv('pandas/data/5674-833/part3/auto-mpg.csv', header=None)
# print(df.head())

# df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight','acceleration', 'model year', 'origin', 'name']

# print(df.head())
# print()
# print(df.tail())
# print(df.shape)
# print(df.info())

# print(df.dtypes)
# print(df.mpg.dtypes)

# print(df.describe(include='all'))

#######################################

# import pandas as pd

# df = pd.read_csv('pandas/data/5674-833/part3/auto-mpg.csv', header=None)

# df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight','acceleration', 'model year', 'origin', 'name']

# print(df.count())

# unique_values = df['origin'].value_counts()

# print(unique_values)

# unique_values = df['model year'].value_counts()
# print(unique_values)

# unique_values = df['cylinders'].value_counts()
# print(unique_values)
# print(type(unique_values))

# print(df.mean())
# print()
# print(df['mpg'].mean())
# print(df.mpg.mean())
# print(df[['mpg', 'weight']].mean())

# print(df.median())

# import seaborn as sns
# import matplotlib.pyplot as plt


# # Scatterplot 그리기
# # fig, ax = plt.subplots() #시각화 테이블을 불러오기
# # ax.scatter(titanic['age'], titanic['fare']) # 나이와 요금 선택
# # ax.set_title('Ticket_Price_Based_On_Age') # 제목설정
# # ax.set_xlabel('Age')
# # ax.set_ylabel('Fare')

# sns.boxplot(y = 'mpg', data = df)

# fig, ax = plt.subplots()
# ax.scatter(df.mpg, df.weight)

# fig, ax = plt.subplots()
# ax.hist(df.weight)

# print(df.median())
# print(df.mpg.median())

# print(df.max())

# print(df['mpg'].max())

# print(df.min())
# print(df.info())
# print(df.describe())

# print(df.std())
# print(df['mpg'].std())
# print(df.mpg.std())

# print(df.corr())

# print(df.acceleration.max())
# df.sort_values('acceleration')

########################################################

# import pandas as pd

# df = pd.read_excel('./data/5674-833/part3/남북한발전전력량.xls')

# print(df.head())

# df_ns = df.iloc[[0, 5], 3:]
# df_ns.index = ['South', 'North']
# df_ns.columns= df_ns.columns.map(int)
# print(df_ns.head())

# df_ns.plot()#에바임

# tdf_ns = df_ns.T
# tdf_ns
# print(tdf_ns.head())
# print()
# tdf_ns.plot()

# tdf_ns.plot(kind="bar")

# df_ns.plot(kind="bar") #에바임

# ##################################
# import pandas as pd
# df = pd.read_excel('./data/5674-833/part3/남북한발전전력량.xls')

# print(df.head())

# df = df.iloc[[0, 5], 3:]
# df.head()
# df.index = ['South', 'North']
# df.columns.map(int)
# df.plot()
# tdf = df.T

# tdf.plot(title='남북')

    
# except 
# :
#     pass

# ################################

# import pandas as pd

# df = pd.read_excel('./data/5674-833/part3/남북한발전전력량.xls')
# df

# df_ns = df.iloc[[0, 5], 2:]
# df_ns.index = ['South', 'North']
# df_ns
# df_ns.columns.map(int)
# df_ns

# df_ns.plot()

# tdf_ns = df_ns.T
# tdf_ns.head(n=10)

# tdf_ns.plot(kind="bar")

# #####
# import pandas as pd

# df = pd.read_excel('./data/5674-833/part3/남북한발전전력량.xlsx')
# df.head()
# df_ns = df.iloc[[0, 5], 3:]

# df_ns.head()
# df_ns.index = ['South', 'North']
# df_ns.columns.map(int)
# tdf_ns = df_ns.T
# tdf_ns.plot.barh()
# tdf_ns.plot(kind='barh')

#################

# import pandas as pd

# df = pd.read_excel('./data/5674-833/part3/남북한발전전력량.xls')
# df_ns = df.iloc[[0, 5], 3:]
# df_ns.index = ['South', 'North']
# df_ns.columns.map(int)
# df_ns.plot.hist()
# tdf_ns = df_ns.T
# tdf_ns.plot.hist()

##############

# import pandas as pd

# df = pd.read_excel('./data/5674-833/part3/남북한발전전력량.xls')
# df.head(n=10)

# df_ns = df.iloc[[0, 5], 2:]
# df_ns

# df_ns.index = ['South', 'North']
# df_ns.columns.map(int)
# df_ns

# tdf_ns = df_ns.T

# tdf_ns.plot(kind='hist')

##################################

# import pandas as pd

# df = pd.read_csv('./data/5674-833/part3/auto-mpg.csv')
# df.head()

# df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

# df.plot(x = 'weight', y='mpg', kind='scatter')

########################################

# import pandas as pd

# df = pd.read_csv('./data/5674-833/part3/auto-mpg.csv')

# df.head()

# df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']
# df.head()

# df[['mpg', 'cylinders']].plot(kind='box')





#########################################################
# Part 4 시각화 도구

# import pandas as pd?


##############################
# 아이리스

# import pandas as pd
# import numpy as np
# import seaborn as sns

# iris = sns.load_dataset('iris')

# iris.groupby('species').mean()

# iris = iris.groupby('species').median()

# df = pd.DataFrame(iris)

# df.plot()
# df.plot.bar()


# x = np.where(iris['species'] =='setosa', 'red',
#     np.where(iris['species'] =='versicolor', 'green', 'blue'))

# iris.plot.scatter(x='sepal_length', y='sepal_width', c=x)
# iris.plot.scatter(x='petal_length', y='petal_width', c=x)


###################

# import pandas as pd
# import seaborn as sns

# car = sns.load_dataset('mpg')
# car[['mpg', 'cylinders']].plot.box()
# tcar = car.T
# mcar = car.groupby('cylinders').mean()
# mcar.plot.bar()

###################

# import pandas as pd
# import matplotlib.pyplot as plt

# # 폰트깨짐 해결
# import matplotlib.pyplot as plt
# from matplotlib import rc
# import seaborn as sns
# %matplotlib inline

# rc('font', family='AppleGothic')

# plt.rcParams['axes.unicode_minus'] = False

# df = pd.read_excel('./data/5674-833/part4/시도별 전출입 인구수.xlsx')
# df = df.fillna(method='ffill')

# mask = (df['전출지별'] == '서울특별시') & (df['전입지별'] != '서울특별시')
# df_seoul = df[mask]
# df_seoul = df_seoul.drop(['전출지별'], axis=1)
# df_seoul.rename({'전입지별':'전입지'}, axis = 1, inplace=1)
# df_seoul.set_index('전입지', inplace=True)

# sr_one = df_seoul.loc['경기도']
# sr_one

# plt.plot(sr_one)

# plt.plot(sr_one.index, sr_one.values)
# plt.title('서울->경기 인구이동')


# df
# mask = (df['전출지별']=='서울특별시') & (df['전입지별'] != '서울특별시')
# df_seoul = df[mask]
# df_seoul.rename({'전출지별':'전출지'})
# df_seoul = df_seoul.drop(['전출지별'], axis=1)
# df_seoul.rename({'전입지별':'전입지'},axis = 1, inplace = True)
# df_seoul.set_index('전입지', inplace = True)
# df_seoul
# sr_one = df_seoul.loc['경기도']
# sr_one

# plt.plot(sr_one.index, sr_one.values, c='green')

# df = pd.DataFrame(
#     {
#         'a' : [1,2,3,],
#         'b' : [4,5,6],
#         'c' : ['a' , 'c', 'd']}
#     , index = ['a', 'b', 'c']
# )
# df

# df.rename({'a':'k', 'b':'q'}, axis =0)

# #########################

# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_excel('./data/5674-833/part4/시도별 전출입 인구수.xlsx')
# df.head()
# df = df.fillna(method = 'ffill')
# df.head()

# mask = (df['전출지별'] =='서울특별시') & (df['전입지별'] !='서울특별시')
# df_seoul = df[mask]
# df_seoul = df_seoul.drop(['전출지별'], axis =1)
# df_seoul.reset_index
# df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
# df_seoul.set_index('전입지', inplace = True)
# df_seoul
# sr_one = df_seoul.loc['경기도']
# sr_one

# plt.plot(sr_one.index, sr_one.values)


# plt.figure(figsize=(14, 5))
# plt.xticks(rotation = 'vertical')
# plt.title('서울->경기도 인구이동')
# plt.xlabel('기간')
# plt.ylabel('이동 인구수')

# plt.legend(labels=['서울 -> 경기'], loc = 'best')

# plt.plot(sr_one.index, sr_one.values)

# plt.show()




# sr_one = df_seoul.loc['경기도']

# plt.style.use('ggplot')

# plt.figure(figsize=(14, 5))

# plt.xticks(size=10, rotation = 'vertical')

# plt.plot(sr_one.index, sr_one.values, marker='o', markersize=10)

# plt.title('서울->경기 인구 이동', size=30)
# plt.xlabel('기간', size=20)
# plt.ylabel('이동 인구수', size=20)

# plt.legend(labels =['서울 -> 경기'], loc ='best', fontsize=15)


# plt.ylim(50000, 800000)

# plt.annotate('',
#              xy=(20, 620000),
#              xytext=(2, 290000),
#              xycoords='data',
#              arrowprops=dict(arrowstyle='->', color='skyblue', lw=5)
#              )
# plt.annotate('',
#              xy=(47, 450000),
#              xytext=(30, 580000),
#              xycoords='data',
#              arrowprops=dict(arrowstyle='->', color='olive', lw=5),
#              )
# plt.annotate('인구 이동 증가(1970-1995)',
#              xy=(10, 550000),
#              rotation=25,
#              va='baseline',
#              ha='center',
#              fontsize=15,
#              )
# plt.annotate('인구 이동 감소(1995-2017)',
#              xy=(40, 560000),
#              rotation=-11,
#              va='baseline',
#              ha='center',
#              fontsize=15,
#              )

# plt.show()



################################

# # 판다스 p.117

# # 서울에서 경기도로 이동한 인구 데이터 값만 선택
# sr_one = df_seoul.loc['경기도']
# sr_one

# # 스타일 서식 지정
# plt.style.use('ggplot')

# # 그림 사이즈 지정
# plt.figure(figsize=(14, 5))

# # x축 눈금 라벨 회전하기
# plt.xticks(size = 10, rotation='vertical')

# # x, y축 데이터를 plot 함수에 입력
# plt.plot(sr_one.index, sr_one.values, marker = 'o', markersize=10)

# plt.title('서울 -> 경기 인구 이동', size=30)    # 차트 제목
# plt.xlabel('기간', size = 20)   # x축 이름
# plt.ylabel('이동 인구수', size=20)  #y축 이름

# plt.legend(labels = ['서울', '경기'], loc ='best', fontsize = 15)   # 범례 표시

# # plt.show()   # 변경사항 저장하고 그래프 출력

# #############################

# # 판다스 예제 4-7, p.119

# # y 축 범위지정(최소값, 최대값)
# plt.ylim(50000, 800000)

# # 화살표 모양 주석 생성
# plt.annotate('',    # 멘트 없음
#              xy= (20, 620000), # 화살표 머리끝
#              xytext = (2, 290000), # 화살표 꼬리 끝
#              xycoords='data',
#              arrowprops=dict(arrowstyle='->', color = 'skyblue', lw=5), # 화살표 서식
#             )
# plt.annotate('',    # 멘트 없음
#              xy= (47, 450000),
#              xytext = (30, 580000),
#              xycoords='data',
#              arrowprops=dict(arrowstyle='->', color = 'olive', lw=5), # 화살표 서식
#             )

# # 텍스트 주석 생성
# plt.annotate('인구 이동 증가(1970-1995)',   # 텍스트 입력
#              xy=(10, 550000),            # 텍스트 위치 기준점 
#              rotation=25,                # 텍스트 회전 각도            
#              va='baseline',              # 텍스트 상하 정렬
#              ha='center',                # 텍스트 좌우 정렬
#              fontsize=15,                # 텍스트 크기
#              )
# plt.annotate('인구 이동 감소(1995-2017)',   # 텍스트 입력
#              xy=(40, 560000),            # 텍스트 위치 기준점 
#              rotation=-11,                # 텍스트 회전 각도            
#              va='baseline',              # 텍스트 상하 정렬
#              ha='center',                # 텍스트 좌우 정렬
#              fontsize=15,                # 텍스트 크기
#              )

# # plt.show()

# ##########################################

# # 판다스 예제 4-8, p.121

# # 그래프 객체 생성(figure에 2개의 서브 플롯 생성)
# fig = plt.figure(figsize=(10, 10))
# ax1 = fig.add_subplot(2,1,1)
# ax2 = fig.add_subplot(2,1,2)

# # axe 객체에 plot 함수로 그래프 출력
# ax1.plot(sr_one, 'o', markersize=10)
# ax2.plot(sr_one, marker = 'o', markerfacecolor='green', markersize=10,
#          color='olive', linewidth=2, label='서울->경기')
# ax2.legend(loc='best')

# # y축 범위 지정(최소값, 최대값)
# ax1.set_ylim(50000, 800000)
# ax2.set_ylim(50000, 800000)

# # 축 눈금 라벨 지정 및 75도 회전
# ax1.set_xticklabels(sr_one.index, rotation=75)
# ax2.set_xticklabels(sr_one.index, rotation=75)

# # plt.show()  # 변경사항 저장하고 그래프 출력

# #######################

# # 그래프 객체 생성 (figure에 1개의 서브 플롯 생성)
# fig = plt.figure(figsize = (20, 5))
# ax = fig.add_subplot(1,1,1)

# # axe 객체에 plot 함수로 그래프 출력
# ax.plot(sr_one, marker='o', markerfacecolor='orange', markersize=10, color='olive', linewidth=2, label='서울->경기')
# ax.legend(loc='best')

# # y축 범위 지정(최소값, 최대값)
# ax.set_ylim(50000, 800000)

# # 차트 제목 추가
# ax.set_title('서울 -> 경기 인구이동', size=10)

# # 축 이름 추가
# ax.set_xlabel('기간', size=12)
# ax.set_ylabel('이동 인구수', size=12)

#  # 축 눈금 라벨 지정 및 75도 회전
# ax.set_xticklabels(sr_one.index, rotation=75)

#  # 축 눈금 라벨 크기
# ax.tick_params(axis='x', labelsize=10)
# ax.tick_params(axis='y', labelsize=10)

# plt.show()

# ################################
# # 판다스 예제 4-10, p.124

# # 서울에서 '충청남도', '경상북도', '강원도'로 이동한 인구 데이터 값 선택
# col_years = list(map(str, range(1970, 2018)))
# df_3 = df_seoul.loc[['충청남도', '경상북도', '강원도'], col_years]

# # 스타일 서식 지정
# plt.style.use('ggplot')

# # 그래프 객체 생성(figure에 1개의 서브 플롯 생성)
# fig = plt.figure(figsize=(20, 5))
# ax = fig.add_subplot(1,1,1)

# # axe 객체에 plot 함수로 그래프 출력
# ax.plot(col_years, df_3.loc['충청남도',:], marker='o', markerfacecolor='green',
#         markersize=10, color='olive', linewidth=2, label = '서울->충남')
# ax.plot(col_years, df_3.loc['경상북도',:], marker='o', markerfacecolor='blue',
#         markersize=10, color='skyblue', linewidth=2, label='서울->경북')
# ax.plot(col_years, df_3.loc['강원도',:], marker='o', markerfacecolor='red',
#         markersize=10, color='magenta', linewidth=2, label='서울->강원')

# # 범례 표시
# ax.legend(loc='best')

# # 차트 제목 추가
# ax.set_title('서울->충남, 경북, 강원 인구 이동', size=20)

# # 축 이름 추가
# ax.set_xlabel('기간', size=12)
# ax.set_ylabel('이동 인구수', size=12)

# # 축 눈금 라벨 지정 및 90도 회전
# ax.set_xticklabels(col_years, rotation=90)

# # 축 눈금 라벨 크기
# ax.tick_params(axis="x", labelsize=10)
# ax.tick_params(axis="y", labelsize=10)

# plt.show()  # 변경사항 저장하고 그래프 출력


# ##################################################
# # 판다스 예제 4-11, p.126

# # 서울에서 '충청남도', '경상북도', '강원도', '전라남도'로 이동한 인구 데이터 값만 선택
# col_years = list(map(str, range(1970, 2018)))
# df_4 = df_seoul.loc[['충청남도', '경상북도', '강원도', '전라남도'], col_years]

# # 스타일 서식 지정
# plt.style.use('ggplot')

# # 그래프 객체 생성(figure에 1개의 서브 플롯 생성)
# fig = plt.figure(figsize=(20, 10))
# ax1 = fig.add_subplot(2,2,1)
# ax2 = fig.add_subplot(2,2,2)
# ax3 = fig.add_subplot(2,2,3)
# ax4 = fig.add_subplot(2,2,4)

# # axe 객체에 plot 함수 그래프 출력
# ax1.plot(col_years, df_4.loc['충청남도',:], marker='o', markerfacecolor='green',
#          markersize=10, color='olive', linewidth=2, label = '서울->충남')
# ax2.plot(col_years, df_4.loc['경상북도',:], marker='o', markerfacecolor='blue',
#          markersize=10, color='skyblue', linewidth=2, label = '서울->경북')
# ax3.plot(col_years,df_4.loc['강원도',:], marker='o', markerfacecolor='red',
#          markersize=10, color='magenta', linewidth=2, label = '서울->강원')
# ax4.plot(col_years, df_4.loc['전라남도',:], marker='o', markerfacecolor='orange',
#          markersize=10, color='yellow', linewidth=2, label= '서울->전남')

# # 범레 표시
# ax1.legend(loc='best')
# ax2.legend(loc='best')
# ax3.legend(loc='best')
# ax4.legend(loc='best')

# # 차트 제목 추가
# ax1.set_title('서울->충남 인구 이동', size=15)
# ax2.set_title('서울->경북 인구 이동', size=15)
# ax3.set_title('서울->강원 인구 이동', size=15)
# ax4.set_title('서울->전남 인구 이동', size=15)

# # 축 눈금 라벨 지정 및 90도 회전
# ax1.set_xticklabels(col_years, rotation=90)
# ax2.set_xticklabels(col_years, rotation=90)
# ax3.set_xticklabels(col_years, rotation=90)
# ax4.set_xticklabels(col_years, rotation=90)

# plt.show() # 변경사항 저장하고 그래프 출력


# ####################################

# # 판다스 예제 4-13, P.128

# # 서울에서 다른 지역으로 이동한 데이터만 추출하여 정리
# mask = (df['전출지별'] =='서울특별시')& (df['전입지별'] !='서울특별시')
# df_seoul = df[mask]
# df_seoul = df_seoul.drop('전출지별', axis=1)
# df_seoul.rename({'전입지별':'전입지'}, axis=1, inplace=True)
# df_seoul.set_index('전입지', inplace = True)
# df_seoul

# # 서울에서 '충청남도', '경상북도', '전라남도' 로 이동한 인구 데이터 값만 선택
# col_years = list(map(str, range(1970, 2018)))
# df_4 = df_seoul.loc[['충청남도', '경상북도', '강원도', '전라남도'], col_years]
# df_4 = df_4.T

# # 스타일 서식 지정
# plt.style.use('ggplot')

# # 데이터 프레임의 인덱스를 정수형으로 변경 (x축 눈금 라벨 표시)
# df_4.index = df_4.index.map(int)

# # 면적 그래프 그리기
# df_4.plot(kind='area', stacked = False, alpha = 0.2, figsize=(20, 10))

# plt.title('서울 -> 타시도 인구 이동', size=30)
# plt.ylabel('이동 인구 수', size=20)
# plt.xlabel('기간', size=20)
# plt.legend(loc='best', fontsize=15)

# plt.show()

# #################3
# # 폰트 깨짐 해결
# import matplotlib.pyplot as plt
# from matplotlib import rc
# import seaborn as sns
# %matplotlib inline

# rc('font', family='AppleGothic')

# plt.rcParams['axes.unicode_minus'] = False
# ####################################
# # 폰트 깨짐 해결
# import matplotlib.pyplot as plt
# from matplotlib import rc
# import seaborn as sns
# %matplotlib inline

# rc('font', family='AppltGothic')

# plt.rcParams['axes.unicode_minus']=False

# ############################
# # 파이썬 에제 4-14, p.130

# # 데이터프레임 인덱스를 정수형으로 변경(x축 눈금 라벨 표시)
# df_4.index = df_4.index.map(int)

# # 면적 그래프 그리기
# df_4.plot(kind='area', stacked=True, alpha=0.2, figsize=(20, 10))

# plt.title('서울 -> 타시도 인구 이동', size=30)
# plt.ylabel('이동 인구 수', size=20)
# plt.xlabel('기간', size=20)
# plt.legend(loc='best', fontsize=15)

# plt.show()

#################################
# # 판다스 예제 4-15, p.131
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# df = pd.read_excel('./data/5674-833/part4/시도별 전출입 인구수.xlsx')
# print(df.head())
# df.head()
# df = df.fillna(method = 'ffill')
# df

# mask = (df['전출지별']=='서울특별시') & (df['전입지별']!='서울특별시')
# df[mask]
# df = df[mask]
# df = df.drop('전출지별', axis=1)
# df.rename({'전입지별':'전입지'}, axis = 1, inplace=True)
# df.set_index('전입지', inplace = True)
# df

# df_4 = df.loc[['충청남도', '경상북도' , '강원도', '전라남도']]
# df_4.columns.map(int)
# df_4 = df_4.T

# plt.style.use('ggplot')


# # 면적 그래프 axe 객체 생성

# ax = df_4.plot(kind='area', stacked = True, alpha=0.2, figsize=(20, 10))
# print(type(ax))

# # axe 객체 설정 변경
# ax.set_title('서울 -> 타시도 인구 이동', size = 30, color = 'brown', weight = 'bold')
# ax.set_ylabel('이동 인구 수', size=20, color='blue')
# ax.set_xlabel('기간', size = 20, color='blue')
# ax.legend(loc='best', fontsize=15)

# plt.show()

# ###########################################
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# df = pd.read_excel('./data/5674-833/part4/시도별 전출입 인구수.xlsx')
# df.head()

# df = df.fillna(method='ffill')
# df

# mask = (df['전출지별'] =='서울특별시') & (df['전입지별'] !='서울특별시')
# df_seoul = df[mask]
# df_seoul.drop('전출지별', axis=1, inplace=True)
# df_seoul

# df_seoul.rename({'전입지별':'전입지'}, axis = 1, inplace=True)
# df_seoul.set_index('전입지', inplace=True)
# df_seoul

# col_years = df_seoul.loc[['충청남도', '경상북도', '강원도', '전라남도'], :]
# df_4 = col_years

# df_4 = df_4.loc['2010': '2017']

# col_years.columns.map(int)
# col_years
# df_4
# df_4 = df_4.T
# plt.style.use('ggplot')

# df_4.plot(kind='bar', figsize=(20, 10), width=0.7, color=['orange', 'green', 'skyblue', 'blue'])

# plt.title('서울 -> 타시도 인구이동')
# plt.ylabel('이동 인구 수', size=20)
# plt.xlabel('기간', size=20)
# plt.ylim(5000, 30000)
# plt.legend(loc='best', fontsize=15)

# plt.show()



# df_4['합계'] = df_4.sum(axis=1)
# df_4 = df_4.iloc[:4, :]
# df_4
# df_total = df_4[['합계']].sort_values(by='합계', ascending=True)

# plt.style.use('ggplot')

# df_total.plot(kind='barh', color='cornflowerblue', width=0.5, figsize=(10, 5))

# plt.title('서울 -> 타시도 인구이동')
# plt.ylabel('전입지')
# plt.xlabel('이동 인구 수')
# plt.show()

################################################
# 판다스 예제 4-18, p.135

# # 라이브러리 불러오기
# import pandas as pd
# import matplotlib.pyplot as plt

# plt.style.use('ggplot')
# plt.rcParams['axes.unicode_minus']=False

# # Excel 데이터를 데이터프레임으로 변환
# df = pd.read_excel('/Users/heechankang/Desktop/pythonworkspace/pandas/data/5674-833/part3/남북한발전전력량.xls', convert_float=True)
# df = df.loc[5:9]
# df.drop('전력량 (억㎾h)', axis='columns', inplace=True)
# df.set_index('발전 전력별', inplace=True)
# df = df.T

# # 증감률(변동률 계산)
# df = df.rename({'합계':'총발전량'}, axis=1)
# df['총발전량 - 1년'] = df['총발전량'].shift(1)
# df['증감률'] = ((df['총발전량']/df['총발전량 - 1년'])-1)*100
# df

# # 2축 그래프 그리기
# ax1 = df[['수력', '화력']].plot(kind='bar', figsize=(20, 10), width=0.7, stacked=True)
# ax2 = ax1.twinx()
# ax2.plot(df.index, df.증감률, ls='--', marker='o', markersize=20, color='green', label='전년대비 증감률(%)')

# ax1.set_ylim(0,500)
# ax2.set_ylim(-50, 50)

# ax1.set_xlabel('연도', size=20)
# ax1.set_ylabel('발전량 (억㎾h)')
# ax2.set_ylabel('전년 대비 증감률(%)')

# plt.title('북한 전력 발전량(1990~2016)', size=30)
# ax1.legend(loc='upper left')

# plt.show()
# ########################################

# import pandas as pd
# import matplotlib.pyplot as plt

# plt.style.use('ggplot')
# plt.rcParams['axes.unicode_minus']=False

# df = pd.read_excel('/Users/heechankang/Desktop/pythonworkspace/pandas/data/5674-833/part3/남북한발전전력량.xls', convert_float=True)
# df = df.loc[0:4]
# df
# df.drop('전력량 (억㎾h)', axis=1, inplace = True)
# df.set_index('발전 전력별', inplace = True)
# df = df.rename({'합계':'총발전량'}, axis=0)
# df = df.T
# df['총발전량 - 1년'] = df['총발전량'].shift(1)
# df['증감률'] = ((df['총발전량']/df['총발전량 - 1년']) -1)*100
# df


# ax1 = df[['수력', '화력']].plot(kind='bar', figsize=(20, 10), width=0.7, stacked=True)
# ax2 = ax1.twinx()
# ax2.plot(df.index, df.증감률, ls='--', marker='o', markersize=20, color='green', label='전년대비 증감률(%)')

# ax1.set_ylim(0, 5000)
# ax2.set_ylim(-50, 50)

# ax1.set_xlabel('연도', size=20)
# ax1.set_ylabel('발전량 (억㎾h)')
# ax2.set_ylabel('전년 대비 증감률(%)')

# plt.title('남한 전력 발전량(1990~2016)', size=30)
# ax1.legend(loc='upper left')

# plt.show()

# ########################################

# import pandas as pd
# import matplotlib.pyplot as plt

# plt.style.use('classic')   # 스타일 서식 지정

# # read_csv() 함수로 df 생성
# df = pd.read_csv('/Users/heechankang/Desktop/pythonworkspace/pandas/data/5674-833/part3/auto-mpg.csv', header=None)

# # 열 이름 지정
# df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

# # 연비(mpg) 열에 대한 히스토그램 그리기
# df['mpg'].plot(kind='hist', bins=10, color='coral', figsize=(10, 5))

# # 그래프 꾸미기
# plt.title('HIstorgram')
# plt.xlabel('mpg')
# plt.show()

# ##############################################
# import pandas as pd
# import numpy as np

# df2= pd.DataFrame({
#     'a':[4,5,6],
#     'b':[7,8,9],
#     'c':[10,11,12]},
#     index=[1,2,3],
# )

# df2

# # ##############################################

# #################3
# # 폰트 깨짐 해결
# import matplotlib.pyplot as plt
# from matplotlib import rc
# import seaborn as sns
# %matplotlib inline

# rc('font', family='AppleGothic')

# plt.rcParams['axes.unicode_minus'] = False

# ####################################
# # 폰트 깨짐 해결
# import matplotlib.pyplot as plt
from matplotlib import rc
# import seaborn as sns
# %matplotlib inline

rc('font', family = "AppleGothic")

# plt.rcParams['axes.unicode_minus'] = False

# ####################################

# import pandas as pd
# import matplotlib.pyplot as plt

# df = pd.read_excel('/Users/heechankang/projects/pythonworkspace/pandas/data/5674-833/part4/시도별 전출입 인구수.xlsx', header = 0)

# df = df.fillna(method = 'ffill')

# mask = (df['전출지별'] =='서울특별시') & (df['전입지별'] != '서울특별시')
# df = df[mask]
# df.drop('전출지별', axis = 1, inplace=True)
# df.rename({'전입지별':'전입지'}, axis = 1, inplace=True)
# df.set_index('전입지', inplace=True)
# df
# sr_one = df.loc['경기도']
# sr_one

# plt.plot(sr_one.index, sr_one.values)

# plt.plot(sr_one)
# plt.scatter(sr_one.index, sr_one.values)

# plt.title('서울 -> 경기 인구이동')
# plt.xlabel('기간')
# plt.ylabel('이동 인구수')
# plt.show()


# ####################################
# import pandas as pd
# import numpy as np

# df = pd.DataFrame(columns=['x1', 'x2'])
# df['x1'] = ['A', 'B', 'C','D']
# df['x2'] = [1,2,3,4]

# df.loc[4]=['E', 5]
# df.loc[6] = ['G', 7]
# df.loc[5] = ['F', 6]
# df.sort_index(inplace = True)
# df



# #######################

# import pandas as pd
# import numpy as np
# import seaborn as sns

# df = sns.load_dataset('mpg')
# df.head()

# df.groupby(by='origin').size()

# df.groupby(by='origin')

# df['origin'].value_counts()

# df.groupby(by="origin").min()

# df.groupby(by='origin').max()
# df.groupby(by='origin').mean()

# df['model_year'].head()

# df['model_year'].rank(pct=True)

# df['model_year'].rank(method='first')

# df2=pd.DataFrame(columns = ['a','b','c'])
# df2.loc[1] = [4,7,10]
# df2.loc[2] = [5, 11, 8]
# df2.loc[3] = [6,9,12]

# df2.cummax()
# df2.

import pandas as pd
import matplotlib.pyplot as 

plt.style.use('default')

df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/pandas/data/part5/auto-mpg.csv', header = None)
df.head()

df.columns = ['mpg', 'cylinders', 'displacement', 'horesepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

df.head()

df.plot(kind = 'scatter', x='weight', y= 'mpg', c='coral', s=10, figsize = (10,5))
plt.title('Scatter Plot - mpg vs weight')
plt.show()

cylinders_size = df.cylinders / df.cylinders.max()*300

df.plot(kind='scatter', x='weight', y='mpg', c='coral', s=cylinders_size, figsize=(10, 5), alpha=0.3)


df.plot(kind='scatter', x='weight', y='mpg', marker='+', figsize=(10,5), cmap='viridis', c = cylinders_size, s=50, alpha=0.3)
plt.title('Scatter plot: mpg-weight-cylinders')


import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/pandas/data/part5/auto-mpg.csv', header = None)
df
df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

df['count'] = 1
df_origin = df.groupby('origin').sum()
print(df_origin.head())

df_origin.index = ['USA', 'EU', 'JPN']
df_origin.head()


df_origin['count'].plot(kind='pie',
                        figsize = (7, 5),
                        autopct='%1.1f%%',
                        startangle = 10,
                        colors = ['chocolate', 'bisque', 'cadetblue'])


plt.title('Model Origin', size = 20)

plt.axis('equal')
plt.legend(labels=df_origin.index, loc = 'upper right')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt

plt.stype.use('seaborn-poster')
plt.rcParams['axes.unicode_minus']=False

df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/pandas/data/part5/auto-mpg.csv', header = None)

df.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

fig = plt.figure(figsize=(15, 15))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

ax1. boxplot(x = [df[df['origin']==1]['mpg'],
                  df[df['origin']==2]['mpg'],
                  df[df['origin']==3]['mpg']],
                  labels = ['USA', 'EU', 'JPN'])

ax2.boxplot(x=[df[df['origin']==1]['mpg'],
               df[df['origin']==2]['mpg'],
               df[df['origin']==3]['mpg']],
               labels = ['USA', 'EU', 'JPN'],
               vert=False)

ax3.boxplot(x=[df[df['origin']==1]['mpg'],
               df[df['origin']==2]['mpg'],
               df[df['origin']==3]['mpg']],
               labels = ['USA', 'EU', 'JPN'])

ax4.boxplot(x=[df[df['origin']==1]['mpg'],
               df[df['origin']==2]['mpg'],
               df[df['origin']==3]['mpg']],
               labels=['USA', 'EU', 'JPN'])

ax1.set_title('제조국가별 연비 분포(수직 박스 플롯)')
ax2.set_title('제조국가별 연비 분포(수평 박스 플롯)')         

import seaborn as sns

titanic = sns.load_dataset('titanic')

print(titanic.head())

titanic.info()

import matplotlib.pyplot as plt
import seaborn as sns

titanic = sns.load_dataset('titanic')

sns.set_style('whitegrid')

fig = plt.figure(figsize=(15,5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

sns.regplot(x='age',
            y='fare',
            data=titanic,
            ax=ax1)

sns.regplot(x='age',
            y='fare',
            data = titanic,
            ax=ax2,
            color = 'coral',
            fit_reg=False)

plt.show()

################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import rc

rc('font', family = 'AppleGothic')

titanic = sns.load_dataset('titanic')

fig = plt.figure(figsize = (15, 5))
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

sns.distplot(titanic['fare'], ax = ax1)
sns.distplot(titanic['fare'], ax = ax2, hist = False)
sns.distplot(titanic['fare'], ax = ax3, kde = False)


table = titanic.pivot_table(index = ['sex'], columns = ['class'], aggfunc='size')

sns.heatmap(table, # 데이터프레임
            annot=True, fmt = 'd', # 데이터값 표시 여부, 정수형 포멧
            cmap = 'YlGnBu', # 컬러 맵 지정
            linewidth = 5, # 구분선
            cbar = False)   # 컬러 바 표시 여부

plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# seaborn 제공 데이터셋 가져오기
titanic = sns.load_dataset('titanic')

# 스타일 테마 설정 ( 5가지, whitegrid, darkgrid, dark, white, ticks)
sns.set_style('whitegrid')

# 그래프 객체 생성(figure에 2개의 서브 플롯 설정)
fig = plt.figure(figsize = (15, 5))
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

# 이산형 변수의 분포 - 데이터 분산 미고려(중복 표시 O)
sns.stripplot(x = 'class',
              y = 'age',
              data = titanic,
              ax = ax1)

# 이산형 변수의 분표 - 데이터 분산 고려(중복 표시 X)
sns.swarmplot(x = 'class',
              y = 'age',
              data = titanic,
              ax = ax2)

# 차트 제목 표시
ax1.set_title('Strip Plot')
ax2.set_title('Strip Plot')

plt.show()


fig = plt.figure(figsize = (15, 5))
ax1 = fig.add_subplot(1,3,1)
ax2 = fig.add_subplot(1,3,2)
ax3 = fig.add_subplot(1,3,3)

# x축, y축에 변수 할당
sns.barplot(x= 'sex', y = 'survived', data = titanic, ax = ax1)


# x축, y축에 변수 할당하고 hue 옵션 추가
sns.barplot(x= 'sex', y='survived', data = titanic, ax = ax2, hue = 'class')

# x축, y축에 변수 할당하고 hue 옵션을 추가하여 누적 출력
sns.barplot(x='sex', y='survived', hue='class', dodge = False, data = titanic, ax = ax3)

ax1.set_title('titanic survived - sex')
ax2.set_title('titanic survived - sex/class')
ax3.set_title('titanic survived - sex/class(stacked)')


# 기본값
sns.countplot(x='class', palette='Set1', data=titanic, ax = ax1)

# hue 옵션에 who 추가
sns.countplot(x='class', hue='who', palette = 'Set2', data = titanic, ax=ax2)

# dodge = False 옵션 추가
sns.countplot(x = 'class', hue='who', palette = 'Set3', dodge = False, data = titanic, ax=ax3)

# 차트 제목 표시
ax1.set_title('titanic class')
ax2.set_title('titanic class - who')
ax3.set_title('titanic class - who(stacked)')

plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Seaborn 제공 데이터셋 가져오기
titanic = sns.load_dataset('titanic')

# 스타일 테마 설정 (5가지 : darkgrid, whitegrid, dark, white, ticks)
sns.set_style('dark')

# 그래프 객체 설정
fig = plt.figure(figsize = (15, 10))
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)
ax4 = fig.add_subplot(2,2,4)

# 박스 플롯 - 기본값
sns.boxplot(x = 'alive', y = 'age', data = titanic, ax = ax1)

# 박스 플롯 - hue 변수 추가
sns.boxplot(x = 'alive', y= 'age', hue = 'sex', data = titanic, ax = ax2)

# 바이올린 그래프 - 기본값
sns.violinplot(x = 'alive', y='age', data = titanic, ax = ax3)

# 바이올린 그래프 - hue 변수 추가
sns.violinplot(x = 'alive', y='age', hue='sex', data= titanic, ax = ax4)

plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

titanic = sns.load_dataset('titanic')

sns.set_style('whitegrid')

# 조인트 그래프 - 산점도(기본값)
j1 = sns.jointplot(x='fare', y='age', data=titanic)

# 조인트 그래프 -회귀선
j2 = sns.jointplot(x='fare', y='age', kind='reg', data = titanic)

# 조인트 그래프 - 육각 그래프
j3 = sns.jointplot(x='fare', y='age', kind='hex', data = titanic)

# 조인트 그래프 - 커널 밀집 그래프
j4 = sns.jointplot(x = 'fare', y = 'age', kind = 'kde', data = titanic)

# 차트 제목 표시

j1.fig.suptitle('titanic fare - scatter', size = 15)
j2.fig.suptitle('titanic fare - reg', size = 15)
j3.fig.suptitle('titanic fare - hex', size = 15)
j4.fig.suptitle('titanic fare - kde', size = 15)

plt.show()




import seaborn as sns

titanic = sns.load_dataset('titanic')

df = titanic.loc[:, ['age', 'fare']]

df['ten'] = 10

print(df.head())

def add_10(n):
    return n+10

def add_two_obj(a, b):
    return a+b

print(add_10(10))
print(add_two_obj(10, 11))

sr1 = df['age'].apply(add_10)
print(sr1.head())

sr2 = df['age'].apply(add_two_obj, b=10)
print(sr2)

sr3 = df['age'].apply(lambda x: add_10(x))
print(sr3)


import seaborn as sns

titanic = sns.load_dataset('titanic')

df = titanic.loc[:, 'age', 'fare']
df

df.info()

def add_10(n):
    return n+10

df_map = df.applymap(add_10)



def missing_count(x):
    return missing_value(x).sum()

def total_missing_num(x):
    return(missing_count(x).sum()

missing_result = df.pipe(missing_value)
missing_result

df = titanic
df
df = titanic.loc[0:4, 'survived':'age']

columns = list(df.columns.values)

list(df.index.values)


import seaborn as sns

titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age', 'fare']]
print(df.head())

def add_10(n):
    return n+10

df_map = df.applymap(add_10)
print(df_map.head())

#######

import seaborn as sns

titanic = sns.load_dataset('titanic')

df = titanic.loc[:, ['age', 'fare']]
df.head()


def missing_value(series):
    return series.isnull()

result= df.apply(missing_value, axis = 0)
print(result.head())
print(type(result))

import seaborn as sns

titanic = sns.load_dataset('titanic')

df = titanic.loc[:, ['age', 'fare']]

def min_max(x):
    return x.max() - x.min()

result = df.apply(min_max)
print(result)

import seaborn as sns

titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age', 'fare']]
df

df['ten'] = 10

def add_two_obj(a, b):
    return a+b

df['add'] = df.apply(lambda x: add_two_obj(x['age'], x['ten']), axis = 1)
df.head()

import seaborn as sns

titanic = sns.load_dataset('titanic')

df = titanic.loc[:, ['age', 'fare']]
df

def missing_value(x):
    return x.isnull()

def missing_count(x):
    return missing_value(x).sum()

def total_number_missing(x):
    return missing_count(x).sum()

result_df = df.pipe(missing_value)
print(result_df.head())
print(type(result_df))

result_series = df.pipe(missing_count)
result_series

df = titanic.loc[0:4, 'survived':'age']
df

columns= list(df.columns.values)
columns

columns_sorted = sorted(columns)
df_sorted = df[columns_sorted]

df_sorted
print(df_sorted, '\n')

columns_reversed = list(reversed(columns))
df_reversed = df[columns_reversed]
df_reversed

columns_customed = ['pclass', 'sex', 'age', 'survived']
df_customed = df[columns_customed]

print(df_customed)

df = pd.read_excel('/Users/heechankang/projects/pythonworkspace/pandas/data/part6/주가데이터.xlsx')
print(df)

df['연월일'] = df['연월일'].astype('str')
dates = df['연월일'].str.split('-')
print(dates.head(), '\n')

df['연'] = dates.str.get(0)
df['월'] = dates.str.get(1)
df['일'] = dates.str.get(2)

print(df.head())
df

import seaborn as sns

titanic = sns.load_dataset('titanic')

mask1 = (titanic.age >=10) & (titanic.age < 20)
df_teenage = titanic.loc[mask1, :]

df_teenage.head()

mask2 = (titanic.age < 10) & (titanic.sex == 'female')
df_female_under10 = titanic.loc[mask2,:]

df_female_under10

mask3 = (titanic.age < 10) | (titanic.age >= 60)
df_under10_morethan60 = titanic.loc[mask3, ['age', 'sex', 'alone']]
df_under10_morethan60.head()

df_f = titanic.loc[(titanic.age < 10) | (titanic.sex == 'female'), ['age', 'sex', 'alone']]
df_f

pd.set_option('display.max_columns', 10)

titanic

mask3 = titanic['sibsp'] == 3
mask4 = titanic['sibsp'] == 4
mask5 = titanic['sibsp'] == 5

titanic['sibsp']==3

df_bool = titanic[mask3|mask4|mask5]

df_bool.head()

titanic[mask5]

titanic['sibsp'].isin([3,4,5])

filter_isin = titanic['sibsp'].isin([3,4,5])

filter_isin

df1 = pd.DataFrame({
    'a': ['a0', 'a1', 'a2', 'a3'],
    'b': ['b0', 'b1', 'b2', 'b3'],
    'c': ['c0', 'c1', 'c2', 'c3'],
    'd': ['d0', 'd1', 'd2', 'd3']
}, index= [0, 1, 2, 3])

df2 = pd.DataFrame({
    'a':['a2', 'a3', 'a4', 'a5'],
    'b':['b2', 'b3', 'b4',' b5'],
    'c':['c2', 'c3', 'c4', 'c5'],
    'd':['d2', 'd3', 'd4', 'd5']
}, index= [2,3,4,5])

df1
df2

result = pd.concat([df1, df2])
result

result2 = pd.concat([df1, df2], ignore_index=True)
print(result2)
print(result2, '\n')

result3 = pd.concat([df1, df2], axis=1)
result3

result3_in = pd.concat([df1, df2], axis=1, join = 'inner')
result3_in
df1
df2

sr1 = pd.Series(['e0', 'e1', 'e2', 'e3'], name = 'e')
sr2 = pd.Series(['f0', 'f1', 'f2'], name= 'f', index=([3,4,5]))
sr3 = pd.Series(['g0', 'g1', 'g2', 'g3'], name = 'g')

result4 = pd.concat([df1, sr1], axis = 1)
result4
re = pd.concat([df1, sr1])

result5 = pd.concat([df2, sr2], axis = 1, sort=True)
result5

re = pd.concat([df2, sr2], axis=1)
re

result6 = pd.concat([sr1, sr3], axis=1)
result6

result7 = pd.concat([sr1, sr3], axis = 0)
result7

import pandas as pd

pd.set_option('display.max_columns', 10)
pd.set_option('display.max_colwidth', 20)
pd.set_option('display.unicode.east_asian_width', True)

df1 = pd.read_excel('/Users/heechankang/projects/pythonworkspace/pandas/data/part6/stock price.xlsx', index_col='id')
df2 = pd.read_excel('/Users/heechankang/projects/pythonworkspace/pandas/data/part6/stock valuation.xlsx', index_col='id')

df1

df2

merge_inner = pd.merge(df1, df2)
merge_inner

merge_outer = pd.merge(df1, df2, how = 'outer', on='id')
merge_outer

merge_left = pd.merge(df1, df2, how = 'left', left_on='stock_name', right_on='name')
merge_left

merge_right = pd.merge(df1, df2, how = 'right', left_on='stock_name', right_on='name')
merge_right

df1[df1.price < 5000]

df1.price
df1.price<5000
price = df1[df1.price<5000]

pd.merge(price, df2)


df3 = df1.join(df2)
df3

df4 = df1.join(df2, how = 'inner')
df4

import pandas as pd
import seaborn as sns

titanic = sns.load_dataset('titanic')
df = titanic.loc[:, ['age', 'sex', 'class', 'fare', 'survived']]

print('승객 수 : ', len(df))

print(df.head())

grouped = df.groupby(['class'])

print(grouped)

for k, g in grouped:
    print('* key : ', k)
    print('* number : ', len(g))
    print(g.head())

grouped.mean()

grouped.median()

grouped.get_group('Second')

grouped = df.groupby(['class', 'sex'])
for k, g in grouped:
    print( 'key : ', k)
    print( 'number : ', len(g))
    print(grouped.head())

grouped.mean()

grouped.get_group(('Third', 'female'))
grouped.get_group('Female')
grouped.get_group('Third')

df = titanic

grouped = df.groupby(['class'])

len(grouped)

grouped.std()

grouped.age.mean()
grouped.age.std()

grouped.fare.mean()
grouped.fare.std()


def min_max(x):
    return x.max() - x.min()

grouped.agg(['min', 'max'])

grouped.agg(min_max)


grouped.agg({'fare':['min', 'max'], 'age':'mean'})