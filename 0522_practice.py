import pandas as pd

dict_data = {'a':1, 'b':2, 'c':3}

sr = pd.Series(dict_data)

sr

list_data = ['2019-01-02', 3.14, 'ABC', 100, True]

sr = pd.Series(list_data)

print(sr)
sr

idx = sr.index
val = sr.values
print(idx)
print(val)

tup_data = ('영인', '2010-05-01', '여', True)

sr = pd.Series(tup_data, index = ['이름', '생년월일', '성별', '학생여부'])

sr
print(sr[0])
print(sr['이름'])

print(sr[[1, 2]])
print(sr[['생년월일', '이름']])

print(sr[1:2])

print(sr['생년월일':'성별'])


dict_data = {'c0':[1,2,3], 'c1':[4,5,6], 'c2':[7,8,9], 'c3':[10,11,12], 'c4':[13, 14,15]}

df = pd.DataFrame(dict_data)
df
type(df)

df = pd.DataFrame([[15, '남', '덕영중'], [17, '여', '수리중']],
                  index = ['준서', '예은'],
                  columns = ['나이', '성별', '학교'])

print(df)

df.index = ['학생1', '학생2']
df
df.columns = ['연령', '남녀','소속']
df


exam_data = {'수학':[90, 80, 70], '영어':[98,89,95],
             '음악':[85, 95, 100], '체육': [100, 90, 90]}

df= pd.DataFrame(exam_data)

df2 = df[:]
df2

df = pd.DataFrame(exam_data, index = ['서준', '우현', '인아'])
df

df2 = df[:]

df2.drop('우현', inplace = True)

df2

df