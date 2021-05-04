# 1. Object Creation (객체 생성)
# 2. Viewing Data (데이터 확인하기)
# 3. Selection (선택)
# 4. Missing Data (결측치)
# 5. Operation (연산)
# 6. Merge (병합)
# 7. Grouping (그룹화)
# 8. Reshaping (변형)
# 9. Time Series (시계열)
# 10. Categoricals (범주화)
# 11. Plotting (그래프)
# 12. Getting Data In / Out (데이터 입 / 출력)
# 13. Gotchas (잡았다!)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

s = pd.Series([1,3,5,np.nan, 6,8])
s

dates = pd.date_range('20130101', periods=6)
dates

df = pd.DataFrame(np.random.randn(6,4), index=dates, columns = list('ABCD'))
df

df2 = pd.DataFrame({'A':1.,
                    'B':pd.Timestamp('20130102'),
                    'C':pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D':np.array([3]*4, dtype='int32'),
                    'E':pd.Categorical(['test', 'train', 'test', 'train']),
                    'F':'foo'})

df2

df2.dtypes

df.head()
df.index
df.columns
df.values

df.describe

df.T
df
df.sort_index(axis=1, ascending=True)
df.sort_values('A', ascending=False)

df['A']
df.A
df[0:3]

df['20130102':'20130105']

df.loc[dates[4]]

df.loc[:, ['A', 'B']]

df.loc['20130102':'20130104', ['A', 'B']]

df.loc['20130102', ['A', 'B']]

df.loc[dates[0],'A']

df.at[dates[0], 'A']

df.iloc[3]

df.iloc[3:5, 0:2]

df.iloc[[1,2,4], [0,2]]

df.iloc[1:3, :]

df.iloc[:, 1:3]

df.iloc[1,1]

df.iat[1,1]

df[df.A > 0]

df[df>0]

df2 = df.copy()

df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
df2

df2[df2['E'].isin(['two', 'four'])]

s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))
s1

df['F'] = s1
df
df.at[dates[0], 'A']=0
df.iat[0,1] = 0
df
df.loc[:, 'D'] = np.array([5]*len(df))

df

df2 = df.copy()

df2[df2>0] = -df2

df2

df1 = df.reindex(index=dates[0:4], columns = list(df.columns)+['E'])
df1

df1.loc[dates[0]:dates[1], 'E']=1
df1

df1.dropna(how='any')

df1 = df1.drop('F', axis = 1)
df1.dropna(how= 'any')

df1.fillna(value=5)

pd.isna(df1)

pd.notna(df1)

df.mean()

df.mean(1)

s = pd.Series([1,3,5,np.nan, 6,8], index= dates).shift(2)
s

df.sub(s, axis='index')

df.apply(np.cumsum)

df.apply(lambda x: x.max() - x.min())

s = pd.Series(np.random.randint(0, 7, size=10))
s

s.value_counts()

s = pd.Series(['A', 'B', 'C', 'Aaba', 'Baca', np.nan, 'CABA', 'dog', 'cat'])
s

s.str.lower()

df = pd.DataFrame(np.random.randn(10,4))
df

pieces = [df[:3], df[3:7], df[7:]]
pieces

pd.concat(pieces)

left = pd.DataFrame(
    {
        'key': ['foo', 'foo'],
        'lval':[1,2],
    }
)
right = pd.DataFrame(
    {
        'key':['foo', 'foo'],
        'rval':[4,5]
    }
)
left
right

pd.merge(left, right, on='key')

left = pd.DataFrame(
    {
        'key':['foo', 'bar'],
        'lval' : [1,2]
    }
)
right= pd.DataFrame(
    {
        'key':['foo', 'bar'],
        'rval':[4,5]
    }
)
left
right

pd.merge(left, right, on='key')

df = pd.DataFrame(np.random.randn(8, 4), columns=['A', 'B', 'C', 'D'])

df

s = df.iloc[3]
s

df.append(s, ignore_index=True)


df = pd.DataFrame(
    {
        'A':['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo','foo'],
        'B':['one', 'one', 'two', 'three', 'two' ,'two', 'one', 'three'],
        'C':np.random.randn(8),
        'D':np.random.randn(8)
    }
)
df

df.groupby('A').sum()

df.groupby(['A', 'B']).sum()

tuples = list(zip(*[['bar', 'bar','baz','baz',
                     'foo', 'foo', 'qux', 'qux'],
                     ['one', 'two', 'one', 'two',
                     'one', 'two', 'one', 'two']]))

tuples

index = pd.MultiIndex.from_tuples(tuples, names = ['firtst', 'second'])
df = pd.DataFrame(np.random.randn(8, 2), index = index, columns = ['A', 'B'])
df2 = df[:4]
df2

stacked = df2.stack()
stacked

stacked.unstack()

stacked.unstack(0)

import pandas as pd
import numpy as np

df = pd.DataFrame(
    {
        'A': ['one', 'one', 'two', 'three'] *3,
        'B': ['A', 'B', 'C']*4,
        'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar']*2,
        'D': np.random.randn(12),
        'E': np.random.randn(12)
    }
)
df

pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C'])

rng = pd.date_range('1/1/2012', periods=100, freq='S')
rng
ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
ts.resample('5Min').sum()

rng = pd.date_range('3/6/2012 00:00', periods = 5, freq='D')
rng

ts = pd.Series(np.random.randn(len(rng)), rng)
ts

ts_utc = ts.tz_localize('UTC')
ts_utc

ts_utc.tz_convert('US/Eastern')

rng = pd.date_range('1/1/2012', periods=5, freq='M')
ts = pd.Series(np.random.randn(len(rng)), index=rng)
ts

ps = ts.to_period()
ps

ps.to_timestamp()


prng = pd.period_range('1990Q1', '2000Q4', freq='Q-NOV')

ts = pd.Series(np.random.randn(len(prng)), prng)
ts

ts.index = (prng.asfreq('M', 'e') +1).asfreq('H', 's') +9

ts.head()


df = pd.DataFrame(
    {
        'id':[1,2,3,4,5,6],
        'raw_grade':['a', 'b' ,'b', 'a','a', 'e']
    }
)

df

df['grade'] = df['raw_grade'].astype