import pandas as pd
import numpy as np
%matplotlib inline

s = pd.Series(np.random.randn(1000),
                              index = pd.date_range('1/1/2015', periods=1000))
s
s.plot()

s = s.cumsum() # 누적 합
s.plot()

# rolling : 이동평균 구할 때 사용
r = s.rolling(window = 30)
r

# window : size of moving window

r.mean()

# 이동평균
s.plot(style = 'b--')

# 이동평균
r.mean().plot(style = 'r')

s.plot(style = 'b--')
r.mean().plot(style = 'r')

# expanding
df = pd.DataFrame(np.random.randn(1000, 4),
                  index = pd.date_range('1/1/2015', periods = 1000),
                  columns = ['A', 'B', 'C', 'D'])

df = df.cumsum()
df

df.plot()

df.rolling(window=60).sum().plot(subplots = True)

df.rolling(window = len(df), min_periods = 1).mean().plot()

df.expanding?
# 확장된 변형 제공
# 누적도니 변경 값의 정보 제공

df.expanding(min_periods = 1).mean()

df.expanding(min_periods = 1).mean().plot()

dfe = pd.DataFrame({'B' : [0,1,2,np.nan,4]})
dfe

dfe.plot()

dfe.expanding(2).sum() # 누적합

dfe.expanding(2).sum().plot()

dfe.expanding(2).mean().plot()

####################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
%matplotlib inline

# 폰트깨짐 해결
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

df = pd.DataFrame()

df.plot?

ts = pd.Series(np.random.randn(1000),
               index = pd.date_range('1/1/2000', periods = 1000))
# 1000개의 수를 랜덤하게 2000년 1월 1일부터 1000일 간 생성

ts.plot()

ts = ts.cumsum()
ts
ts.plot()

df = pd.DataFrame(np.random.randn(1000, 4),
                  index = ts.index, columns = list('ABCD'))
df

df.plot()

df = df.cumsum()
df.plot()

df3 = pd.DataFrame(np.random.randn(1000, 2), columns = ['B', 'C']).cumsum()
df3.head()

df3['A'] = pd.Series(list(range(len(df))))
# A 컬럼 생성 (index)
df3.head()

df3.plot(x = 'A', y = 'B')

######################################################

ts = pd.Series(np.random.randn(1000),
               index = pd.date_range('1/1/2000', periods = 1000))

ts.head()

df = pd.DataFrame(np.random.randn(1000, 4),
                  index=pd.date_range('1/1/2000', periods = 1000))

df.head()
df.tail()


df.iloc[5]

df.iloc[5].plot(kind='bar')

df.iloc[5].plot.bar()

df.iloc[5].plot.bar()
plt.axhline(0, color='k')

df2 = pd.DataFrame(np.random.rand(10, 4), columns = ['a','b', 'c', 'd'])
df2.head(3)

df2.plot.bar()

df2.plot.bar(stacked=True)

df2.plot.barh(stacked=True)


###############################################

df4 = pd.DataFrame(
    {
        'a' : np.random.randn(1000)+1,
        'b' : np.random.randn(1000),
        'c' : np.random.randn(1000)-1,
    }, columns = ['a', 'b','c']
)
df4.head()

df4.plot.hist(alpha = 0.5)

df4.plot.hist(stacked=True, bins=20)

df4['a'].plot.hist(orientation = 'horizontal', cumulative = True)

df4['a'].diff().hist()

df4['a_diff'] = df4['a'].diff()
df4[['a','a_diff']]

df4['a_shift'] = df4['a'].shift(1)
df4['a_minus'] = df4['a'] - df4['a_shift']
df4[['a', 'a_shift', 'a_minus', 'a_diff']].head()

df4[['a', 'b', 'c']].diff().hist(color = 'k', alpha = 0.5, bins = 30)

data = pd.Series(np.random.randn(1000))
data.hist(by=np.random.randint(0, 4, 1000), figsize=(6, 4))

data = pd.DataFrame(
    {
        'a': np.random.randn(1000),
        'b': np.random.randint(0, 4, 1000)
    }
)
data.head()

data['a'].hist(by=data['b'], figsize=(6, 4))

######################################################

df = pd.DataFrame(np.random.rand(10, 5), columns= ['A','B','C','D','E'])
df.head()

df.describe()

df = df

df.plot.box()

color = {'boxes': 'DarkGreen', 'whiskers':'DarkOrange','medians':'DarkBlue', 'caps':'Gray'}

df.plot.box(color = color, sym = 'r+')

df.plot.box(vert = False, positions = [1,4,5,6,8])

df = pd.DataFrame(np.random.rand(10,5))

df.head(1)

bp = df.boxplot()

df = pd.DataFrame(np.random.rand(10,2), columns = ['Col1', 'Col2'])
df.head(2)

df['X'] =pd.Series(['A','A','A','A','A', 'B', 'B', 'B', 'B', 'B'])
df.head()

dp = df.boxplot(by='X')

np.random.seed(1234)
df_box = pd.DataFrame(np.random.randn(50, 2))
df_box.head()

df_box['g'] = np.random.choice(['A', 'B'], size = 50)
df_box['g'].head()

df_box.loc[df_box['g'] == 'B', 1] +=3
df_box.boxplot(by='g')

bp = df_box.boxplot(by='g')

bp = df_box.groupby('g').boxplot()

######################################################

df = pd.DataFrame(np.random.rand(10, 4), columns = ['a', 'b','c','d'])
df.head()

df.plot()

df.plot(grid = True)

df.plot.area(grid=True)

df.plot.area(stacked=False, grid = True)