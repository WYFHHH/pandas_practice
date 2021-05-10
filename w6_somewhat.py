import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['axes.unicode_minus'] = False

df = pd.DataFrame(np.random.rand(50, 4), columns = ['a', 'b', 'c', 'd'])

df.plot.scatter(x='a', y = 'b', s = 30, grid = True)

ax = df.plot.scatter(x='a', y='b', color='DarkRed', label='Group 1')
df.plot.scatter(x='c', y='d', color = 'DarkGreen', label = 'Group 2', ax = ax)

df.plot.scatter(x = 'a', y = 'b', c = 'c', s = 100)

df.plot.scatter(x = 'a', y = 'b', s = df['c'] * 200)

#######################

import pandas as pd
import numpy as np
import matplotlib as mpl

df = pd.DataFrame(np.random.randn(1000, 2), columns = ['a', 'b'])
df['b']

df = pd.DataFrame(np.random.randn(1000, 2), columns = ['a', 'b'])
df['b'] = df['b'] + np.arange(1000) 
df['b']

df.plot.hexbin(x = 'a', y = 'b', gridsize = 10)

df.plot.hexbin(x = 'a', y = 'b', gridsize = 5)

df['z'] = np.random.uniform(0, 3, 1000)
df['z']

df.plot.hexbin(x = 'a', y = 'b', C = 'z', reduce_C_function = np.max, gridsize = 20)

df.plot.hexbin(x = 'a', y = 'b', C = 'z', reduce_C_function = np.median, gridsize = 50)

###############
#Pie plot

import pandas as pd
import numpy as np
import matplotlib as mpl

series = pd.Series(3 * np.random.rand(4), index = ['a', 'b', 'c', 'd'], name = 'series')
series

series.plot.pie(figsize = (6, 6))

series.plot.pie(labels = ['AA', 'BB', 'CC','DD'],
                colors = ['r', 'g', 'b','c'],
                autopct = '%.2f',
                fontsize = 20, 
                figsize = (6, 6))

df = pd.DataFrame(
    {'mass' : [0.330, 4.87, 5.97],
     'radius' : [2439.7, 6051.8, 6378.1]},
    index = ['Mercury', 'Venus', 'Earth'])
plot = df.plot.pie(y='mass', figsize = (5,5))

plot = df.plot.pie(subplots = True, figsize = (10, 5))


################
# Scatter Matrix Plot
import pandas as pd
import numpy as np
import matplotlib as mpl

from pandas.plotting import scatter_matrix

df = pd.DataFrame(np.random.randn(1000, 4), columns = ['a', 'b','c','d'])
df.head()

scatter_matrix(df, alpha=0.2, figsize = (6,6), diagonal='kde')


####################
# 확률 밀도 함수
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ser = pd.Series(np.random.randn(1000))
ser.plot.hist()

ser.plot.density()

ser.plot.kde()


##############
# 서울 코로나 발생동향 분석
import pandas as pd
import numpy as np
url = 'http://www.seoul.go.kr/coronaV/coronaStatus.do'
url

table = pd.read_html(url)
len(table)

table[0]
table[0].T
table[1]
table[2]
table[3]
table[4]
table[5]
table[7]
table[5]

df = table[1]
df.shape

df.head()

df_temp = table[1]
df_temp.head()
df_temp = df_temp.drop(['광진구'], axis = 1)
df.head()

last_place = df.loc[0, '강서구']
last_place = last_place.replace('0', '_')
last_place

import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/pandas')

file_name = f'seoul_covid19_{last_place}.csv'
file_name

df.head(2)

df.to_csv(file_name)
df.to_csv(file_name, index = False, encoding = 'cp949')

pd.read_csv(file_name, encoding = 'cp949')

df = df.sort_values(['강동구'], ascending = False)
df.head()
df.tail()