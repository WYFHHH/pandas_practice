# import pandas as pd
# import numpy as np

# df = pd.read_csv('/Users/heechankang/projects/pythonworkspace/git_study/pandas/data/seoul_covid19_9_28.csv', encoding = 'cp949')
# df

# df = df.sort_values(by = ['연번'])
# df.head()

# import matplotlib.pyplot as plt
# plt.rc('font', family = 'AppleGothic')
# plt.rc('axes', unicode_minus = False)
# plt.style.use('ggplot')

# pd.Series([1,3,-5,-7,]).plot.bar(title = '한글제목')

# from IPython.display import set_matplotlib_formats
# set_matplotlib_formats('retina')

# df.info()
# df.describe(include = object)

# df.groupby('확진일').count().sort_values(by='환자', ascending=False)

# df['확진일'].value_counts()


# # 데이터 구조 파악
# df['확진일'].head()
# df['확진일'].value_counts()

# df['확진일자'] = pd.to_datetime('2020-'+df['확진일'].str.replace('/', '-'))
# df['확진일자'].head()
# df

# df['확진일자'].value_counts().plot()

# # 확진일자로 선 그래프 그리기
# df['확진일자'].value_counts().sort_index().plot(figsize = (15, 4))
# plt.axhline(50, color = 'blue', linestyle = ':')

# # 뒤에서 문자 5개 가지고오기
# df['확진일자'].astype(str).head()
# df['확진일자'].astype(str).map(lambda x: x[-5:]).head()

# # 주 컬럼 만들기
# df['주'] = df['확진일자'].dt.isocalendar().week
# df['주'].head()
# df['test'] = df['확진일자'].dt.time
# df
# df.info()

# # '월일' 컬럼 만들기
# df['월일'] = df['확진일자'].astype(str).map(lambda x : x[-5:])
# day_count = df['월일'].value_counts().sort_index()
# day_count

# # 선 그래프에 text 삽입
# g = day_count.plot(figsize = (15, 4))
# g.text(x = 2, y = 3, s = 20)

# # 환자 발생이 50명 이상인것만 추출
# for i in range(len(day_count)):
#     case_count = day_count.iloc[i]
#     if case_count > 50:
#         print(i, case_count)

# g = day_count.plot(figsize = (15, 4))
#  for i in range(len(day_count)):
#      case_count = day_count.iloc[i]
#      if case_count > 50:
#          g.text(x = i, y = case_count, s = case_count)       

# # 확진자가 가장 많이 나온 날
# day_count.describe()

# day_count[day_count ==day_count.max()]

# # 확진자가 가장 많았던 날의 발생이력
# df[df['월일'] == '08-29'].head()

# # 막대그래프 그리기
# day_count.plot.bar(figsize = (15, 4))


# # 슬라이싱으로 나누어 그리기
# day_count[-50:].plot.bar(figsize = (15, 4))

# g = day_count[-50:].plot.bar(figsize = (15, 4))
# g.axhline(day_count.median(), linestyle = ':', color = 'blue')

# for i in range(50):
#     case_count = day_count[-50:].iloc[i]
#     if case_count >= 50:
#         g.text(x = i-0.5, y = case_count, s = case_count)


# # 월별 확진자수에 대한 빈도수를 구해서 시각화
# df['월'] = df['월일'].astype(str).map(lambda x : x[:2])
# df['일'] = df['월일'].astype(str).map(lambda x : x[-2:])
# df.head(15)

# month_case = df['월'].value_counts().sort_index()
# g = month_case.plot.bar(rot = 0)

# for i in range(len(month_case)):
#     g.text(x = i-0.2, y = month_case.iloc[i]+10, s = month_case.iloc[i])

# # 주 단위 확진자수 그리기
# import datetime
# df['주'] = df['월일'].astype(datetime).isocalendar()[1]
# df['주'] = df['월일'].astype(str).map(lambda x : x[-2:])
# weekly_case = df['']

######################################
import pandas as pd
import numpy as np
import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/pandas/data')

file_name = 'seoul_covid19_9_28.csv'
file_name

df = pd.read_csv(file_name, encoding = 'cp949')
df

# 연번 기준으로 정렬
df = df.sort_values(by='연번')
df

import matplotlib.pyplot as plt

sr = pd.Series([1,3,5,7,9])
plt.rc('font', family = 'AppleGothic')
plt.rc('axes', unicode_minus = False)
plt.style.use('ggplot')
sr.plot.bar(title = '한글 제목')

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')


# 가장 많은 확진자가 나온 일자
df['확진일'].max()
df[df['확진일']=='9.28.'].count()

df.info()
df['확진일자'] = pd.to_datetime(df['확진일'].str)
df.drop('확진일자', axis = 1, inplace = True)

df
df['확진일자'] = pd.to_datetime('2020-'+df['확진일'])
df

df['월'] = df['확진일자'].astype('str').map(lambda x : x[5:7])
df['주'] = df['확진일자'].dt.week
df = df.reset_index().drop('index', axis = 1)
df

df['확진일자']
df['확진일자'].value_counts()

df['확진일자'].value_counts().plot()
df['월일'] = df['확진일자'].astype('str').map(lambda x : x[-5:])
df

day_count = df['확진일자'].value_counts().sort_index()
day_count.plot()
plt.axhline(50, color = 'blue', linestyle = ':')

day_count.plot.bar()

plt.axhline(50, color = 'blue', linestyle = ':')
day_count[-50:].plot.bar(figsize = (15, 4))

g = day_count[-50:].plot.bar(figsize = (15, 4))
g.axhline(day_count.median(), linestyle = ':', color ='blue')

for i in range(50):
    case_count = day_count[-50:].iloc[i]
    if case_count >= 50:
        g.text(x = i-0.5, y = case_count, s = case_count)

month_case = df['월'].value_counts().sort_index()
month_case
g = month_case.plot.bar(rot = 0)

for i in range(len(month_case)):
    g.text(x = i-0.3, y = month_case.iloc[i] + 10, s = month_case.iloc[i])

weekly_case = df['주'].value_counts().sort_index()
weekly_case
weekly_case.plot.bar(figsize = (15, 4), rot = 0)

df.groupby(['월', '주']).count()

month_weekly_case = df.groupby(['월', '주'])['확진일'].count()
month_weekly_case.plot.bar(figsize = (15, 4), rot = 60)

day_count[-20:].plot.bar()

first_day = df.iloc[-1, 7]
last_day = df.iloc[0, 7]
first_day

days = pd.date_range(first_day, last_day)
days

daily_case = df['확진일자'].value_counts()
daily_case.head()
df_daily_case = df['확진일자'].value_counts()
df_daily_case.head()

all_day = days.merge(df_daily_case,
                        left_on = '확진일자',
                        right_on = df_daily_case.index, how = 'left')
all_day.head()

