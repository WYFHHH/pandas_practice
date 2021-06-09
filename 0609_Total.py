# from numpy.core.numeric import NaN
# import pandas as pd
# import numpy as np


# lists = [1,2,3,4,5]
# sets = {1,2,3,4,5}

# dicts = {1:[1,2,3,4], 2:[1,0,3,4], 3:[1,2,3,4], 4:[1,2,3,4], 5:[1,2,3,4]}

# lists
# sr = pd.Series(lists)
# sr

# sr2 = pd.Series(dicts)
# sr2

# sr3 = pd.Series(sets)

# df = pd.DataFrame(lists)
# df
# df2 = pd.DataFrame(dicts)
# df2
# df2.iloc[2][2] 

# list(sr.index)
# list(sr.values)

# df = pd.DataFrame([[18, '남', '김천고'], [19, '여', '울산고']])
# df.columns

# df.loc['0':]
# df.iloc[0:1]

df = pd.DataFrame({'수학':[100, 40, 70], '영어':[50, 70, 90], '생물':[50, 90, 70]}, index = ['진현', '민지', '성철'])
df
df.loc['진현':'민지', '수학':'영어']
df.iloc[0:2, 0:2]
df.loc[:'민지']
df.iloc[:1]

df.loc[['민지']]
df['수학']
df[['수학']]
df
df.loc['경철'] = [10, 20, 30]
df
df.set_index('수학')
df.iloc[1,1]
df.iloc[1][1]
df.sort_index()
df.T.sort_index().T
df = df.reset_index()
df = df.set_index('index')
df.value_counts()

df.corr()

import os
os.chdir('/Users/heechankang/projects/pythonworkspace/git_study/pandas')
df.to_csv('check1')

ddf = pd.read_csv('check1')
ddf

l = [1,2,3,4]
del l[1::2]
print(l)
df = df.reindex(['진현', '민지', '성철', '경철', '민환'])
df
df.iloc[1][1] = np.NaN
df.isnull().sum()
df.to_clipboard()

missing_df = df.isnull()
for col in missing_df.columns:
    missing_count = missing_df[col].value_counts()
    try:
        print(col, ':', missing_count[True])
    except:
        print(col,':', 0)

a = np.arange(12).reshape(3,4)
a = pd.DataFrame(a)
a = a.T
a.iloc[1,0] = 0
a.iloc[2,2] = 0
a.iloc[0,1] = 5
a.iloc[1,2] = 11
a
a.value_counts().idxmax()
###################
from urllib.request import urlopen
from urllib.parse import urlencode, unquote, quote_plus

import pandas as pd
import xmltodict
import json

key = '%2Bn6sp%2BC0PPCjdXCvaUOBOw40kXcyTxDHkl5NVIR0cGwPcstPd0exjN5htFO8mY7ni06KEd8FQ3D5HbcK%2BCrFhQ%3D%3D'
url = f'http://openapi.data.go.kr/openapi/service/rest/Covid19/getCovid19InfStateJson?serviceKey={key}&'
queryParams = urlencode({quote_plus('pageNo') : 1,
                         quote_plus('numOfRows') : 10,
                         quote_plus('startCreateDt') : '20210119',
                         quote_plus('endCreateDt') : '20210120'})

url2 = url + queryParams
response = urlopen(url2)
#print(type(response)) # HTTPS response
results = response.read().decode('utf-8')
# print(type(results)) # str
results_to_json = xmltodict.parse(results)
data = json.loads(json.dumps(results_to_json))
print(type(data)) # dict
print(data)


my_data = data['response']['body']['items']['item'][:]


df = pd.DataFrame(my_data)
df = df[['stateDt', 'decideCnt', 'clearCnt','careCnt', 'deathCnt']]
df.columns = ['날짜', '누적확진자', '격리해제환자', '치료중환자', '사망자수']
df = df.sort_values(by='날짜')
df
###########
# traget
# STATE_DT
# DECIDE_CNT
# CLEAR_CNT
# CARE_CNT
# DEATH_CNT

# 날짜,누적확진자,격리해제환자,치료중환자,사망자수
# 20210119,73104,59468,12353,1283
# 20210120,73508,60180,12028,1300

###########################
import collections
deq = collections.deque()
deq = [1,2,3,4,5]
type(deq)


df['수학'].apply(int)

mean_value = df.mean()
df = df.fillna(mean_value)
df['수학'].apply(int)
df


sel_1 = (df['수학']>50) & (df['영어']<90)
df_sel = df.loc[sel_1,'영어']
df_sel

a = (df['수학'] ==10.0)
a

####################



{'response': 
    {'header': 
        {'resultCode': '00', 'resultMsg': 'NORMAL SERVICE.'},
    'body': 
        {'items': 
            {'item': 
                [
                    {'accDefRate': '1.4573418044', 'accExamCnt': '5192119', 'accExamCompCnt': '5043978', 'careCnt': '12028', 'clearCnt': '60180',
                     'createDt': '2021-01-20 09:38:16.549', 'deathCnt': '1300', 'decideCnt': '73508', 'examCnt': '148141', 'resutlNegCnt': '4970470',
                     'seq': '392', 'stateDt': '20210120', 'stateTime': '00:00', 'updateDt': '2021-04-20 15:23:25.562'},
                    {'accDefRate': '1.4661941407', 'accExamCnt': '5140315', 'accExamCompCnt': '4985970', 'careCnt': '12353', 'clearCnt': '59468',
                     'createDt': '2021-01-19 09:39:49.588', 'deathCnt': '1283', 'decideCnt': '73104', 'examCnt': '154345', 'resutlNegCnt': '4912866',
                     'seq': '391', 'stateDt': '20210119', 'stateTime': '00:00', 'updateDt': '2021-04-20 15:23:34.538'}
                ]
             },
        'numOfRows': '10', 'pageNo': '1', 'totalCount': '2'}
    }
}
