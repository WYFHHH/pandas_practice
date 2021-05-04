import pandas as pd

# 파일 경로
file_path = '/Users/heechankang/Desktop/pythonworkspace/pandas/data/5674-833/part2/read_csv_sample.csv'

# read_csv() 함수로 데이터프레임 변환, 변수 df1에 저장
df1 = pd.read_csv(file_path)
print(df1)

df2 = pd.read_csv(file_path, header = None)
df2

df3 = pd.read_csv(file_path, index_col=None)
df3

df4 = pd.read_csv(file_path, index_col='c0')
df4

###########

import pandas as pd

df1 = pd.read_excel('/Users/heechankang/Desktop/pythonworkspace/pandas/data/5674-833/part2/남북한발전전력량.xlsx')
df1 
df2 = pd.read_excel('/Users/heechankang/Desktop/pythonworkspace/pandas/data/5674-833/part2/남북한발전전력량.xlsx', header=None)
df2


############

import pandas as pd

df = pd.read_json('/Users/heechankang/Desktop/pythonworkspace/pandas/data/5674-833/part2/read_json_sample.json')
df

##############

import pandas as pd

url = '/Users/heechankang/Desktop/pythonworkspace/pandas/data/5674-833/part2/sample.html'

tables = pd.read_html(url)

print(len(tables))

for i in range(len(tables)):
    print("tables[{}]".format(i))
    print(tables[i])
    print()

df = tables[1]

df.set_index(['name'], inplace=True)
print(df)


#######################################

from bs4 import BeautifulSoup
import requests
import re
import pandas as pd

url = 'https://en.wikipedia.org/wiki/List_of_American_exchange-traded_funds'
resp = requests.get(url)
soup = BeautifulSoup(resp.text, 'lxml')
rows = soup.select('div>ul>li')

# print(url)
# print(resp)
# print(soup)
# print(rows)

etfs={}
for row in rows:

    try:
        etf_name = re.findall('^(.*) \(NYSE', row.text)
        etf_market = re.findall('\((.*)\|', row.text)
        etf_ticker = re.findall('NYSE Arca\|(.*)\)', row.text)

        if (len(etf_ticker) > 0) & (len(etf_market) > 0) & (len(etf_name) > 0):
            etfs[etf_ticker[0]] = [etf_market[0], etf_name[0]]

    except AttributeError as err:
        pass

# etfs 딕셔너리 출력    
print(etfs)
print()

# etfs 딕셔너리를 데이터프레임으로 변환
df = pd.DataFrame(etfs)
print(df)


for i in etfs:
    print("etfs[{0}]".format(i))
    print(etfs[i])
    print()


############################################

import pandas as pd

data = {'name':['Jerry', 'Riah', 'Paul'],
        'algol':['A', 'A+', 'B'],
        'basic':['C', 'B', 'B+'],
        'c++':['B+', 'C','C+']
        }

df = pd.DataFrame(data)
df.set_index('name', inplace = True)
print(df)

df.to_csv("./df_sample.csv")


###################


import pandas as pd

data = {'name':['Jerry', 'Riah', 'Paul'],
        'algol':['A', 'A+', 'B'],
        'basic':['C', 'B', 'B+'],
        'c++':['B+', 'C','C+']
        }

df = pd.DataFrame(data)
df.set_index('name', inplace = True)
print(df)

df.to_json('./df_sample.json')

########################


import pandas as pd

data = {'name':['Jerry', 'Riah', 'Paul'],
        'algol':['A', 'A+', 'B'],
        'basic':['C', 'B', 'B+'],
        'c++':['B+', 'C','C+']
        }

df = pd.DataFrame(data)
df.set_index('name', inplace = True)
print(df)

df.to_excel('./df_sample.xlsx')

################################


import pandas as pd

data1 = {'name':['Jerry', 'Riah', 'Paul'],
        'algol':['A', 'A+', 'B'],
        'basic':['C', 'B', 'B+'],
        'c++':['B+', 'C','C+']
        }

data2 = {'c0':[1,2,3],
         'c1':[4,5,6],
         'c2':[7,8,9],
         'c3':[10,11,12],
         'c4':[13,14,15]}

df1 = pd.DataFrame(data1)
df1.set_index('name', inplace=True)
df1

df2 = pd.DataFrame(data2)
df2.set_index('c0', inplace=True)
df2

writer = pd.ExcelWriter("./df_excelwriter.xlsx")
df1.to_excel(writer, sheet_name='sheet1')
df2.to_excel(writer, sheet_name='sheet2')
writer.save()