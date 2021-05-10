import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.support.select import Select

result = requests.get('http://www.seoul.go.kr/coronaV/coronaStatus.do')
print(result.text)
DRIVER_PATH = '/Users/heechankang/projects/pythonworkspace/dr.woo_deep_learning/1-1.crawling/chromedriver'
driver = webdriver.Chrome(executable_path=DRIVER_PATH) # DRIVER_PATH에 있는 크롬드라이버를 켜라는건가?
driver.get('http://www.seoul.go.kr/coronaV/coronaStatus.do')

title = driver.find_element_by_xpath('//*[@id="DataTables_Table_0"]/tbody')
print('제목: ' + title.text)
data = title.text


driver.close()

data2
data2 = data[:]
dt = data2.replace('\n', ' ').split(' ')
data2.split(' ')
type(data)

dt3
for i in dt2:
    if i!='소재':
        dt3.append(i)
dt2

len(dt3)

for i in range(666):
    if i%6

real_data = {}
