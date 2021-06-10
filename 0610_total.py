import pandas as pd
from statsmodels import datasets
import matplotlib.pyplot as plt
import statsmodels.api as sm
raw_set = datasets.get_rdataset('co2', package = 'datasets')
raw = raw_set.data

raw_set = datasets.get_rdataset('co2', package = 'datasets')
raw = raw_set.data

type(raw_set)
raw = raw_set.data

raw.head()
raw.info()
raw.shape
raw.describe()

plt.bar(raw.time, raw.value)
plt.plot(raw.time, raw.value)

result = sm.OLS.from_formula(formula='value~time', data=raw).fit()
result.summary()

result.params

trend = result.params[0] + result.params[1] * raw.time
plt.plot(raw.time, raw.value, raw.time, trend)
plt.show()

result = sm.OLS.from_formula(formula='value~time+I(time**2)', data=raw).fit()
result.summary()

result.params

trend = result.params[0] + result.params[1] * raw.time + result.params[2]*raw.time**2
plt.plot(raw.time, raw.value, raw.time, trend)
plt.show()

result.resid

plt.plot(raw.time, result.resid)

plt.figure(figsize=(10, 10))
trend = result.params[0] + result.params[1] * raw.time + result.params[2]*raw.time**2 + result.resid
plt.plot(raw.time, raw.value, raw.time, trend)
plt.show()


################
# adf, kpss
def stationarity_adf_test(Y_Data, Target_name):
    if len(Target_name) == 0:
        Stationarity_adf = pd.Series(sm.tsa.stattools.adfuller(Y_Data)[0:4],
                                     index=['Test Statistics', 'p-value', 'Used Lag', 'Used Observations'])
        for key, value in sm.tsa.stattools.adfuller(Y_Data)[4].items():
            Stationarity_adf['Critical Value(%s)'%key] = value
            Stationarity_adf['Maximum Information Criteria'] = sm.tsa.stattools.adfuller(Y_Data)[5]
            Stationarity_adf = pd.DataFrame(Stationarity_adf, columns=['Stationarity_adf'])
    else:
        Stationarity_adf = pd.Series(sm.tsa.stattools.adfuller(Y_Data[Target_name])[0:4],
                                     index=['Test Statistics', 'p-value', 'Used Lag', 'Used Observations'])
        for key, value in sm.tsa.stattools.adfuller(Y_Data[Target_name])[4].items():
            Stationarity_adf['Critical Value(%s)'%key] = value
            Stationarity_adf['Maximum Information Criteria'] = sm.tsa.stattools.adfuller(Y_Data[Target_name])[5]
            Stationarity_adf = pd.DataFrame(Stationarity_adf, columns=['Stationarity_adf'])
    return Stationarity_adf

def stationarity_kpss_test(Y_Data, Target_name):
    if len(Target_name) == 0:
        Stationarity_kpss = pd.Series(sm.tsa.stattools.kpss(Y_Data)[0:3],
                                      index=['Test Statistics', 'p-value', 'Used Lag'])
        for key, value in sm.tsa.stattools.kpss(Y_Data)[3].items():
            Stationarity_kpss['Critical Value(%s)'%key] = value
            Stationarity_kpss = pd.DataFrame(Stationarity_kpss, columns=['Stationarity_kpss'])
    else:
        Stationarity_kpss = pd.Series(sm.tsa.stattools.kpss(Y_Data[Target_name])[0:3],
                                      index=['Test Statistics', 'p-value', 'Used Lag'])
        for key, value in sm.tsa.stattools.kpss(Y_Data[Target_name])[3].items():
            Stationarity_kpss['Critical Value(%s)'%key] = value
            Stationarity_kpss = pd.DataFrame(Stationarity_kpss, columns=['Stationarity_kpss'])
    return Stationarity_kpss

'''
ADF, KPSS : 정상성 확인하는 테스트 방법
ADF : 추세 제거된것 확인
KPSS : 계절성 제거된것 확인
ADF 검증조건 : p-value < 0.05 => 정상
KPSS 검증조건 : p-value > 0.05 => 정상
'''
stationarity_adf_test(result.resid,[])


stationarity_kpss_test(result.resid,[])


'''
ACF(Autocorrelation Function, 자기상관함수)
'''
sm.graphics.tsa.plot_acf(result.resid, lags = 10)
plt.show()


raw.value
data = raw.value - raw.value.shift(1)
data
plt.plot(data)

raw.value.diff(1)
plt.plot(raw.value.diff(1))
plt.plot(raw.value.diff(30))

plt.plot(raw.time[1:], raw.value.diff(1).dropna())
plt.show()

display(stationarity_adf_test(raw.value.diff(12).dropna(),[]))
display(stationarity_kpss_test(raw.value.diff(12).dropna(),[]))
sm.graphics.tsa.plot_acf(raw.value.diff(12).dropna(), lags = 100)
plt.show()

stationarity_adf_test(raw.value.diff(12).dropna(),[])
stationarity_kpss_test(raw.value.diff(12).dropna(),[])
sm.graphics.tsa.plot_acf(raw.value.diff(12).dropna(), lags = 100)
plt.show()


#####
# 계절성 제거
###
# 호흡기 질환 사망 데이터

'''
참고 블로그 : https://emilkwak.github.io/python-toy-datasets
rdatasets https://vincentarelbundock.github.io/Rdatasets/articles/data.html
'''
raw_set = datasets.get_rdataset('deaths', package='MASS')
raw = raw_set.data
raw

plt.plot(raw.time, raw.value)
plt.plot(raw)

plt.plot(raw.value.diff(1))


stationarity_adf_test(raw.value,[])
stationarity_kpss_test(raw.value,[])
sm.graphics.tsa.plot_acf(raw.value, lags = 50, title='ACF')
plt.show()

stationarity_adf_test(raw.value.diff(12).dropna(),[])
stationarity_kpss_test(raw.value.diff(12).dropna(),[])
sm.graphics.tsa.plot_acf(raw.value.diff(12).dropna(), lags = 50, title='ACF')
plt.show()

raw.head()
raw.time = pd.date_range('1974-01-01', periods = len(raw), freq = 'M')
raw['month'] = raw.time.dt.month
raw['week'] = raw.time.dt.week
raw.drop('week', axis = 1, inplace = True)
raw


result = sm.OLS.from_formula(formula='value~C(month)-1', data=raw).fit()
display(result.summary())

plt.plot(raw.time, raw.value, raw.time, result.fittedvalues)
plt.show()

plt.plot(raw.time, result.resid)
plt.show()

stationarity_adf_test(result.resid,[])
stationarity_kpss_test(result.resid,[])
sm.graphics.tsa.plot_acf(result.resid, lags = 50)

plt.plot(raw.time, raw.value)
plt.title('Raw')
plt.show()

seasonal_lag = 3
plt.plot(raw.time[seasonal_lag:], raw.value.diff(seasonal_lag).dropna(), label='Lag{}'.format(seasonal_lag))
seasonal_lag = 6
plt.plot(raw.time[seasonal_lag:], raw.value.diff(seasonal_lag).dropna(), label='Lag{}'.format(seasonal_lag))
seasonal_lag = 12
plt.plot(raw.time[seasonal_lag:], raw.value.diff(seasonal_lag).dropna(), label='Lag{}'.format(seasonal_lag))
plt.title('Lagged')
plt.legend()
plt.show()

size_graph = 5
for seasonal_lag in range(1, 13):   
    fig, axes = plt.subplots(1, 3figsize = (12, 12))
    plt.plot(raw.time[seasonal_lag:], raw.value.diff(seasonal_lag).dropna(), label='Lag{}'.format(seasonal_lag))
    axseasonal_lag = fig.add_subplot(3,4,seasonal_lag) 
    plt.title('Lagged')
    plt.legend()
    plt.show()

for idx_x in range(1, 13, 3):
    fig, axes = plt.subplots(1, 3, figsize=(25, 3))
    axes[0].plot(raw.time[idx_x:], raw.value.diff(idx_x).dropna(), label='Lag{}'.format(idx_x))
    axes[0].grid()
    
    axes[1].plot(raw.time[idx_x+1:], raw.value.diff(idx_x+1).dropna(), label='Lag{}'.format(idx_x+1))
    axes[1].grid()

    axes[2].plot(raw.time[idx_x+2:], raw.value.diff(idx_x+2).dropna(), label='Lag{}'.format(idx_x+2))
    axes[2].grid()