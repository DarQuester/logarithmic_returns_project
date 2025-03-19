import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy import stats
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import statsmodels
from math import sqrt, log, exp, pi
from random import uniform



tabl = open('dataBTc.csv')
df = pd.read_csv(tabl)
df.info()

data = df.iloc[::4]
data.head()

array = data.to_numpy()
arr = array[:, 1]
print(arr)

price = [0]

for i in range (1, len(arr)):
  s = arr[i]-arr[i-1]
  price.append(s)

print('Объём выборки:', len(price))
res1 = stats.ttest_1samp(price, 0)
print('Одновыборочный тест Стьюдента: statistic =', res1.statistic, ' p_value =', res1.pvalue)
res2 = stats.normaltest(price)
print('Тест Д`Агостино и Пиросона: statistic =', res2.statistic, ' p_value =', res2.pvalue)
res3 = stats.shapiro(price)
print('Тест Шапиро-Уилка: statistic =', res3.statistic, ' p_value =', res3.pvalue)

log_d = [0]

for i in range (1, len(arr)):
  s = np.log(arr[i]/arr[i-1])
  log_d.append(s)

print('Объём выборки:', len(log_d))
res1 = stats.ttest_1samp(price, 0)
print('Одновыборочный тест Стьюдента: statistic =', res1.statistic, ' p_value =', res1.pvalue)
res2 = stats.normaltest(log_d)
print('Тест Д`Агостино и Пиросона: statistic =', res2.statistic, ' p_value =', res2.pvalue)
res3 = stats.shapiro(log_d)
print('Тест Шапиро-Уилка: statistic =', res3.statistic, ' p_value =', res3.pvalue)

