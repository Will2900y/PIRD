import warnings
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sn
from statsmodels.tsa.stattools import acf

#import data
               
df25 = pd.read_csv('/Users/belin/OneDrive/Documentos/Donnes_Pird/25-EC.csv',
                 parse_dates=['Date'], index_col=['Date'])
df25 = df25 ['2021-04-26 00:00:00' : '2021-05-31 00:00:00']

df34 = pd.read_csv('/Users/belin/OneDrive/Documentos/Donnes_Pird/34-EC.csv',
                 parse_dates=['Date'], index_col=['Date'])
df34 = df34 ['2021-04-26 00:00:00' : '2021-05-31 00:00:00']

df64 = pd.read_csv('/Users/belin/OneDrive/Documentos/Donnes_Pird/64-EC.csv',
                 parse_dates=['Date'], index_col=['Date'])
df64 = df64 ['2021-04-26 00:00:00' : '2021-05-31 00:00:00']

df25w = pd.read_csv('/Users/belin/OneDrive/Documentos/Donnes_Pird/25-IECS.csv',
                 parse_dates=['Date'], index_col=['Date'])
df25w = df25w ['2021-04-26 00:00:00' : '2021-05-31 00:00:00']

df34w = pd.read_csv('/Users/belin/OneDrive/Documentos/Donnes_Pird/34-IECS.csv',
                 parse_dates=['Date'], index_col=['Date'])
df34w = df34w ['2021-04-26 00:00:00' : '2021-05-31 00:00:00']

df64w = pd.read_csv('/Users/belin/OneDrive/Documentos/Donnes_Pird/64-IECS.csv',
                 parse_dates=['Date'], index_col=['Date'])
df64w = df64w ['2021-04-26 00:00:00' : '2021-05-31 00:00:00']

#Resample the data
#Energy
s25 = df25.Value.resample('D').mean() 
s34 = df34.Value.resample('D').mean() 
s64 = df64.Value.resample('D').mean() 

#DHW
s25w = df25w.Value.resample('D').mean() 
s34w = df34w.Value.resample('D').mean() 
s64w = df64w.Value.resample('D').mean() 



#correlation

matriz_corre = {'ap.25' : s25,
                'ap.34' : s34,
                'ap.64' : s64}
df_matriz = pd.DataFrame(matriz_corre)
df1_matriz = df_matriz.corr()
sn.heatmap(df1_matriz, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.title('Energy consumption correlation', fontsize=15)
plt.show()


matriz_corre_w = {'ap.25' : s25w,
                'ap.34' : s34w,
                'ap.64' : s64w}
df_matriz_w = pd.DataFrame(matriz_corre_w)
df1_matriz_w = df_matriz_w.corr()
sn.heatmap(df1_matriz_w, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.title('DHW consumption correlation', fontsize=15)
plt.show()

#plot


plt.figure(figsize=(15,6))
plt.plot(s25, label='ap.25', lw=3 ) #fix the legend of axe x
plt.plot(s34, label='ap.34', lw=3 )
plt.plot(s64, label='ap.64', lw=3 )
plt.xlabel('Date')
plt.ylabel('Power (W)')
plt.title('Energy consumption', fontsize=15)
plt.legend(loc='upper left', fontsize=20)
plt.show()


plt.figure(figsize=(15,6))
plt.plot(s25w, label='ap.25', lw=3 ) #fix the legend of axe x
plt.plot(s34w, label='ap.34', lw=3 )
plt.plot(s64w, label='ap.64', lw=3 )
plt.xlabel('Date')
plt.ylabel('Flow rate (l/s)')
plt.title('DHW consumption', fontsize=15)
plt.legend(loc='upper left', fontsize=20)
plt.show()


#Analisis for each ap.
s = s64w # chose the ap. of analysis and if is DHW or Power


#verify if is stationary - Dickey-Fuller
from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(s.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])


# Create Training and Test

train = s[:28]     # use 80% from the data
test = s[28:]


#AUTO ARIMA
#! pip install pmdarima
#import pmdarima as pm
#model_auto = pm.auto_arima(train, d=None, trace=False) #attention with d                    
#print(model_auto.summary())


from statsmodels.tsa.arima.model import ARIMA  
model_auto1 = ARIMA(train, order=(1, 0, 0)) # chose the order from AUTO # chose the order from AUTO
model_auto2 = ARIMA(s, order=(1, 0, 0))  # chose the order from AUTO # chose the order from AUTO
fitted_auto1 = model_auto1.fit()
fitted_auto2 = model_auto2.fit()
print(fitted_auto1.summary()) 
fc_auto = fitted_auto1.forecast(7, alpha=0.05)
#print (fc_auto)



# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training') 
plt.plot(test, label='test') 
plt.plot(fc_auto, label='forecast ARIMA') 
plt.title('Forecast vs training and testing')
plt.xlabel('Date')
plt.ylabel('Power (W)')
plt.legend(loc='upper left', fontsize=8)
plt.show()

# Model evaluation AUTO
# Accuracy metrics AUTO

def forecast_accuracy_auto (forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    mae = np.mean(np.abs(forecast - actual))    # MAE
    corr = (np.corrcoef(forecast, actual)[0,1]) * 100   # corr
    return({'mape':mape, 'mae': mae, 'corr(%)':corr})

Error_Arima = forecast_accuracy_auto (fc_auto, test.values)
print(Error_Arima)

#SARIMA

# Build SARIMA
#!pip3 install pyramid-arima
import pmdarima as pm

# Seasonal - fit stepwise auto-ARIMA
#smodel = pm.auto_arima(train,
                        # test='adf',
                        # max_p=3, max_q=3, m=24,
                         #start_P=0, seasonal=True,
                        # d=None, D=None, trace=True,
                        # error_action='ignore',  
                       # suppress_warnings=True, 
                        # stepwise=True)

#smodel.summary()

smodel1 = ARIMA(train, order=(0, 0, 0), seasonal_order = (0, 1, 0, 7))  # chose the order from AUTO # chose the order from AUTO
smodel2 = ARIMA(s, order=(0, 0, 0), seasonal_order = (0, 1, 0, 7))  # chose the order from AUTO # chose the order from AUTO
fitted_smodel1 = smodel1.fit()
fitted_smodel2 = smodel2.fit()
print(fitted_smodel1.summary())
fc_smodel = fitted_smodel1.forecast(7, alpha=0.05)  # 95% conf #mudar valor

# Plot

plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training') 
plt.plot(test, label='test') 
plt.plot(fc_smodel, label='forecast SARIMA') 
plt.title('Forecast vs training and testing')
plt.xlabel('Date')
plt.ylabel('Power (W)')
plt.legend(loc='upper left', fontsize=8)
plt.show()

def forecast_accuracy_SARIMA (forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    mae = np.mean(np.abs(forecast - actual))    # MAE
    corr = (np.corrcoef(forecast, actual)[0,1]) * 100   # corr           
    return({'mape':mape, 'mae': mae, 'corr(%)':corr})

Error_Sarima = forecast_accuracy_SARIMA (fc_smodel, test)
print(Error_Sarima)


#Exponential Smoothing
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

seasonal_decompose(s, model='additive').plot(resid=False,)
plt.xticks(rotation ='vertical')
plt.show()

from statsmodels.tsa.holtwinters import ExponentialSmoothing

model_ES = ExponentialSmoothing(train, trend='add', seasonal='add', 
                                seasonal_periods=9)                   
fitted_ES = model_ES.fit()
fc_ES = fitted_ES.forecast(7)
print (fc_ES)


# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training') 
plt.plot(test, label='test') 
plt.plot(fc_ES, label='forecast Exponential Smoothing') 
plt.title('Forecast vs training and testing')
plt.xlabel('Date')
plt.ylabel('Flow rate (l/s)')
plt.legend(loc='upper left', fontsize=8)
plt.show()

# Model evaluation ES
# Accuracy metrics ES

def forecast_accuracy_auto (forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    mae = np.mean(np.abs(forecast - actual))    # MAE
    corr = (np.corrcoef(forecast, actual)[0,1]) * 100   # corr
    return({'mape':mape, 'mae': mae, 'corr(%)':corr})

Error_ES = forecast_accuracy_auto (fc_ES, test)
print(Error_ES)


# Comparison plot models
import math
from collections import OrderedDict

linestyles_dict = OrderedDict(
    [('solid',               (0, ())),
     
     ('loosely dotted',      (0, (1, 10))),
     ('dotted',              (0, (1, 5))),
     ('densely dotted',      (0, (1, 1))),

     ('loosely dashed',      (0, (5, 10))),
     ('dashed',              (10, (12, 11))),
     ('densely dashed',      (0, (5, 1))),

     ('loosely dashdotted',  (0, (3, 10, 1, 10))),
     ('dashdotted',          (0, (3, 5, 1, 5))),
     ('densely dashdotted',  (0, (3, 1, 1, 1))),

     ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
     ('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
     ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])


plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training') 
plt.plot(test, label='test') 
plt.plot(fc_auto,color='red', 
         linestyle = linestyles_dict['solid'], 
         label='forecast ARIMA') 
plt.plot(fc_smodel, color='green', 
         linestyle = linestyles_dict['solid'], 
         label='forecast SARIMA') 
plt.plot(fc_ES, color='purple', 
         linestyle = linestyles_dict['solid'],
         label='forecast Exponential Smoothing') 
plt.title('Forecast vs training and testing')
plt.xlabel('Date') 
plt.ylabel('Flow rate (l/s)')
plt.legend(loc='upper left', fontsize=8)
plt.xticks(rotation ='vertical')
plt.show()


#Comparison Error

Error_Arima = pd.DataFrame(list(Error_Arima.items()),columns = ['Loss function','Value'])
Error_Sarima = pd.DataFrame(list(Error_Sarima.items()),columns = ['Loss function','Value'])
Error_ES = pd.DataFrame(list(Error_ES.items()),columns = ['Loss function','Value'])
Comp_A_S = Error_Arima[['Loss function','Value']].copy()

Comp_A_S.insert(1, 'Value Sarima', Error_Sarima.Value)
Comp_A_S.rename(columns = {'Value':'Value ARIMA'}, inplace = True)
Comp_A_S.insert(1, 'Value ES', Error_ES.Value)

print(Comp_A_S)

Comp_A_S.plot(kind = 'bar', x = 'Loss function')
plt.title('ARIMA X SARIMA X Exponential Smoothing', fontsize=15)
plt.legend(loc='upper right', fontsize=10)
plt.show()

