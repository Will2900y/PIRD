import warnings
warnings.filterwarnings('ignore')

import numpy as np, pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
import seaborn as sn
from statsmodels.tsa.stattools import acf

#import data
               
df25 = pd.read_csv('./data_csv/25-EC.csv',
                 parse_dates=['Date'], index_col=['Date'])
df25 = df25 ['2021-01-26 00:00:00' : '2021-01-31 00:00:00']

df34 = pd.read_csv('/Users/wisar/OneDrive/Escritorio/GCU - INSA Lyon/5 annee/S9/PIRD/Ibrahim/data_csv/34-EC.csv',
                 parse_dates=['Date'], index_col=['Date'])
df34 = df34 ['2021-01-26 00:00:00' : '2021-01-31 00:00:00']

df64 = pd.read_csv('/Users/wisar/OneDrive/Escritorio/GCU - INSA Lyon/5 annee/S9/PIRD/Ibrahim/data_csv/64-EC.csv',
                 parse_dates=['Date'], index_col=['Date'])
df64 = df64 ['2021-01-26 00:00:00' : '2021-01-31 00:00:00']

df25w = pd.read_csv('/Users/wisar/OneDrive/Escritorio/GCU - INSA Lyon/5 annee/S9/PIRD/Ibrahim/data_csv/25-IECS.csv',
                 parse_dates=['Date'], index_col=['Date'])
df25w = df25w ['2021-01-26 00:00:00' : '2021-01-31 00:00:00']

df34w = pd.read_csv('/Users/wisar/OneDrive/Escritorio/GCU - INSA Lyon/5 annee/S9/PIRD/Ibrahim/data_csv/34-IECS.csv',
                 parse_dates=['Date'], index_col=['Date'])
df34w = df34w ['2021-01-26 00:00:00' : '2021-01-31 00:00:00']

df64w = pd.read_csv('/Users/wisar/OneDrive/Escritorio/GCU - INSA Lyon/5 annee/S9/PIRD/Ibrahim/data_csv/64-IECS.csv',
                 parse_dates=['Date'], index_col=['Date'])
df64w = df64w ['2021-01-26 00:00:00' : '2021-01-31 00:00:00']

#Resample the data
#Energy
s25 = df25.Value.resample('H').mean() 
s34 = df34.Value.resample('H').mean() 
s64 = df64.Value.resample('H').mean() 

#DHW
s25w = df25w.Value.resample('H').mean() 
s34w = df34w.Value.resample('H').mean() 
s64w = df64w.Value.resample('H').mean() 



#correlation

matriz_corre = {'ap.25' : s25,
                'ap.34' : s34,
                'ap.64' : s64}
df_matriz = pd.DataFrame(matriz_corre)
df1_matriz = df_matriz.corr()
sn.heatmap(df1_matriz, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
plt.show()


matriz_corre_w = {'ap.25' : s25w,
                'ap.34' : s34w,
                'ap.64' : s64w}
df_matriz_w = pd.DataFrame(matriz_corre_w)
df1_matriz_w = df_matriz_w.corr()
sn.heatmap(df1_matriz_w, annot=True, vmax=1, vmin=-1, center=0, cmap='vlag')
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


#Analisis for each app.
s = s25w # chose the app of analysis and if if DHW or Power


#verify if is stationary - Dickey-Fuller
from statsmodels.tsa.stattools import adfuller
from numpy import log
result = adfuller(s.dropna())
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])

# Create Training and Test

train = s[:96]     # use 80% from the data
test = s[96:]


#AUTO ARIMA
! pip install pmdarima
import pmdarima as pm
model_auto = pm.auto_arima(train, d=1, trace=True) #attention with d                    
print(model_auto.summary())


from statsmodels.tsa.arima.model import ARIMA  
model_auto1 = ARIMA(train, order=(0, 1, 2)) # chose the order from AUTO # chose the order from AUTO
model_auto2 = ARIMA(s, order=(0, 1, 2))  # chose the order from AUTO # chose the order from AUTO
fitted_auto1 = model_auto1.fit()
fitted_auto2 = model_auto2.fit()
print(fitted_auto1.summary()) 
fc_auto = fitted_auto1.forecast(24, alpha=0.05)
#print (fc_auto)



# Plot
plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training') 
plt.plot(test, label='test') 
plt.plot(fc_auto, label='forecast ARIMA') 
plt.title('Forecast vs training and testing')
plt.xlabel('Date')
plt.ylabel('Flow rate (l/s)')
plt.legend(loc='upper left', fontsize=8)
plt.show()

# Model evaluation AUTO
# Accuracy metrics AUTO

def forecast_accuracy_auto (forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc_auto-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy_auto (fc_auto, test.values)


#SARIMA

# Build SARIMA
!pip3 install pyramid-arima
import pmdarima as pm

# Seasonal - fit stepwise auto-ARIMA
smodel = pm.auto_arima(train,
                         test='adf',
                         max_p=3, max_q=3, m=24,
                         start_P=0, seasonal=True,
                         d=0, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)

smodel.summary()

smodel1 = ARIMA(train, order=(0, 0, 0), seasonal_order = (0, 1, 0, 24))  # chose the order from AUTO # chose the order from AUTO
smodel2 = ARIMA(s, order=(0, 0, 0), seasonal_order = (0, 1, 0, 24))  # chose the order from AUTO # chose the order from AUTO
fitted_smodel1 = smodel1.fit()
fitted_smodel2 = smodel2.fit()
print(fitted_smodel1.summary())
fc_smodel = fitted_smodel1.forecast(24, alpha=0.05)  # 95% conf #mudar valor

# Plot

plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training') 
plt.plot(test, label='test') 
plt.plot(fc_smodel, label='forecast SARIMA') 
plt.title('Forecast vs training and testing')
plt.xlabel('Date')
plt.ylabel('Flow rate (l/s)')
plt.legend(loc='upper left', fontsize=8)
plt.show()

def forecast_accuracy_SARIMA (forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None], 
                              actual[:,None]]), axis=1)
    minmax = 1 - np.mean(mins/maxs)             # minmax
    acf1 = acf(fc_smodel-test)[1]                      # ACF1
    return({'mape':mape, 'me':me, 'mae': mae, 
            'mpe': mpe, 'rmse':rmse, 'acf1':acf1, 
            'corr':corr, 'minmax':minmax})

forecast_accuracy_SARIMA (fc_smodel, test)
