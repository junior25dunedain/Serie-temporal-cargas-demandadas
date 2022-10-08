import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

base = pd.read_csv('Carga_06022020_08032020.csv')
base = base.iloc[:,1:2].values

normalizador = MinMaxScaler(feature_range=(0,1))
base_normalizada = normalizador.fit_transform(base)

previsoes = []
target = []

for i in range(25,len(base)):
    previsoes.append(base_normalizada[i-24-1:i-1,0])
    target.append(base_normalizada[i-1:i,0])

previsoes, target = np.array(previsoes), np.array(target)

I_treinamento, I_teste, t_treinamento, t_teste = train_test_split(previsoes, target, test_size= 0.25, shuffle= False)

Previsor = Sequential()
Previsor.add(Dense(units=13, activation='relu', input_dim= 24))
Previsor.add(Dense(units=13, activation='relu'))
Previsor.add(Dense(units=1, activation='linear'))
Previsor.compile(optimizer='rmsprop', loss = 'mean_squared_error', metrics=['mean_absolute_error'])

Previsor.fit(I_treinamento, t_treinamento, epochs= 100)

resultados = Previsor.predict(I_teste)
resultados = normalizador.inverse_transform(resultados)
t_teste = normalizador.inverse_transform(t_teste)

plt.title('Carga ONS - Horário')
plt.ylabel('Carga (kW)')
plt.xlabel(u'Períodos (Horas)')
reg_val, = plt.plot(resultados, color='b',label= u'Previsão')
true_val, = plt.plot(t_teste, color='g', label='Valores Reais')
# plt.xlim([0,184])
plt.legend(handles=[true_val,reg_val])
plt.show()

def media_absoluta(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true-y_pred)/y_true)) *100

from sklearn.metrics import r2_score

r2 = r2_score(t_teste, resultados)
print('R2 = ',r2)
MAPE = media_absoluta(t_teste, resultados)
print('MAPE = ',MAPE)