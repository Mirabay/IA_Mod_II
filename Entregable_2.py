import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
os.system('cls' if os.name == 'nt' else 'clear')

import warnings

# Ignorar todas las advertencias
warnings.filterwarnings('ignore')

# Ignorar solo advertencias específicas
warnings.filterwarnings('ignore', category=UserWarning)  # Ejemplo para ignorar UserWarnings


__errors__ = []

# Leer el archivo CSV
df = pd.read_csv('Proyecto_II/Data_Sets/Processed.csv')
# Shuffle the DataFrame rows
df = df.sample(frac=1).reset_index(drop=True) 

df_x = df.drop('Class', axis=1)
df_y = df['Class']

df_x_train = df_x.iloc[:int(len(df_x)*0.999)]
df_y_train = df_y.iloc[:int(len(df_y)*0.999)]

df_x_test = df_x.iloc[int(len(df_x)*0.999):]
df_y_test = df_y.iloc[int(len(df_y)*0.999):]

# Escalado de las características utilizando z-score
def scale_features(df):
    scaled_df = (df - df.mean()) / df.std(ddof=0)
    scaled_df = scaled_df.replace([np.inf, -np.inf], 0)  # Reemplazar infinitos por 0
    scaled_df = scaled_df.fillna(0)  # Reemplazar NaN por 0
    return scaled_df


# Función de hipótesis
def h(params, sample, bias):
    z = np.dot(sample, params) + bias  # Producto punto
    hyp = 1 / (1 + np.exp(-z))
    return hyp

# Mostrar errores
def show_errors(params, samples, bias, y):
    global __errors__
    
    hyp = h(params, samples, bias)
    hyp = np.clip(hyp, 1e-10, 1 - 1e-10)  # Evitar log(0)
    
    loss = -np.mean(y * np.log(hyp) + (1 - y) * np.log(1 - hyp))

    __errors__.append(loss)
    return loss 
    
    


# Gradiente Descendiente
def Gradiente_Descendiente(params, bias, samples, y, alfa):
    hyp = h(params, samples, bias)
    error = hyp - y
    gradient = np.dot(samples.T, error) / len(samples)
    params -= alfa * gradient  # Actualización de parámetros
    bias -= alfa * np.mean(error)  # Actualización del bias
    return params, bias

# Entrenamiento
def train(params, bias, samples, y, learning_rate, epochs):
    global __errors__
    samples = scale_features(samples)
    for epoch in range(int(epochs)):
        # Actualizar parámetros
        params, bias = Gradiente_Descendiente(params, bias, samples, y, learning_rate)
        # Mostrar errores
        error = show_errors(params, samples, bias, y)
        sys.stdout.write(f'\rEpoch: {(epoch/epochs)*100:.2f}%, Error: {error*100:.2f}%')
        sys.stdout.flush()
        time.sleep(0.0001)  # Simular tiempo de entrenamiento
        # Condición para detener el entrenamiento
        if len(__errors__) > 1 and abs(__errors__[-1] - __errors__[-2]) < 1e-7 :
            break
    # Graficar errores
    plt.plot(range(len(__errors__)), __errors__)
    plt.title('Error vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.show()
    print(f'Params: {params} + Bias: {bias:.2f}')
    print(f'Error: {error * 100:.1f}%')
    return params, bias

def test(params, bias, samples, y):
    samples = scale_features(samples)
    hyp = h(params, samples, bias)
    
    accuracy = np.mean(hyp.round() == y)
    print(f'Accuracy: {accuracy * 100:.1f}%')
    
    return hyp

# Inicializar parámetros
params = np.random.randn(df_x_train.shape[1])
bias = np.random.rand()
samples = df_x_train
y = df_y_train

learning_rate = 1e-1
epochs = 1e4

params, bias = train(params, bias, samples, y, learning_rate, epochs)
predictions = test(params, bias, df_x_test, df_y_test)

# Guardar predicciones
df_x_test['Class'] = df_y_test
df_x_test['Predictions'] = predictions

df_x_test.to_csv('Proyecto_II/Data_Sets/Predictions.csv', index=False)
# Plot predictions vs real values con más detalles
plt.figure(figsize=(10, 6))

# Graficar valores predichos
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Primera gráfica en el primer subplot
ax1.scatter(range(len(df_y_test)), df_x_test['Predictions'], label='Predictions', color='blue', alpha=0.5)
ax1.scatter(range(len(df_y_test)), df_y_test, label='Real Values', color='red', alpha=0.5)

ax2.scatter(range(len(df_y_test)), df_x_test['Predictions'].round(), label='Predictions', color='blue', alpha=0.25)
ax2.scatter(range(len(df_y_test)), df_y_test, label='Real Values', color='red', alpha=0.5)

ax1.set_title('Predictions vs Real_Values')
ax2.set_title('Round Prediction vs Real Values')
ax1.set_xlabel('Index')
ax1.set_ylabel('Class')
ax2.set_xlabel('Index')
ax2.set_ylabel('Class')

ax1.grid()
ax2.grid()

ax1.legend()
ax2.legend()

# Mostrar la gráfica
plt.show()