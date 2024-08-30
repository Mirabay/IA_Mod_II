import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import sys
import os
os.system('cls' if os.name == 'nt' else 'clear')

import warnings
#########################################################################
#                               ETL                                     #
#########################################################################
df_mixed = pd.read_csv('Data_Sets\Delicates.csv')
df_cottons = pd.read_csv('Data_Sets\Jeans.csv')

# n = 120 # los primeros 3 minutos no se consideran
# df_mixed = df_mixed[df_mixed['Time(min)']>n]
# df_cottons = df_cottons[df_cottons['Time(min)']>n]

# El peso es un factor importante en el comportamiento
# de la secadora.
w1 = 2
w2 = 20
# incluir peso 1 y peso 2
df_mixed = df_mixed[(df_mixed['Weight']>=w1) & (df_mixed['Weight']<=w2)]
df_cottons = df_cottons[(df_cottons['Weight']>=w1) & (df_cottons['Weight']<=w2)]

df_mixed['Class']= 0 # 0 para mixed
df_cottons['Class']= 1 # 1 para cottons


# Concatenar los dataframes para poder hacer la clasificacion
df = pd.concat([df_mixed, df_cottons], axis=0)

df = df.drop(columns=['Weight','Segment','RDB','Restriction','Energy','Potenza','Smooth','T-A amb','RH amb'])

min_class = df['Class'].value_counts().min()
df = df.groupby('Class').apply(lambda x: x.sample(min_class)).reset_index(drop=True)
print(df['Class'].value_counts())
print('\n\n')


# Agregar el diferencial de Inlet
df['Delta T in'] = (df['Inlet'] - df['Inlet'].shift(1))**2
# Agregar el diferencial de Outlet
df['Delta T out'] = (df['Outlet'] - df['Outlet'].shift(1))**2

# Agregar el cuadrado
df['Inlet']= df['Inlet']**2
df['Outlet']= df['Outlet']**2

# Filtered en porcentaje
df['Filtered'] = df['Filtered']/ 100
# Agregar RMC/Filtered
df['RMC/Filtered'] = (df['RMC']/df['Filtered'])**2

# Shuffle 
df = df.sample(frac=1).reset_index(drop=True)

df.to_csv('Data_Sets\Processed.csv', index=False)

#########################################################################
#                      Regresion Logistica                              #
#########################################################################
# Ignorar todas las advertencias
warnings.filterwarnings('ignore')

# Ignorar todas las advertencias
warnings.filterwarnings('ignore')

# Ignorar solo advertencias específicas
warnings.filterwarnings('ignore', category=UserWarning)  # Ejemplo para ignorar UserWarnings


__errors__ = []



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
# Función de costo
def CostFunction(params, samples, bias, y): # Función de costo
    global __errors__
    hyp = h(params, samples, bias)
    hyp = np.clip(hyp, 1e-10, 1 - 1e-10)  # Evitar log(0)
    loss = -np.mean(y * np.log(hyp) + (1 - y) * np.log(1 - hyp)) # Función de costo logarítmica llamada entropía cruzada
    __errors__.append(loss)
    return loss 
# Validación
def validate(params, bias, samples, y):
    hyp = h(params, samples, bias)
    hyp = hyp.round()
    loss = np.mean(hyp == y)
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

    val_samples = samples[:int(len(samples)*0.8)]
    val_y = y[:int(len(y)*0.8)]
    for epoch in range(int(epochs)):
        # Actualizar parámetros
        params, bias = Gradiente_Descendiente(params, bias, samples, y, learning_rate)
        # Mostrar errores
        error = CostFunction(params, samples, bias, y)
        val_error = validate(params, bias, val_samples, val_y)
        sys.stdout.write(f'\rEpoch: {(epoch/epochs)*100:.2f}%, Loss: {error*100:.2f}%, Validation Error: {100-val_error*100:.2f}%')
        sys.stdout.flush()
        time.sleep(1e-99)  # Simul1ear tiempo de entrenamientoe
        # Condición para detener el entrenamiento
        if len(__errors__) > 1 and abs(__errors__[-1] - __errors__[-2]) < 1e-5 :
            break
    # Graficar errores
    plt.plot(range(len(__errors__)), __errors__)
    plt.plot(range(len(__errors__)), [0.5] * len(__errors__), '--r')
    plt.title('Error vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Error')
    plt.show()
    print(f'\nParams: {params} + Bias: {bias:.2f}')
    print(f'Error: {error * 100:.1f}%')
    return params, bias

def test(params, bias, samples, y):
    samples = scale_features(samples)
    hyp = h(params, samples, bias)
    accuracy = np.mean(hyp.round() == y)
    print(f'Accuracy: {accuracy * 100:.1f}%')
    plt.figure(figsize=(10, 6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.scatter(range(len(y)), hyp, label='Predictions', color='blue', alpha=0.25)
    ax1.scatter(range(len(y)), y, label='Real Values', color='red', alpha=0.25)
    ax2.scatter(range(len(y)), hyp.round(), label='Predictions', color='blue', alpha=0.25)
    ax2.scatter(range(len(y)), y, label='Real Values', color='red', alpha=0.25)
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
    plt.show()
    return hyp

def confusionMat(y, y_pred):
    TP = np.sum((y == 1) & (y_pred == 1))
    TN = np.sum((y == 0) & (y_pred == 0))
    FP = np.sum((y == 0) & (y_pred == 1))
    FN = np.sum((y == 1) & (y_pred == 0))
    
    # Graficar matriz de confusión usando seaborn
    sns.heatmap([[TP, FP], [FN, TN]], annot=True, fmt='d', cmap='Blues', xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    return TP, TN, FP, FN
def roc_curve(y, y_pred):
    fpr = []
    tpr = []
    for threshold in np.linspace(0, 1, 100):
        y_pred_round = y_pred > threshold
        TP = np.sum((y == 1) & (y_pred_round == 1))
        TN = np.sum((y == 0) & (y_pred_round == 0))
        FP = np.sum((y == 0) & (y_pred_round == 1))
        FN = np.sum((y == 1) & (y_pred_round == 0))
        fpr.append(FP / (FP + TN))
        tpr.append(TP / (TP + FN))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], '--r')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    return fpr, tpr
#########################################################################
#                      Pruebas                                          #
#########################################################################
# Leer el archivo CSV
df = pd.read_csv('Data_Sets\Processed.csv')
# Shuffle the DataFrame rows
df = df.sample(frac=1).reset_index(drop=True) 

df_x = df.drop('Class', axis=1)
df_x = scale_features(df_x)
df_y = df['Class']

df_x_train = df_x.iloc[:int(len(df_x)*0.99)]
df_y_train = df_y.iloc[:int(len(df_y)*0.99)]

df_x_test = df_x.iloc[int(len(df_x)*0.99):]
df_y_test = df_y.iloc[int(len(df_y)*0.99):]
# Inicializar parámetros
params = np.random.randn(df_x_train.shape[1])
bias = np.random.rand()
samples = df_x_train
y = df_y_train

learning_rate = 1e-1
epochs = 1e4

params, bias = train(params, bias, samples, y, learning_rate, epochs)

predictions = test(params, bias, df_x_test, df_y_test)
print(confusionMat(df_y_test ,predictions.round()))
roc_curve(df_y_test, predictions)

# Guardar predicciones
df_x_test['Class'] = df_y_test
df_x_test['Predictions'] = predictions

df_x_test.to_csv('Data_Sets/Predictions.csv', index=False)
