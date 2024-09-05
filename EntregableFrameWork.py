import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
import os

# Leer el archivo CSV
df = pd.read_csv('Data_Sets\Processed.csv') # Ocupamos el mismo dataset 
                                            # que ocupamos para nuestra implementación manual
# Shuffle the DataFrame rows
df = df.sample(frac=1).reset_index(drop=True)

# Replace infinite values with NaN
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Drop rows with NaN values
df.dropna(inplace=True)

# Normalizar los datos
scaler = StandardScaler()
X = df.drop('Class', axis=1)
X = scaler.fit_transform(X)
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_Val = X_train[:int(len(X_train)*0.99)]
Y_Val = y_train[:int(len(y_train)*0.99)]

#########################################################################
#                      Random Forest                                    #
#########################################################################

# Limpiar Terminal
os.system('cls')

n_estimators = [10, 50, 100,150,200,250,500]
train_scores = []
test_scores = []

for n in n_estimators:
    rf = RandomForestClassifier(n_estimators=n,
                                random_state=42,
                                n_jobs=-1,
                                max_depth=8,
                                )
    rf.fit(X_train, y_train)
    TrainAcc=accuracy_score(y_train, rf.predict(X_train))
    TestAcc=accuracy_score(y_test, rf.predict(X_test))
    print(f'Training Random Forest with {n} trees')
    print(f'Train Accuracy: {TrainAcc}')
    print(f'Test Accuracy: {TestAcc}')
    print('-----------------------------------')
    train_scores.append(TrainAcc)
    test_scores.append(TestAcc)

# Visualiza los resultados
plt.plot(n_estimators, train_scores, label='Train Accuracy')
plt.plot(n_estimators, test_scores, label='Test Accuracy')
plt.xlabel('Number of Trees')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
# Mostrar el mejor modelo
best_n = n_estimators[np.argmax(test_scores)]
print(f'Best Random Forest with {best_n} trees')
# Matiz de Confusión del mejor modelo
rf = RandomForestClassifier(n_estimators=best_n,
                            random_state=42,
                            n_jobs=-1,
                            max_depth=8,
                        )
rf.fit(X_train, y_train)

y_pred = rf.predict(X_Val)
rf_cm = confusion_matrix(Y_Val, y_pred)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion MatrixTrain')
roc=roc_curve(Y_Val, y_pred)
plt.show()
#########################################################################
#                      Neural Network                                   #
#########################################################################

# Limpiar Terminal
os.system('cls')
# Convertir los DF a numpy arrays

# Convertir los datos a tensores de PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = y_train.values
y_test = y_test.values
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_train, X_test, y_train, y_test = X_train.to(device), X_test.to(device), y_train.to(device), y_test.to(device)

# Definir la red neuronal
input_size = X_train.shape[1]
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)  # ajustar dinámicamente el tamaño de entrada
        self.fc2 = nn.Linear(50, 2)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Inicializar el modelo, el optimizador y la función de pérdida
model = SimpleNN()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# Entrenamiento de la red neuronal
num_epochs = 35000
train_losses = []

for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass y optimización
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Guardar la pérdida para graficarla
    train_losses.append(loss.item())
    
    # Mostrar el progreso
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    if len(train_losses) > 1 and abs(train_losses[-1] - train_losses[-2]) < 1e-6:
        break

# Graficar la curva de pérdida
plt.plot(train_losses)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Durante la evaluación:
model.eval()
with torch.no_grad():
    outputs = model(X_test)
    _, nn_pred = torch.max(outputs, 1)  # Obtener las predicciones de la red neuronal
    nn_pred = nn_pred.cpu().numpy()     # Convertir a NumPy array
#########################################################################
#                      Evaluación de los modelos                        #
#########################################################################
# Evaluación usando Sklearn
rf_pred = rf.predict(X_test.cpu().numpy())  # Random Forest predicciones
y_test_np = y_test.cpu().numpy()            # Convertir a numpy array para compatibilidad

rf_accuracy = accuracy_score(y_test_np, rf_pred)
nn_accuracy = accuracy_score(y_test_np, nn_pred)

print(f'Random Forest Accuracy: {rf_accuracy}')
print(f'Neural Network Accuracy: {nn_accuracy}')

# Matriz de Confusión
rf_cm = confusion_matrix(y_test, rf_pred)
nn_cm = confusion_matrix(y_test, nn_pred)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')

plt.subplot(1, 2, 2)
sns.heatmap(nn_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Neural Network Confusion Matrix')

plt.show()

# Curva ROC y AUC
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_pred)

rf_auc = auc(rf_fpr, rf_tpr)
nn_auc = auc(nn_fpr, nn_tpr)

plt.figure(figsize=(10, 5))
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot(nn_fpr, nn_tpr, label=f'Neural Network (AUC = {nn_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
