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

plt.style.use('bmh')

#########################################################################
#                      Random Forest                                    #
#########################################################################

# Limpiar Terminal
os.system('cls')

n_estimators = [10, 50, 100,150,200,250,500]
val_scores = []
test_scores = []

for n in n_estimators:
    rf = RandomForestClassifier(n_estimators=n,
                                random_state=42,
                                n_jobs=-1,
                                max_depth=8,
                                min_samples_split=2,
                                )
    rf.fit(X_train, y_train)
    ValAcc=accuracy_score(Y_Val, rf.predict(X_Val))
    TestAcc=accuracy_score(y_test, rf.predict(X_test))
    print(f'Training Random Forest with {n} trees')
    print(f'Validation Accuracy: {ValAcc:.4f}')
    print(f'Test Accuracy: {TestAcc:.4f}')
    print('-----------------------------------')
    val_scores.append(ValAcc)
    test_scores.append(TestAcc)

# Visualiza los resultados
plt.plot(n_estimators, val_scores, label='Val Accuracy')
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
                            min_samples_split=2,
                        )
rf.fit(X_train, y_train)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
rf_pred_val = rf.predict(X_Val)
rf_cm = confusion_matrix(Y_Val, rf_pred_val)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ciclo 1', 'Ciclo 0'], yticklabels=['Cicle 1', 'Ciclo 0'])
plt.title('Random Forest Validation Confusion Matrix')
plt.subplot(1, 2, 2)
rf_pred = rf.predict(X_test)
rf_cm = confusion_matrix(y_test, rf_pred)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ciclo 1', 'Ciclo 0'], yticklabels=['Cicle 1', 'Ciclo 0'])
plt.title('Random Forest Test Confusion Matrix')
plt.show()

# plot ROC Curve

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
fpr, tpr, _ = roc_curve(Y_Val, rf_pred_val)
auc_score = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Validation')
plt.legend()
plt.subplot(1, 2, 2)
fpr, tpr, _ = roc_curve(y_test, rf_pred)
auc_score = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Test')
plt.legend()
plt.show()


#########################################################################
#                      Neural Network                                   #
#########################################################################

# Limpiar Terminal
os.system('cls')
# Convertir los datos a tensores de PyTorch
if torch.cuda.is_available():
    print('CUDA is available!  Training on GPU ...')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Convertir los datos a tensores de PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
X_Val = torch.tensor(X_Val, dtype=torch.float32).to(device)


y_train = y_train.values
y_test = y_test.values
Y_Val = Y_Val.values
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)
Y_Val = torch.tensor(Y_Val, dtype=torch.long).to(device)

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
optimizer = optim.Adam(model.parameters(), lr=0.00001)


# Entrenamiento de la red neuronal
num_epochs = 35000
train_losses = []
val_accs = []
test_accs = []

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
    val_accs.append(accuracy_score(Y_Val, model(X_Val).argmax(dim=1).cpu().numpy()))
    test_accs.append(accuracy_score(y_test, model(X_test).argmax(dim=1).cpu().numpy()))
    
    # Mostrar el progreso
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    if len(train_losses) > 1 and abs(train_losses[-1] - train_losses[-2]) < 1e-6:
        break

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
# Graficar la curva de pérdida
plt.plot(train_losses)
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(1, 2, 2)
# Graficar la precisión en el conjunto de validación
plt.plot(val_accs, label='Validation Accuracy')
plt.plot(test_accs, label='Test Accuracy')
plt.title('Validation Accuracy Curve')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Durante la evaluación:
model.eval()
with torch.no_grad():
    valOutputs = model(X_Val)
    testOutputs = model(X_test)
    nn_pred_Val = valOutputs.argmax(dim=1).cpu().numpy()
    nn_pred_test = testOutputs.argmax(dim=1).cpu().numpy()

# Matriz de Confusión
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
nn_cm = confusion_matrix(Y_Val, nn_pred_Val)
sns.heatmap(nn_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ciclo 1', 'Ciclo 0'], yticklabels=['Cicle 1', 'Ciclo 0'])
plt.title('Neural Network Validation Confusion Matrix')
plt.subplot(1, 2, 2)
nn_cm = confusion_matrix(y_test, nn_pred_test)
sns.heatmap(nn_cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ciclo 1', 'Ciclo 0'], yticklabels=['Cicle 1', 'Ciclo 0'])
plt.title('Neural Network Test Confusion Matrix')
plt.show()

# ROC Curve
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
nn_fpr, nn_tpr, _ = roc_curve(Y_Val, valOutputs[:, 1].cpu().numpy())
nn_auc = auc(nn_fpr, nn_tpr)
plt.plot(nn_fpr, nn_tpr, label=f'Neural Network (AUC = {nn_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Validation')
plt.legend()
plt.subplot(1, 2, 2)
nn_fpr, nn_tpr, _ = roc_curve(y_test, testOutputs[:, 1].cpu().numpy())
nn_auc = auc(nn_fpr, nn_tpr)
plt.plot(nn_fpr, nn_tpr, label=f'Neural Network (AUC = {nn_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Test')
plt.legend()
plt.show()
os.system('cls')

# Evaluación de los modelos
print('Random Forest Evaluation')
print('-----------------------')
print(f'Validation Accuracy: {accuracy_score(Y_Val, rf_pred_val):.2f}')
print(f'Test Accuracy: {accuracy_score(y_test, rf_pred):.2f}')
print(f'False Positive Rate: {fpr[1]:.2f}')
print(f'True Positive Rate: {tpr[1]:.2f}')
print(f'Validation AUC: {auc(fpr, tpr):.2f}')
print(f'Test AUC: {auc(fpr, tpr):.2f}')
print('-----------------------')

print('Neural Network Evaluation')
print('-----------------------')
print(f'Validation Accuracy: {accuracy_score(Y_Val, nn_pred_Val):.2f}')
print(f'Test Accuracy: {accuracy_score(y_test, nn_pred_test):.2f}')
print(f'False Positive Rate: {nn_fpr[1]:.2f}')
print(f'True Positive Rate: {nn_tpr[1]:.2f}')
print(f'Validation AUC: {nn_auc:.2f}')
print(f'Test AUC: {nn_auc:.2f}')
print('-----------------------')
