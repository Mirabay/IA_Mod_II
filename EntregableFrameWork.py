import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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

#########################################################################
#                      Random Forest                                    #
#########################################################################

# Create and train the Random Forest model
rf = RandomForestClassifier(
                            n_estimators=500, # Number of trees in the forest
                            random_state=42,
                            n_jobs=-1, # Use all processors, -1 means use all processors, 1 means use 1 processor and None means use 1 processor
                            max_depth=5, # Maximum depth of the tree
                            min_samples_leaf=6, # Minimum number of samples required to be at a leaf node
                            min_samples_split=3 # Minimum number of samples required to split an internal node
                            )
rf.fit(X_train, y_train)
#########################################################################
#                      Neural Network                                   #
#########################################################################

# Create and train the Neural Network model
nn = MLPClassifier(
                   hidden_layer_sizes=(100, 100), # Number of neurons in the ith hidden layer
                   max_iter=1000, # Maximum number of iterations
                   alpha=0.0001, # L2 penalty (regularization term) parameter
                   solver='adam', # Optimization algorithm
                   random_state=42 # Seed for random number generator
                   )
nn.fit(X_train, y_train)

#########################################################################
#                      Evaluación de los modelos                        #
#########################################################################
rf_pred = rf.predict(X_test)
nn_pred = nn.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)
nn_accuracy = accuracy_score(y_test, nn_pred)

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
nn_fpr, nn_tpr, _ = roc_curve(y_test, nn.predict_proba(X_test)[:, 1])

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
