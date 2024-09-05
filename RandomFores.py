'''
Impelenta un randomForest para nuestro problema
'''
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def scaleData(df):
    # Normalizar los datos
    df = (df - df.mean()) / df.std()
    return df

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

df = pd.read_csv('Data_Sets\Processed.csv')

X = df.drop('Class', axis=1)
X = scaleData(X)
y = df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.02, random_state=42)

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
# Make predictions
y_pred = rf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
confusion_matrix = confusionMat(y_test, y_pred)

