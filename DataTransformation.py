# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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