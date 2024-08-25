# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Tomamos nuestro Data Set
df_mixed = pd.read_csv('Data_Sets\Jeans.csv')
df_cottons = pd.read_csv('Data_Sets\Mixed.csv')

n = 150 # los primeros 3 minutos no se consideran
df_mixed = df_mixed[df_mixed['Time(min)']>n]
df_cottons = df_cottons[df_cottons['Time(min)']>n]

# El peso es un factor importante en el comportamiento
# de la secadora.
w1 = 16 # Peso en lbs
# incluir peso 1 y peso 2
df_mixed = df_mixed[(df_mixed['Weight']==w1)]# | df_mixed['Weight']==w2)]
df_cottons = df_cottons[(df_cottons['Weight']==w1)]# | (df_cottons['Weight']==w2)] 

# Comviene quitar los valores de delta time en 0
# es cuando la secadora esta apagada
# df_mixed = df_mixed[df_mixed['Delta time']!=0]
# df_cottons = df_cottons[df_cottons['Delta time']!=0]


df_mixed['Class']= 0 # 0 para mixed
df_cottons['Class']= 1 # 1 para cottons


# Concatenar los dataframes para poder hacer la clasificacion
df = pd.concat([df_mixed, df_cottons], axis=0)
# Shuffle 
df = df.sample(frac=1).reset_index(drop=True)


df = df.drop(columns=['Segment','Restriction','Energy','Potenza','Filtered','Smooth','T-A amb','RH amb'])

min_class = df['Class'].value_counts().min()
df = df.groupby('Class').apply(lambda x: x.sample(min_class)).reset_index(drop=True)
print(df['Class'].value_counts())

# Agregar relacion entre las variables
df['out/in'] = df['Outlet']/df['Inlet']
# Agregar el diferencial de Inlet
df['Delta T in'] = (df['Inlet'] - df['Outlet'].shift(1))**2
# Agregar el diferencial de Outlet
df['Delta T out'] = (df['Outlet'] - df['Outlet'].shift(1))**2

df['Delta RDBOutlet'] = (df['Outlet']-df['RDB'])**2
# Agregar el cuadrado
df['Inlet']= df['Inlet']**2
df['Outlet']= df['Outlet']**2

df.to_csv('Data_Sets\Processed.csv', index=False)