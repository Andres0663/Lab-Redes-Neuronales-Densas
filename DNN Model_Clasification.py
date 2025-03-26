"""
-------------------------------------------------------------
 __________     ________      ________
|___    ___|   |   _____|    |   _____|  |
    |  |       |  |___       |  |        |     Tecnológico
    |  |       |   ___|      |  |        |    de Costa Rica
    |  |       |  |_____     |  |_____   |
    |__|       |________|    |________|  |

-------------------------------------------------------------

    Área Académica Ingeniería Mecatrónica
    MT 8008 - Inteligencia Artificial

    Solución a Laboratorio: Redes neuronales densas (clasificación)

    Autora: Clara Catalina Madrigal Sánchez
    Última modificación: 19:46, 14 de marzo, 2025

-------------------------------------------------------------
"""


# Librerías a utilizar
import tensorflow as tf
import pandas
from keras import layers, models, Input
import matplotlib.pyplot as plt


# Hiperparámetros
train_percentage = 0.75  # % de datos a apartar para el entrenamiento
n = 2  # Número de capas ocultas del modelo
units = 5  # Número de neuronas por capa
activation = "relu"  # Función de activación
learning_rate = 0.01  # Taza de aprendizaje
loss = "BinaryCrossentropy"  # Función de pérdida
batch_size = 250  # Tamaño de lote
epochs = 10  # Número de iteraciones de entrenamiento


# Lectura de los datos
dataset = pandas.read_excel("Airline_Data.xlsx")
print(dataset)

# Truncado de datos
dataset = dataset.drop(columns="Airline")  # Se elimina la columna "Airline"
print(dataset)

for category in dataset:  # Se procede a eliminar toda información que bo esté  
    if category != "Duration":  # relacionada con las elegidas para el análisis.
        if category != "Price":
            dataset = dataset.drop(columns=category)
print(dataset)

# Función que convierte los datos no numéricos de
# la propiedad "Duration" en numéricos.
# Entrada: stringDtype; Salida: integer
def uniform_duration(data):
    if data.find("m") == -1:  # Dato con formato #h
        [horas] = data.split(" ")  # Asigna a "horas"
        horas = horas.strip("h")  # el número de horas
        minutos = 0  # y a "minutos" 0
    elif data.find("h") == -1:  # Dato con formato #m
        [minutos] = data.split(" ")  # Asigna a "minutos"
        minutos = minutos.strip("m")  # el número de minutos
        horas = 0  # y a "horas" 0
    else:
        [horas, minutos] = data.split(" ")  # Dato con formato #h #m
        minutos = minutos.strip("m")  # Extrae las horas y minutos
        horas = horas.strip("h")  # y los asigna a la variable apropiada
    # Se suma la cantidad de minutos totales que durará el vuelo
    data = int(horas)*60 + int(minutos)
    return data


# Procesamiento de datos, por medio del llamado a la función
dataset["Duration"] = dataset["Duration"].apply(lambda x: uniform_duration(x))
print(dataset)

# Normalizado
max_val = dataset.max(axis=0)  # Se obtiene el máximo de cada columna
min_val = dataset.min(axis=0)  # Se obtiene el mínimo de cada columna
difference = max_val - min_val  # Se obtiene la diferencia de los dos
new_dataset = (dataset - min_val)/(difference)  # Y se utiliza para normalizarlas
new_dataset = new_dataset.astype(float)  # Se asegura que los datos sean tipo float

# División en entrenamiento y validación.
trainset = new_dataset.sample(frac=train_percentage)  # ATENCIÓN: HIPERPARÁMETRO
# Se extraen datos para el entrenamiento

testset = new_dataset.drop(trainset.index)  # Y se le quitan esos mismos
# al dataset para crear los datos de prueba
print(trainset)

# Función que convierte los datos numéricos de
# la propiedad "Price" en categorías, según una media.
# Entrada: stringDtype; Salida: integer
def price_category(data, median):
    if data >= median:
        return 1  # Dato es sobre media
    else:
        return 0  # Dato es bajo media


median_value = dataset.median().iloc[1]
dataset["Price"] = dataset["Price"].apply(lambda x: price_category(x, median_value))

median_value = new_dataset.median().iloc[1]
trainset["Price"] = trainset["Price"].apply(lambda x: price_category(x, median_value))
testset["Price"] = testset["Price"].apply(lambda x: price_category(x, median_value))
print(trainset)

# Inicialización del modelo
network = models.Sequential()

# Declaración de la capa de entrada
network.add(Input(shape=(1,)))  # Note: una única variable por vez

# Ciclo de capas de neuronas intermedias
for i in range(n-1):  # n: número de neuronas intermedias
    network.add(layers.Dense(
            units=units,  # units: número de neuronas por capa
            activation=activation))  # activation: función de activación elegida.
            # VER DOCUMENTACIÓN PARA VER LAS POSIBLES OPCIONES A ELEGIR

# Declaración de la capa de salida
network.add(layers.Dense(
        units=1,  # Una única salida
        activation="sigmoid"))  # Activación sigmoide

network.compile(
        optimizer=tf.keras.optimizers.Adam(  # optimizer: algoritmo de optimización
            learning_rate=learning_rate  # learning_rate: ritmo de aprendizaje
        ),
        loss=loss)  # loss: función de pérdida

losses = network.fit(x=trainset["Duration"],  # Datos de entrada (entrenamiento)
                     y=trainset['Price'],  # Datos de salida (entrenamiento)
                     validation_data=( # Conjuntos de datos de validación
                            testset["Duration" ], # Entrada de validación
                            testset['Price']  # Salida de validación
                            ),
                     batch_size=batch_size,  # Tamaño de muestreo
                     epochs=epochs  # Cantidad de iteraciones de entrenamiento
                     )


# Sección opcional: Notificación auditiva de que terminó el entrenamiento.
import winsound
winsound.Beep(350, 500)


# Se extrae el historial de error contra iteraciones de la clase
loss_df = pandas.DataFrame(losses.history)

loss_df.loc[:, ['loss', 'val_loss']].plot() # Se crea la curva a graficar

# Y se llama a la ventana que se muestre la grafica
plt.show()


# Se eligen 10 datos al azar
dato = dataset.sample(frac=10/dataset.shape[0])

# Se preparan los datos a probar
new_dato = (dato - min_val)/(difference)
new_dato = new_dato.astype(float)
datoPrueba = new_dato.drop(columns=["Price"])

# Se predice el precio que tendría, según el modelo
precio = network.predict(datoPrueba)

# Se revierte la normalización al mostrar los resultados
print("\nDato ingresado:")
print(dato)
dato["Price"] = precio
print("\nEstimacion: ")
print(new_dato*difference+ min_val)
