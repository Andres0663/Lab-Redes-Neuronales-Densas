# Sección de imports
import tensorflow as tf
import pandas as pd
from keras import layers, models, Input
import matplotlib.pyplot as plt
from keras.utils import to_categorical  # Importar to_categorical
from sklearn.preprocessing import LabelEncoder  # Importar LabelEncoder

# Hiperparámetros
train_percentage = 0.75  # Porcentaje de datos para entrenamiento
n = 2  # Número de capas ocultas
units = 5  # Neuronas por capa
activation = "relu"  # Función de activación
learning_rate = 0.01  # Tasa de aprendizaje
loss = "categorical_crossentropy"  # Función de pérdida para clasificación multiclase
batch_size = 250  # Tamaño del lote
epochs = 10  # Iteraciones de entrenamiento

# Lectura de los datos
dataset = pd.read_csv("./SESION 01/LABORATORIO/milknew.csv")

# División en entrenamiento y validación
trainset = dataset.sample(frac=train_percentage, random_state=42)  # Se fija la semilla para reproducibilidad
testset = dataset.drop(trainset.index)

# Codificación de las etiquetas
label_encoder = LabelEncoder()
trainset["Grade"] = label_encoder.fit_transform(trainset["Grade"])
testset["Grade"] = label_encoder.transform(testset["Grade"])

# Inicialización del modelo
network = models.Sequential()

# Capa de entrada
network.add(Input(shape=(7,)))  # 7 características de entrada

# Capas ocultas
for i in range(n-1):  # n capas ocultas
    network.add(layers.Dense(units=units, activation=activation))

# Capa de salida
network.add(layers.Dense(units=3, activation="softmax"))  # Softmax para clasificación en 3 clases

# Compilación del modelo
network.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=loss,
    metrics=["accuracy"]
)

# Definir las columnas de entrada y salida
X_train = trainset[["pH", "Temprature", "Taste", "Odor", "Fat", "Turbidity", "Colour"]].values
y_train = to_categorical(trainset["Grade"].values, num_classes=3)  # One-hot encoding
X_test = testset[["pH", "Temprature", "Taste", "Odor", "Fat", "Turbidity", "Colour"]].values
y_test = to_categorical(testset["Grade"].values, num_classes=3)  # One-hot encoding

# Entrenamiento
history = network.fit(
    x=X_train,
    y=y_train,
    validation_data=(X_test, y_test),
    batch_size=batch_size,
    epochs=epochs
)

# Gráfico de pérdida
loss_df = pd.DataFrame(history.history)
loss_df.loc[:, ['loss', 'val_loss']].plot()
plt.savefig('learning_curve.png')  # Guardar la gráfica en un archivo
