import tensorflow as tf
import numpy as np

#cargar el modelo entrenado
modelo = tf.keras.models.load_model('model.h5')

#ejemplo de datos para predecir
datos = np.array([0, 10, 25], dtype=float)

# realizar predicciones
predicciones = modelo.predict(datos)
print(predicciones)


