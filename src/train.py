import numpy as np
import matplotlib.pyplot as plt
from model import crear_modelo

#datos de entrenamiento
celcius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

#crear modelo
modelo = crear_modelo()

#entrenear modelo
print("--> Comenzando entrenamiento...")
historial = modelo.fit(celcius, fahrenheit, epochs=500, verbose=False)
print("modelo terminado!!!")

#guardar el modelo entrenado
modelo.save('model.h5')
print("modelo guardado como 'model.h5")

# Visualización del entrenamiento
plt.plot(historial.history['loss'])
plt.title('Pérdida del Modelo durante el Entrenamiento')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.show()
