from flask import Flask, jsonify, request
import tensorflow as tf
import numpy as np

app = Flask(__name__)

#cargar modelo una vez al iniciar la aplicacion
modelo = tf.keras.models.load_model('model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # obtener datos de la solicitud
        data = request.get_json(force=True)
        celcius_values = data.get('celcius')
        if celcius_values is None:
            return jsonify(error="El campo 'celcius es requerido."),400
        
        # convertir el valor a un numpy array para la prediccion
        celcius = np.array([[celcius_values]], dtype=float)

        #realizar prediccion
        prediccion = modelo.predict(celcius)

        # convertir el valor de prediccion a un tipo de dato nativo para python
        fahrenheit_value = float(prediccion[0][0])

        # devolver la prediccion 
        return jsonify(fahrenheit=fahrenheit_value)
    
    except Exception as e:
        # Imprimir el error en la consola para depuración
        print(f"Error: {e}")
        return jsonify(error=str(e)), 500
        

if __name__ == '__main__':
    app.run(debug=True)

    