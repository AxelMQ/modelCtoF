import tensorflow as tf

def crear_modelo():
    oculta1 = tf.keras.layers.Dense(units=3, input_shape=[1])
    oculta2 = tf.keras.layers.Dense(units=3)
    salida = tf.keras.layers.Dense(units=1)
    modelo = tf.keras.Sequential([oculta1, oculta2, salida])

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(0.1),
        loss='mean_squared_error'
    )

    return modelo
