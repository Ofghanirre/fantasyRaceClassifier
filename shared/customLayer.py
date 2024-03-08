import tensorflow as tf
import keras

class SquareImageLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SquareImageLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Récupérer les dimensions de l'image
        height = tf.shape(inputs)[1]
        width = tf.shape(inputs)[2]
        
        # Calculer la dimension maximale (côté du carré)
        max_dim = tf.maximum(height, width)

        # Calculer les marges à ajouter pour transformer l'image en un carré
        top_pad = (max_dim - height) // 2
        bottom_pad = max_dim - height - top_pad
        left_pad = (max_dim - width) // 2
        right_pad = max_dim - width - left_pad

        # Ajouter les marges
        padded_image = tf.pad(inputs, [[0, 0], [top_pad, bottom_pad], [left_pad, right_pad], [0, 0]])

        return padded_image

    def compute_output_shape(self, input_shape):
        # La forme de sortie sera celle de l'entrée, mais avec la dimension maximale comme côté pour les deux premières dimensions (hauteur et largeur)
        input_shape = tf.TensorShape(input_shape)
        return tf.TensorShape([input_shape[0], None, None, input_shape[3]])
