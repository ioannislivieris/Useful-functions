from tensorflow.keras.layers import Activation
from tensorflow.keras.utils  import get_custom_objects

def gelu(x):
#     return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * x^3)))
    return tf.keras.backend.hard_sigmoid(1.702 * x) * x

 
def swish(x, beta = 1):
    return (x * tf.keras.backend.sigmoid(beta * x))


get_custom_objects().update({'gelu':  Activation(gelu)})
get_custom_objects().update({'swish': Activation(swish)})