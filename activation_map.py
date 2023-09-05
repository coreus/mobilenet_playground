import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import sys
import urllib.request
import matplotlib.pyplot as plt
from PIL import Image

class MatMul(layers.Layer):

    def __init__(self, **kwargs):
        super(MatMul, self).__init__(**kwargs)

    def call(self, inputs):
        print(inputs[1])
        return tf.matmul(inputs[0],inputs[1])

class ThresholdLayer(layers.Layer):

    def __init__(self, threshold=0.5, custom_weights=None, **kwargs):
        super(ThresholdLayer, self).__init__(**kwargs)
        self.threshold = threshold
        self.custom_weights = custom_weights

    def call(self, inputs):
        return tf.tensordot(self.custom_weights, tf.transpose(tf.where(inputs > self.threshold, 1.0, 0.0)), 1)
        
net = MobileNetV2(weights='imagenet')
dense_weights = net.get_layer(name='predictions').get_weights()

y=ThresholdLayer(0.4,dense_weights[0])(net.output)
out_relu = net.get_layer(name='out_relu')


y = MatMul()([out_relu.output,y])

new_net = Model(inputs=net.input,outputs=[net.output,y])

new_net.summary()


url = sys.argv[1]

urllib.request.urlretrieve(url, "/dev/shm/img.jpg")
image = tf.keras.utils.load_img("/dev/shm/img.jpg")
input_arr = tf.keras.utils.img_to_array(image)
input_arr = np.array([input_arr])  # Convert single image to a batch.
predictions = new_net.predict(preprocess_input(input_arr))

np.set_printoptions(threshold=sys.maxsize)
print(predictions[1].reshape((7,7)))
print(np.argmax(predictions[0]))
plt.imsave("/dev/shm/heatmap.jpg",np.repeat(np.repeat(predictions[1].reshape((7,7)),32, axis=0), 32, axis=1), cmap="jet")
#plt.imsave("/dev/shm/heatmap.jpg",predictions[1].reshape((7,7)), cmap="jet")
background = Image.open("/dev/shm/img.jpg")
overlay = Image.open("/dev/shm/heatmap.jpg").resize((224,224))

background = background.convert("RGBA")
overlay = overlay.convert("RGBA")

new_img = Image.blend(background, overlay, 0.7)
new_img.save("/mnt/d/heatmap.png","PNG")