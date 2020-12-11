
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

from tensorflow import keras

import tempfile
import layers
import sys
from utils import *



LAYERS_DICT = {"Hashed": layers.HashedLinear }
def main(layer_name):
    assert layer_name in LAYERS_DICT.keys(), f"No such layer type,  only available: {LAYERS_DICT.keys()}"
    layer = LAYERS_DICT[layer_name]
    model = keras.Sequential([keras.layers.Reshape((-1,)),
                            layer(128,128),
                            keras.layers.ReLU(),
                            layer(64,512),
                            keras.layers.ReLU(),
                            layer(10, 50)
                             ])
    model.compile(optimizer='adam',
             loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics='accuracy')

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train, x_test = x_train/255., x_test/255.
    
    model.fit(x_train,y_train, epochs = 5)
    model.evaluate(x_test,y_test)
    
    
    _, keras_file = tempfile.mkstemp('.h5')
    print('Saving clustered model to: ', keras_file)
    keras.models.save_model(model, keras_file, 
                    include_optimizer=False)

    print("Size of gzipped model: %.2f bytes" % (get_gzipped_model_size(keras_file)))

    
    

if __name__ == "__main__":
    layer_name = sys.argv[1]
    main(layer_name)

