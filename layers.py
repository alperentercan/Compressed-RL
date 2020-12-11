import xxhash
import tensorflow as tf

class HashedLinear(tf.keras.layers.Layer):
    '''
    A simple HashedNet implementation for Keras.
    '''
    def __init__(self, units=32, n_weights=8, hash_seed=2):
        super(HashedLinear, self).__init__()
        self.hash_seed = hash_seed
        self.n_weights = n_weights
        self.units = units
        
    
    def build(self,input_dim):
        assert len(input_dim) == 2
        self.input_dim = input_dim[1]
        
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(self.n_weights,), dtype="float32"),
            trainable=True, name="Weights")
        
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(self.units,), dtype="float32"), trainable=True, name = "Bias")
        
        self.hash_matrix()


    def hash_matrix(self):
        '''
        Creates hash_matrix with the weight indices
        '''

        self.indx = tf.convert_to_tensor(
            [[xxhash.xxh32("{}_{}".format(i,j),self.hash_seed).intdigest() %self.n_weights for j in range(self.units)]
             for i in range(self.input_dim)], dtype=tf.int32)
            
        
    def call(self, inputs):
        weights = tf.gather(params=self.w,indices=self.indx,name="Indexing")
        return tf.matmul(inputs, weights) + self.b
    
    def get_config(self):
        return {"units": self.units,"n_weights": self.n_weights, "hash_seed": self.hash_seed}
    
    
    
class Temperatured_softmax(tf.keras.layers.Layer):
    '''
    A temperatured softmax layer that could be used for KLL-divergence
    '''
    def __init__(self, temperature=1,**kwargs):
        super(Temperatured_softmax, self).__init__(**kwargs)
        self.softmax = tf.keras.layers.Softmax()
        self.temperature = temperature

    def call(self, inputs,training=None):
        if training:
            return self.softmax(tf.divide(inputs, self.temperature))
        else:
            return inputs
    def get_config(self):
        config = super(Temperatured_softmax, self).get_config()
        config.update({"temperature": self.temperature})
        return config
    

