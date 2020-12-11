import tempfile
import tensorflow as tf
def get_compressed_size(model):
    _, keras_file = tempfile.mkstemp('.h5')
    tf.keras.models.save_model(model, keras_file, 
                               include_optimizer=False)
    
    return get_gzipped_model_size(keras_file)
    
    
def get_gzipped_model_size(file):
    # It returns the size of the gzipped model in bytes.
    import os
    import zipfile

    _, zipped_file = tempfile.mkstemp('.zip')
    with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
        f.write(file)

    return os.path.getsize(zipped_file)

def post_training_quantize(model, quantization):
    
    _,tflite_file = tempfile.mkstemp('.tflite')

    assert quantization in ["float32","float16","dynamic_range"]
    
    if quantization == "float32":
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
               
    
    elif quantization == "float16":
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()        

    elif quantization == "dynamic_range":
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()   
        
    
    with open(tflite_file, 'wb') as f:
        f.write(tflite_model)
    print(f"Gzipped Model Size: {get_gzipped_model_size(tflite_file)}")
    return tflite_model, tflite_file
