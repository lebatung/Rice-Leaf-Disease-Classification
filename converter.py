# Convert .h5 file into a tflite file for embedding to flutter 
import tensorflow as tf
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

h5_model_path = 'rice_disease_classifier.h5'

model = tf.keras.models.load_model(h5_model_path)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

# Lưu mô hình .tflite ra file
tflite_model_path = 'model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)
