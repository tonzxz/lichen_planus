import os
import tensorflow as tf
model_path = "./model_classifier.tflite"
mymodel = tf.lite.Interpreter(model_path=model_path)
mymodel.allocate_tensors()
output = mymodel.get_output_details()
input = mymodel.get_input_details()
print("INPUT TENSORS:")
index = 0
for i in input:
    print(input[index]['shape'])
    index+=1
print("OUTPUT TENSORS:")
index = 0
for i in output:
    print(output[index]['shape'])
    index+=1