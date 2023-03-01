import os
import tensorflow as tf
model_path = "./rcnn.tflite"
mymodel = tf.lite.Interpreter(model_path=model_path)
mymodel.allocate_tensors()
output = mymodel.get_output_details()
print(output)
index = 0
for i in output:
    print(output[index]['shape'])
    index+=1