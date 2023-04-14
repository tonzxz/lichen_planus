-----------------------------------  
# Lichen Planus Rash Detection using FRCNN (Flutter App)
-----------------------------------  
+ Setup FRCNN Workspace +   
  ** Add frcnn folder into flutter project to directly deploy a tflite model **  
  Python 3.8  
  pip version 22.3.1  

  pip install tensorflow==2.8.4  
  pip install numpy==1.23.5  
  pip install pandas==1.5.1  
  pip install pillow==9.3.0  
  pip install matplotlib  
  pip install scikit-learn==1.1.3  
  pip install opencv-python  

  python compileannotations.py ( python compileannotations.py --bg, if there are background images )  
  python train.py  
  ** Hyperparameters are located on Config.py from keras_frcnn folder **  
  python test.py ( Test the model, includes calculation of maP from test images )  
  python tflite.py --deploy  

+ Start a flutter project  +  

  Copy files from flutter folder to project folder  
  ** Flutter project must contain a model, train and deploy a model before running the app **  
  ** use "python tflite.py --deploy" to deploy a model to the flutter project **  
  ** Provide proper labels in classifier.dart **  
  flutter run  
  flutter build apk ( build apk )  
  
 + Directory Structure +  

<pre>
 /flutter-proj   
      /**  
      /android  
      /assets     
      /lib  
      /test_icons  
      /frcnn  
           /**  
           -train.py  
           -tflite.py  
      -pubspec.yaml  
      -pubspec.lock
</pre>
     

References:  
(1)  
https://indiantechwarrior.com/building-faster-r-cnn-on-tensorflow/ (article)  
https://github.com/indiantechwarrior/faster_rcnn_tensorflow (repository)  
(2)  
https://github.com/RockyXu66/Faster_RCNN_for_Open_Images_Dataset_Keras (repository)  
