-----------------------------------  
# Lichen Planus Rash Detection using FRCNN (Flutter App)
-----------------------------------  
+ Project Description +

Planus Identifier is a medical diagnostic tool designed to assist individuals in detecting and monitoring symptoms associated with lichen planus skin rash using image processing. The process by which a system analyzes and manipulates digitalized images of diseased skin areas in order to identify and classify the type of lichen planus rash.

Lichen planus is a skin condition that manifests as a rash of flat-topped purple itchy areas. Many autoimmune disorders are linked to lichen planus, particularly hair loss and digestive problems. Patients with lichen planus are also five times more likely to be infected with hepatitis C.
 
To build our system, we will employ an agile methodology known as the SCRUM framework, which provides a flexible and iterative approach to project management. With the advancement of our system, people will be able to identify lichen planus rash using a mobile phone that supports certain camera features.

Our proposed system deals with dermatology and can greatly aid the health sector. The project will not only focus on a certain age group but rather, it can be utilized by any individual. It then makes the system very beneficial to anyone with skin complications
  
+ Tools used +  
  - Flutter Framework
  - Python 3.8
  - Tensorflow API
  - Tensorflow Lite
  - Android Studio (Dev Tools)
     
+ Interface +  
  
<img src="screenshots/camera.jpg " width="320" height="680"> <img src="screenshots/camera-negative.jpg " width="320" height="680"> <img src="screenshots/detect-camera.jpg " width="320" height="680"> <img src="screenshots/detections.png " width="320" height="600"> <img src="screenshots/manual.png " width="320" height="600"> <img src="screenshots/diagnosis.png " width="320" height="600">

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
  run install.bat  
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
