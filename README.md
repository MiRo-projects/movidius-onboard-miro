# movidius-onboard-miro
Running deep learning models onboard the MiRo robot via the Intel Movidius compute stick.
## Abstract
MiRo is a robot with a biomimetic brain-based control system. In order to recognise human faces, a much more efficient and reliable deep learning model should be deployed on MiRo. However, deep learning model is computationally expensive and power-hungry while our MiRo robot only has limited computation resources. The aim of this project is to make deep learning model lightweight and deploy it into Intel Movidius neural compute stick which can be accessed by MiRo. Specifically, this project is targeted at implementing face recognition lightweight and configuring it to run on the MiRo, designing a face recognition system, applying knowledge distillation techniques for model lightweight, and finally presenting experimental results.
## Steps to use
First run collect_my_faces.py, which will automatically call your computer's camera to capture your face image and save it in the faces_my folder.

Next, download the LFW data as training data and unzip it into img_source:http://vis-www.cs.umass.edu/lfw/lfw.tgz

Run preprocess_other_faces.py which will automatically process the face data just downloaded, by using dlib to batch recognize the face part of the image and save it to the specified directory faces_other. The face size is 64*64.

Run train_model.py for model training. Wait for the training to finish and the student and teacher models will appear in their respective folders.

Open the Model Optimizer that is included with OpenVINO and type the following command.
```
python3 mo_tf.py --input_model inference_graph.pb --input Placeholder --input_shape [1,64,64,3]
```
to convert the saved TensorFlow freeze model into Intermediate Representation(IR) for deploying on MiRo via the Intel Movidius compute stick.

The following command is used to run face recognition. (Argument XXXX can be 'teacher' or 'student' which means running teacher model or student model)
```
python face_recognition_with_computestick.py --model XXXX
```
Note: Intel neural compute stick should be plugged in.
