# Real-Time-Face-Mask-Detection
This is a code for a real-time face mask detection system using a pre-trained Convolutional Neural Network (CNN) model. The system detects faces in real-time using a webcam or any other camera connected to the computer, and then classifies each detected face as either "with mask" or "without mask".


DATASET: https://www.kaggle.com/datasets/bismaimran17/face-mask-dataset

the flow of the project step by step:

### 1. Model Training (`train.py`):

1. **Importing Libraries:**
   - Libraries like `ImageDataGenerator`, `Sequential`, `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, `Dropout`, `ModelCheckpoint` are imported.

2. **Model Architecture:**
   - A Convolutional Neural Network (CNN) model is defined using Keras.
   - It consists of two convolutional layers, each followed by a max-pooling layer.
   - The output is flattened and passed through a dropout layer to reduce overfitting.
   - Two fully connected layers are added, the last one using softmax activation for binary classification.
   - The model is compiled using the Adam optimizer, binary crossentropy loss, and accuracy as the evaluation metric.

3. **Data Augmentation:**
   - The training data is loaded using `flow_from_directory` from the specified directory.
   - Image data augmentation is applied using `ImageDataGenerator` to create variations in the training dataset, including rotation, shifting, shearing, zooming, and horizontal flipping.
   - The validation data is loaded without augmentation.

4. **Model Training:**
   - Model training is performed using `fit` on the training generator.
   - The `ModelCheckpoint` callback is used to save the model weights after each epoch only if the validation loss improves.

### 2. Face Mask Detection (`testing.py`):

1. **Importing Libraries:**
   - Libraries like `cv2`, `numpy`, and `load_model` from Keras are imported.

2. **Loading Pre-trained Model:**
   - The pre-trained model (trained using `train.py`) is loaded using `load_model`.

3. **Setting Up Face Detection:**
   - A webcam feed is initiated using OpenCV.
   - A pre-trained Haar Cascade classifier is loaded for face detection.

4. **Real-time Face Mask Detection:**
   - Inside the infinite loop:
     - Frames are continuously captured from the webcam.
     - The image is flipped horizontally to act as a mirror.
     - The image is resized for faster face detection.
     - Faces are detected using the Haar Cascade classifier, and bounding boxes are drawn around them.
     - For each detected face:
       - The face region is extracted, resized, normalized, and reshaped to match the model input shape.
       - The model predicts whether a mask is present or not.
       - Bounding boxes are drawn around faces with different colors for mask and without mask.
       - The predicted label is displayed on the image.
     - The processed image is displayed in real-time.

5. **Exiting the Program:**
   - The program can be exited by pressing the 'Esc' key.
   - After exiting, the webcam is released, and all windows are closed.

### Overall Flow:
1. **Training:**
   - Run `train.py` to train the face mask detection model. This saves the model weights after each epoch.

2. **Testing/Detection:**
   - Run `testing.py` to open a real-time webcam feed.
   - The pre-trained model is loaded.
   - Face detection is performed, and predictions are made for each detected face.
   - Results are displayed on the webcam feed.
   - Exit the program using the 'Esc' key.

This project essentially combines the training of a face mask detection model with real-time application using a webcam feed. The training script generates a model that is then used for making predictions in the testing script, providing a practical application for face mask detection.
