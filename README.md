This repository contains code for a Convolutional Neural Network (CNN) image classification project. The project includes the process of gathering data from Kaggle, installing required packages, preprocessing the data, defining the CNN model, saving and loading the model, deploying it as a simple web app, and troubleshooting techniques such as data augmentation, increasing model complexity, and using early stopping.

                                                                            Let's Get Started 
                                                                                                                 
1) Data gathering
     Dowload a suitable image classification dataset and save it in a new folder in the same filepath as your pycharmprojects. eg: C:\Users\GKamau\PycharmProjects\CNN\ data.
     The data folder in this case has both the training and testing dataset
2) Installing required packages:
         a) numpy: For numerical operations.
         b) pandas: For data manipulation and analysis.
         c) matplotlib: For creating plots and visualizations.
         d) scikit-learn: For machine learning tools and utilities.
         e) tensorflow: For building and training neural networks.
         f) Flask: For creating a web application.
         g) Pillow: For working with images.
3) Preprocessing. - resizing images, normalizing pixel values and encoding labels using the load_and_preprocess_data function.
4) Defining the model
   The CNN model is defined using the Keras library.
   The architecture includes convolutional layers, batch normalization,early stopping, max-pooling,ReLu and Softmax activation functions and fully connected layers. 
5) Saving the model - so that you can reuse the model without having to retrain it.
6) Trouble-shooting the model --using data augmentation, increasing model complexity and adding drop out rates to prevent overfitting 
   
7) Deploying the web app using Flask -- you'll need an app.py, style.css file, index.html and result.html files.
8) Run the app
