# Attendance-System-based-on-Face-Recognition
This repository contains the files and code I have used to make this project.

A facial recognition attendance system uses facial recognition technology to identify and verify a person using the person's facial features and automatically mark attendance in an excel sheet.

Steps:
Step 1: Finding all the Faces
The first step is face detection.
We’re going to use a method invented in 2005 called Histogram of Oriented Gradients (HOG).

Step 2: Posing and Projecting Faces
Now we have to deal with the problem that faces turned different directions look totally different to a computer.
So we will try to warp each picture so that the eyes and lips are always in the sample place in the image. This will make it a lot easier for us to compare faces.

Step 3: Encoding Faces
We need our model to be fast, so we have to check for only the relevant information which distinguishes every face, it measurements of ear, space between eyes and train our model using Convolution Neural Network.

Step 4: Finding the person’s name from the encoding
Now we have to find the person in our database of known people who has the closest measurements to our test image.
We can do this by using a machine learning classification algorithm, here we have used SVM (Support Vector Machine) Classifier.
