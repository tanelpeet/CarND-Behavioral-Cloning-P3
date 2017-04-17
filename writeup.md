# **Behavioral Cloning** 


The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model MSE loss"

## Rubric Points

### Files Submitted & Code Quality

**1. Submission includes all required files and can be used to run the simulator in autonomous mode**

My project includes the following files:
* model.py containing the script to create and train the model (NB! for training, folders named *examples* and *weights* have to be in the same root as model.py)
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

**2. Submission includes functional code**

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

**3. Submission code is usable and readable**

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.



### Model Architecture and Training Strategy

**1. An appropriate model architecture has been employed**

My model consists of a convolutional neural network with 5 convolutional layers and 4 fully connected layers. The first three convolutional layers have 5x5 filter size with stride 2, while the following two has filters with size of 3x3 and stride 1. The depth of the layers are correspondingly 24, 36, 48, 64, 64. The dense layers contain correspondingly 100, 50, 10 and finally 1 neurons. Each layer uses ReLU as an activation function.

The data is normalized using Keras lambda layer by dividing each value with 127.5 and substracting 1 from the result. 

**2. Attempts to reduce overfitting in the model**

Dropout is being used after each hidden layer in order to reduce overfittin.

The data was split into training and validation sets to understand whether model is overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

**3. Model parameter tuning**

The model used an adam optimizer, so the learning rate was not tuned manually.

**4. Appropriate training data**

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. In addition data was gathered from the second track in order to generalize better.

For details about how I created the training data, see the next section. 


## Detailed Model Architecture and Training Strategy


**1. Solution Design Approach**

The model architecture was based on NVIDIA's [End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) paper. The model used consisted of a convolutional neural network with 5 convolutional layers and 4 fully connected layers. The first three convolutional layers have 5x5 filter size with stride 2, while the following two has filters with size of 3x3 and stride 1. The depth of the layers are correspondingly 24, 36, 48, 64, 64. The dense layers contain correspondingly 100, 50, 10 and finally 1 neurons. Each layer uses ReLU as an activation function.

The data was normalized before it was fed to the first hidden layer. The lower and upper part of the image was also cropped, so the input image wouldn't contain less unneccesary data.

In order to see how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. In order to avoid this, dropout was used after every hidden layer.  

Keras checkpoint files were saved after each epoch, so it would've been possible to use weights from the intermediate epochs, although in practice this wasn't needed. In addition Keras History object was created, so it would be possible to log training loss and validation loss after each epoch. The result of the best run is vizualised on the following image:

![Model MSE loss][image1]

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell close to the edge of the track. These situations might be improved by recording more data for these specific kind of situations.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

**2. Final Model Architecture**

The final model consists of a convolutional neural network with 5 convolutional layers and 4 fully connected layers. The first three convolutional layers have 5x5 filter size with stride 2, while the following two has filters with size of 3x3 and stride 1. The depth of the layers are correspondingly 24, 36, 48, 64, 64. The dense layers contain correspondingly 100, 50, 10 and finally 1 neurons. Each layer uses ReLU as an activation function. After each hidden layer a dropout layer was used.

Amongst the run experiments, the best results were shown, when training was done for 40 epochs, using dropout probability of 0.3.

**3. Creation of the Training Set & Training Process**

The training set was recorded by the help of my collegues. The simulator was set up at my workplace (with gaming steering wheel) and everyone could go and record the data, so in some sense data was collected by crowdsourcing.

Most of the data was recorded on track one using center lane driving, although some laps were also recorded on track two. The tracks were followed in both directions. Some data was also recorded by recovering the vehicle from the left and righ sides of the road back to the center.


To augment the data sat, I also flipped images and angles to generate even more data, so the model would generalize better. The usage of left and right camera was also experimented, but good results weren't achieved.

After the collection process, I had 22128 number of data points. After the data augmentation (flipping images and angles) one epoch consisted of 44256 data points.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. After number of experiments, the best number of epochs was 40, while using dropout of 0.3. I used an adam optimizer so that manually training the learning rate wasn't necessary.