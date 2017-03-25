# **Behavioral Cloning**


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report



#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network
* `writeup.md` summarizing the results
* `model_run` for the video recording on track 1


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used the end-to-end model developed by NVIDIA. (Reference: https://arxiv.org/abs/1604.07316). Through initial test, its computational effort is reasonable. Based on research paper, it's suitable for image learning for self driving.

The model starts with a lamda layer to normalize the image between -1 and 1 so that the result can be more accurate and optimize faster.

The second lamda layer is to remove unnecessary portion which doesn't contain lane.

The model I adopted consists of three layers of convolutional layers with number of convolutional layers to be 24, 36 and 48. All the three lays have kernal size of 5x5, and activated by relu function for nonlinearity. Every convolutional layer is followed by a max polling layer.

After that, the model has 3 fully connected layers, with number of each layer to be 100, 50, and 10.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. The dropout layer has rate of 0.5, and it's introduced after the first fully connected layer.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 164).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used all the center lane driving data.

In order to increase the number of training data, I flipped 50% of randomly selected center lane image. I also used 50% of left and right camera image, and applied steering adjustment of 0.229 through experimentation.


### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use deep learning model and feed in sufficient data to let system react to road condition appropriately.

My first step was to use a convolution neural network model exactly the same as the NVIDIA end to end model because the use case of recognizing lane image is similar.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set with ratio of 8:2.

At first, I spent most of my time finding a model that is computational efficient. The first model took close to an hour to complete one epoch. I added in the max pooling layer to help summarize the learning from the convolution layer, and it helps reduce the training time

The first training result couldn't really pass the first turn. I started to add in more transformed image into the model, such as flipped image, and images from side cameras. This has meaningful improvement on the model performance, the steering is much clever.

I also adjusted the learning rate to find a sweet spot for nice trade off between training effort and performance. I arrived at the learning rate of 1e-4.

I also tested the epoch number. Any number bigger than 3 only reduce the loss marginally. so I kept the number at 4.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consists of three layers of convolutional layers with number of convolutional layers to be 24, 36 and 48. All the three lays have kernal size of 5x5, and activated by relu function for nonlinearity. Every convolutional layer is followed by a max polling layer.

After that, the model has 3 fully connected layers, with number of each layer to be 100, 50, and 10.



#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![image1](https://s25.postimg.org/lll6nmarj/center_2017_03_12_19_02_25_778.jpg)

![image2](https://s25.postimg.org/4c9b2rabj/center_2017_03_12_19_05_20_316.jpg)

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to move back to the lane after drifting offtrack. These images show what a recovery looks like starting from left of the lane :

![left1](https://s25.postimg.org/b3zq5lzb3/center_2017_03_12_20_36_39_333.jpg)

![left2](https://s25.postimg.org/yw91h51bz/center_2017_03_12_20_36_43_188.jpg)


Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would generate additional data point for training.

I have also drove on the reversed direction of the track to generate additional data points.


After the collection process. I then preprocessed this data by removing the portion of image without lane, and normalized it between -1 and 1.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 because anything bigger than 5 doesn't reduce loss meaningfully. I used an adam optimizer so that manually training the learning rate wasn't necessary.
