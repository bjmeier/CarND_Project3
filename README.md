#**Behavioral Cloning** 


### The goals / steps of this project are the following:
- Use the simulator to collect data of good driving behavior
- Build, a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report


[//]: # (Image References)

[image1]: ./report/left.jpg "Left Camera"
[image2]: ./report/center.jpg "Center Camera"
[image3]: ./report/right.jpg "Right Camera"
[image4]:  ./report/flipped.jpg "Flipped Image"
[image5]: ./report/cropped.jpg "Cropped Image"
[image6]: ./report/track2.jpg "Track 2 video"
[image7]:./report/center.jpg "Track 1 video"


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
- model.py containing the script to create and train the model
- drive.py for driving the car in autonomous mode
- model.h5 containing a trained convolution neural network 
- video.mp4 showing a lap around track 1
- video_track2.mp4 showing a partial lap around track 2
- writeup_report.md  summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network that followed the NVIDIA example. 

Top 50 pixels and bottom 20 pixels are cropped from 160 x 320 images (model.py line 63)
Data is normalized by dividing each pixel by 255 and then subtracting 0.5 (model.py line 64)
5 x 5 convolution with a 24 layer depth and a RELU activation
5 x 5 convolution with a 36 layer depth and a RELU activation
5 x 5 convolution with a 48 layer depth and a RELU activation
3 x 3 convolution with a 64 layer depth and a RELU activation
3 x 3 convolution with a 64 layer depth and a RELU activation
Flatten
Fully connected layer with 100 output nodes
Fully conected layer output with 50 output nodes
Fully connected layer output with 10 output nodes
Fully connected layer with final output node
(model.py lines 65 - 75)

#### 2. Attempts to reduce overfitting in the model

To prevent overfitting, training was limited to two epochs. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

A correction factor of 0.9 and a training length of 2 epochs were selected based on track performance.  Smaller correction factors faild to keep the car on the track. Larger correction factors caused chatter.  Dropout was found to be harmful to performance, as was adding RELU activations after the fully connected layers.

#### 4. Creation of the Training Set & Training Process

Training data was chosen to keep the vehicle driving on the road. I used four laps of center lane driving on track one.  The left and right images were used to create corrective actions.

![lalt text][image1]
*left image*

![alt text][image2]
*center image*


![alt text][image3]
*right image*


To augment the data set, I also flipped images and angles thinking that this would add more data and prevent any left or right bias.  For example, here is a flipped version of the above center image:

![alt text][image4]
*flipped center image*


After the collection process, I had 69,594 number of data points. To avoid showing extraneous information, the top 50 and bottom 20 pixels were removed. Here is the cropped version of the above center image:

![alt text][image5]
*cropped center image*


I then preprocessed this data by dividing each pixel value by 255 and then subtracting 0.5 for the result to normalize the data.


I finally randomly shuffled the data set and put 5% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was two as more epochs consistantly failed to impove the validation error. I used an adam optimizer so that manually training the learning rate wasn't necessary.

#### 5. Results
The car successfully navigated track 1. Video of this is given in the video.mp4 file.  This file has been uploaded to YouTube and can be viewed by clicking on the below image.

[![Track 1][image7]](https://youtu.be/vhzxwS0nGf0)
*click image to view track 1 video*

Track 2 was much more challanging, even for a human driver. It was found that staying in the center of the track caused performance on track 1 to suffer as the edges of curves were mistaken for the center dashed lines on track 2. However, using only data from track 1, some success was achieved on track 2.  This is shown in the video_track2.mp4 file which has been uploaded to YouTube and is available by clicking on the below image.

[![Track 2][image6]](https://youtu.be/EIk6GieTMG8)
*click image to view track 2 video*




