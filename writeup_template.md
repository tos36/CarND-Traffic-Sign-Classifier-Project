# **Traffic Sign Recognition** 

## Writeup


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/hist_train.png "Training dataset"
[image2]: ./examples/hist_valid.png "Validation dataset"
[image3]: ./examples/hist_test.png "Test dataset"
[image4]: ./examples/color_image.png "Before processing"
[image5]: ./examples/gray_image.png "After processing"


[image6]: ./mydata/1.jpg "Traffic Sign 1"
[image7]: ./mydata/2.jpg "Traffic Sign 2"
[image8]: ./mydata/3.jpg "Traffic Sign 3"
[image9]: ./mydata/4.jpg "Traffic Sign 4"
[image10]: ./mydata/5.jpg "Traffic Sign 5"
[image11]: ./mydata/6.jpg "Traffic Sign 6"

[image12]: ./examples/1_result.png "Result 1"
[image13]: ./examples/2_result.png "Result 2"
[image14]: ./examples/3_result.png "Result 3"
[image15]: ./examples/4_result.png "Result 4"
[image16]: ./examples/5_result.png "Result 5"
[image17]: ./examples/6_result.png "Result 6"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ? > 34799
* The size of the validation set is ? > 4410
* The size of test set is ? > 12630
* The shape of a traffic sign image is ? > (32, 32, 3)
* The number of unique classes/labels in the data set is ? > 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]
![alt text][image2]
![alt text][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale in order to

- reduces the datasize.

- avoid over fitting against color. Color is not necessary factor to recognize traffic sighs. In some situation like night or backlight, It's hard to distinguish colors.

- avoid over fitting against camera filter. We can apply this function to multipule type of filter, RGB, YUV, RCCB.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image4]

As a last step, I normalized the image data in order to center the data and execute training effectively.

Here is an example of an original image and an augmented image:

![alt text][image5]

The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| ReLU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| ReLU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Fully connected		| inputs 400, outputs 120        									|
| Relu				|     									|
|	Dropout					|												|
| Fully connected		| inputs 120, outputs 84        									|
| Relu				|     									|
|	Dropout					|												|
| Fully connected		| inputs 84, outputs 43        									|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....
- AdamOptimizer as a optimizer
- softmax cross entropy with logits as an loss function
- Batch_size: 128
- Epoch: 30
- learning rate: 0.002
- Drop out probability: 0.5

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* Max validation set accuracy of 0.961 
* test set accuracy of 0.938

I tried with some parameters. Followings are summary of it.
 - epoch:10, lr:0.005, accuracy 0.91, accuracy fluctuated
 - epoch:10, lr:0.001, accuracy 0.90, accuracy fluctuated  
 - epoch:10, lr:0.002, dropout:0.5 , accuracy 0.94   
 - epoch:25, lr:0.002, dropout:0.5 , accuracy 0.96   
 - epoch:40, lr:0.002, dropout:0.5 , accuracy 0.95

If a well known architecture was chosen:
* What architecture was chosen? :Lenet
* Why did you believe it would be relevant to the traffic sign application? 
:It wroks well in MNIST task. I started with LeNet since the image is simple and the number of class is not big in this project.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
: The accuracy for validation dataset is 96% and the one for test dataset is 93%. This prove the model is working well.
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] ![alt text][image7] ![alt text][image8] 
![alt text][image9] ![alt text][image10] ![alt text][image11]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Roundabout mandatory      		| Roundabout mandatory	| 
| Priority road    			| Priority road    				|
| No vehicles					| No vehicles				|
| 30 km/h	      		| 30 km/h	      		|
| General caution			| General caution			|
| No toilets		| Priority road			|


The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 88%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first 5 image, the model is relatively sure that this is a stop sign (probability of > 0.99)
![alt text][image12]
![alt text][image13]
![alt text][image14]
![alt text][image15]
![alt text][image16]


For the last image, this is not a general traffic sign, but we need to set another class like "unknown" to avlid misclassification for this kind of edge cases.

![alt text][image17]



