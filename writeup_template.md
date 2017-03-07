#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./images/training_example_histogram.png "Histogram"
[image2]: ./images/collage_of_training_examples.png "Collage"
[image3]: ./images/clahed_data0.png "Example CLAHE 0"
[image4]: ./images/clahed_data0.png "Example CLAHE 1"
[image5]: ./images/clahed_data0.png "Example CLAHE 2"
[image6]: ./images/clahed_data0.png "Example CLAHE 3"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/yonomitt/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used basic python methods to calculate the data set statistics with the exception of the image size. For that I used the convenient shape parameter on the numpy array.

However, my answers for the statistics vary slightly in that I used an enhanced dataset. I created a script to add more training examples by playing with the translation, scale, and rotation of the provided examples. See [tools/preprocess_data.py](https://github.com/yonomitt/traffic-sign-classifier/blob/master/tools/preprocess_data.py).

This script allowed me to increase the training set from the original 34,799 to one of 452,387 (a factor of 13). Additionally, I also created a balanced training set which had the same number of training examples for every class, as I was worried my classifier would not work as well if it focused on the classes with more data. The balanced data set had 2,340 examples for each of the 43 classes for a total of 100,620 training examples.

In the end, I ended up using the full (452,387) training set to train my classifier because it consistently beat out the other two for accuracy (despite taking significantly longer to train).

* The size of **my** training set is 452,387
* The size of the test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

My first go at exploring the data set was to output a histogram of the classes.

![Histogram of training examples of the 43 classes][image1]

This shocked me and led me to create the balanced data set, which ensured the same number of training examples per class. While this balanced data set did better than the default data set for training the network, it ultimately was inferior to the full data set I created.

The next thing I wanted to see was an example image from each class.

![Collage of a single image from each of the 43 classes][image2]

I generated this image after several rounds of experimentation. Prior to seeing this image, I had been using full color images from the data set. 

This visualization showed me how extreme the lighting conditions vary from one image to another and made me realize that color was probably not going to be very helpful. 

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

I initially did no preprocessing (besides expanding the data set through translation, rotation, and scale transformations). I intended to try grayscale conversions and other experiements, but first wanted to play with creating an actual network (isn't that the fun part?)

Once I generated the 2nd visualization image (see previous section), I realized I needed to prioritize data preprocessing if I wanted to get any decent results.

I initially converted the images to YUV and centered the images and got slight improvements over the full color inputs. While researching various grayscaling techniques, I came across page on OpenCV's documentation site describing [histogram equalization](http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)

This led me to try histogram equalization and Contrast Limited Adaptive Histogram Equalization (CLAHE) on my data set. I found CLAHE far superior to my other preprocessing techniques and adopted it from then on.

Here is an example of some traffic sign images before and after CLAHE.

![Before and after CLAHE][image3]
![Before and after CLAHE][image4]
![Before and after CLAHE][image5]
![Before and after CLAHE][image6]


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

#TODO#

The code for splitting the data into training and validation sets is contained in the fifth code cell of the IPython notebook.  

To cross validate my model, I randomly split the training data into a training set and validation set. I did this by ...

My final training set had X number of images. My validation set and test set had Y and Z number of images.

The sixth code cell of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because ... To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the eigth cell of the ipython notebook. 

I tried a number of different models (cells 6 - 11), including:

- LeNet
- a modified version of LeNet
- a Multi Scale LeNet based on a [paper I read](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
- a modified version of this Multi Scale LeNet
- Inception

The final model I went with was LeNet3_dropout_fc, which is a modified LeNet with 3 convolution layers and dropout **only** on the fully connected layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 12x12x16	|
| RELU					|												|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x32	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x32 				|
| Fully connected		| outputs 200     									|
| RELU					|												|
| Fully connected		| outputs 84     									|
| RELU					|												|
| Fully connected		| outputs 43     									|
| Softmax				|        									|
 

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

#TODO#

To train the model, I minimized the mean of the softmax cross entropy between the logits and the labels. I believe this is fairly standard and was in the lesson.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My personality is that of a person who likes to know the effect of each change. I tend to like to change one variable and see what happens. Due to this nature, I created some scripts external to the notebook that would allow me to rapidly try many, **many** experiments.

---

**Traffic_Sign_Classifier.py**

A converted version of the notebook, restructured to make it easier to train different networks

**project2.py**

A wrapper script that allows me to run a single experiment by passing in parameters for the model via the command line.

**train_the_ocean.sh**

A shell script that allowed me to run many experiments in sequence. It currently only has my last two, but at it's peak, I would have 16 different experiments running via this script.

**tools/analyze_results.py**

A script to help me analyze the results from my many experiments.

---

After running my expermients (I didn't run everything I wanted, I ran out of time... one of the reasons this project is late), I chose one to run through the notebook and use for my project.

One unfortunate note. During my experimenting, I happened on several models and parameters that would give me over 99% accuracy on the validation set. Unfortunately, I could not reproduce this when finalizing my notebook.

#TODO#
The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 