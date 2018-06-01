# Traffic Sign Recognition

## Overview

This Deep Learning project uses a Convolutional Neural Network to classify traffic signs. It uses the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) to train and validate the model. Afterward, I tested the model using pictures I took of traffic signs around my town, some of which belonged to the trained classes and some of which did not.

### Steps:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[histogram]: ./images/training_example_histogram.png "Histogram"
[collage]: ./images/collage_of_training_examples.png "Collage"
[clahe0]: ./images/clahed_data0.png "Example CLAHE 0"
[clahe1]: ./images/clahed_data1.png "Example CLAHE 1"
[clahe2]: ./images/clahed_data2.png "Example CLAHE 2"
[clahe3]: ./images/clahed_data3.png "Example CLAHE 3"
[rotate]: ./images/augmented_rotated.png "Original and rotated 15º"
[scale]: ./images/augmented_scaled.png "Original and scaled 1.1"
[translate]: ./images/augmented_translated.png "Original and translated (-2, -2)"
[known]: ./images/known_signs.png "Known signs"
[unknown]: ./images/unknown_signs.png "Known signs"
[known0_pred]: ./images/known0_pred.png "Turn left ahead"
[known1_pred]: ./images/known1_pred.png "Priority road"
[known2_pred]: ./images/known2_pred.png "No entry"
[known3_pred]: ./images/known3_pred.png "Children crossing"
[known4_pred]: ./images/known4_pred.png "Speed limit (30km/h)"
[known5_pred]: ./images/known5_pred.png "Go straight or right"
[known0_softmax]: ./images/known0_softmax.png "Turn left ahead plus top 5 softmax"
[known1_softmax]: ./images/known1_softmax.png "Priority road plus top 5 softmax"
[known2_softmax]: ./images/known2_softmax.png "No entry plus top 5 softmax"
[known3_softmax]: ./images/known3_softmax.png "Children crossing plus top 5 softmax"
[known4_softmax]: ./images/known4_softmax.png "Speed limit (30km/h) plus top 5 softmax"
[known5_softmax]: ./images/known5_softmax.png "Go straight or right plus top 5 softmax"
[unknown0_softmax]: ./images/unknown0_softmax.png "Parking plus top 5 softmax"
[unknown1_softmax]: ./images/unknown1_softmax.png "Pedestrian crossing road plus top 5 softmax"
[unknown2_softmax]: ./images/unknown2_softmax.png "Parking plus top 5 softmax"
[unknown3_softmax]: ./images/unknown3_softmax.png "Dead end plus top 5 softmax"
[unknown4_softmax]: ./images/unknown4_softmax.png "Pedestrian zone plus top 5 softmax"
[unknown5_softmax]: ./images/unknown5_softmax.png "Rossmann (store) plus top 5 softmax"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/yonomitt/traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used basic python methods to calculate the data set statistics with the exception of the image size. For that I used the convenient shape parameter on the numpy array.

However, my answers for the statistics vary slightly in that I used an enhanced dataset. I created a script to add more training examples by playing with the translation, scale, and rotation of the provided examples. See [tools/preprocess_data.py](https://github.com/yonomitt/traffic-sign-classifier/blob/master/tools/preprocess_data.py).

This script allowed me to increase the training set from the original 34,799 to one of 452,387 (a factor of 13). Additionally, I also created a balanced training set which had the same number of training examples for every class, as I was worried my classifier would not work as well if it focused on the classes with more data. The balanced data set had 2,340 examples for each of the 43 classes for a total of 100,620 training examples.

In the end, I ended up using the full (452,387) training set to train my classifier because it consistently beat out the other two for accuracy (despite taking significantly longer to train).

* The size of **my** training set is 452,387
* The size of the test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

My first go at exploring the data set was to output a histogram of the classes.

![Histogram of training examples of the 43 classes][histogram]

This shocked me and led me to create the balanced data set, which ensured the same number of training examples per class. While this balanced data set did better than the default data set for training the network, it ultimately was inferior to the full data set I created.

The next thing I wanted to see was an example image from each class.

![Collage of a single image from each of the 43 classes][collage]

I generated this image after several rounds of experimentation. Prior to seeing this image, I had been using full color images from the data set. 

This visualization showed me how extreme the lighting conditions vary from one image to another and made me realize that color was probably not going to be very helpful. 

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

I initially did no preprocessing (besides expanding the data set through translation, rotation, and scale transformations). I intended to try grayscale conversions and other experiments, but first wanted to play with creating an actual network (isn't that the fun part?)

Once I generated the 2nd visualization image (see previous section), I realized I needed to prioritize data preprocessing if I wanted to get any decent results.

I initially converted the images to YUV and centered the images and got slight improvements over the full color inputs. While researching various gray scaling techniques, I came across page on OpenCV's documentation site describing [histogram equalization](http://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html)

This led me to try histogram equalization and Contrast Limited Adaptive Histogram Equalization (CLAHE) on my data set. I found CLAHE far superior to my other preprocessing techniques and adopted it from then on.

Here is an example of some traffic sign images before and after CLAHE.

![Before and after CLAHE][clahe0]
![Before and after CLAHE][clahe1]
![Before and after CLAHE][clahe2]
![Before and after CLAHE][clahe3]


#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The initial notebook split out training, validation, and test sets automatically. While I did enhance the training set, I did not thing to additionally enhance the validation or test sets. I'm not sure I would want to, either as my enhancements weren't "real" signs. It might have been fine for the validation set, as it is used to aid training, but the test set seems unnecessary.

I augmented the data set using a separate script found under [tools/preprocess_data.py](./tools/preprocess_data.py)

This script created 12 new images for every 1 image in the original training set with the following transformations:

- Scale
	- 0.9
	- 1.1
- Translate
	- (-2, -2)
	- (-2, 2)
	- (2, -2)
	- (2, 2)
- Rotate
	- -15º
	- -7.5º
	- -5º
	- 5º
	- 7.5º
	- 15º

The idea for this came from [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)

Here are three examples of an original image and an augmented image:

![alt text][rotate]
![alt text][scale]
![alt text][translate]

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the eighth cell of the ipython notebook. 

I tried a number of different models (cells 6 - 11), including:

- LeNet
- a modified version of LeNet with 3 convolution layers
- a Multi Scale LeNet based on a paper I read, [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)
- a modified version of this Multi Scale LeNet with 3 convolution layers
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
 

#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyper parameters such as learning rate.

The model training code can be found in cells 13-15 of the notebook.

To train the model, I minimized the mean of the softmax cross entropy between the logits and the labels. I believe this is fairly standard and was in the lesson.

For my final model, I trained for 200 epochs using a batch size of 128, keep probability of 0.5, and a learning rate of 0.0001.

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My personality is that of a person who likes to know the effect of each change. I tend to like to change one variable and see what happens. Due to this nature, I created some scripts external to the notebook that would allow me to rapidly try many, **many** experiments.

---

**Traffic_Sign_Classifier.py**

A converted version of the notebook, restructured to make it easier to train different networks

**project2.py**

A wrapper script that allows me to run a single experiment by passing in parameters for the model via the command line.

**train_the_ocean.sh**

A shell script that allowed me to run many experiments in sequence. It currently only has my last two, but at it's peak, I would have 16 different experiments running via this script. The name comes from "boil the ocean", which was a phrase we used at IBM when running many experiments simultaneously while tweaking parameters.

I believe the phrase originally comes from trying to do too much at once or something too hard. It became a goal to achieve instead of an insurmountable obstacle.

**tools/analyze_results.py**

A script to help me analyze the results from my many experiments.

---

I learned how to set up an AWS EC2 machine using the p2.xlarge class to greatly speed up my experimenting. This allowed me to run as many experiments as I did.

I discovered that by using spot instances, I was frequently able to get a p2.xlarge for $0.10 - 0.14 per hour compared to the normal price of $0.90. This saved me a lot of money and allowed me a little more freedom.

Since spot instances are not guaranteed and can be shutdown with only 2 minutes notice, my train_the_ocean.sh script allowed me to setup enough experiments to run over night and keep each experiment isolated. If the machine was shutdown, I would be able to start the script where it left off very easily.

I never had any problems getting a spot instance and keeping it running for 12 hours or so over night.

In addition to the base model, the parameters I played around with included:

- epochs
	- 100
	- 150
	- **200**
	- 400
- keep probability (for dropout)
	- **0.5**
	- 0.63
	- 0.75
- learning rate
	- 0.001
	- 0.00032[^from_paper]
	- **0.0001**
- batch size
	- **128**
	- 256
	- 512
- data set to use
	- original (34,799)
	- enhanced balanced (100,620)
	- **enhanced full (452,387)**
- preprocessing used
	- normalized YUV
	- centered YUV
	- histogram equalization
	- **CLAHE**

Bold options are the ones used for my final project

[^from_paper]: This came from a paper I read, [Traffic Sign Classification Using Deep Inception Based Convolutional Networks](https://arxiv.org/pdf/1511.02992.pdf), which said it was the optimal learning rate for their network. I just threw it into the mix to see what would happen.

One thing I wish I would have had time to try is implementing a decaying learning rate. Sadly, I ran out of time. One of the reasons this project is late is because I always wanted to run *just one more experiment*!

After running my experiments, I chose one to run through the notebook and use for my project.

One unfortunate note. During my experimenting, I happened on several models and parameters that would give me over 99% accuracy on the validation set. Unfortunately, I could not reproduce this when finalizing my notebook.

The code for calculating the accuracy of the model is located in the fifteenth and sixteenth cells of the notebook.

My final model results were:
* validation set accuracy of  0.9832 after 200 epochs with a maximum of 0.9846
* test set accuracy of 0.9722

I never looked at the accuracy of the training set, as I didn't think it was a worthwhile number to consider in my experiments.

The architecture I ended up going with was my modified LeNet architecture with 3 convolution layers. I had hoped that the Multi Scale version of this architecture would be hands down better, but the experiments did not support the hypothesis. Additionally, while the Inception architecture was fun to try, it did not show marked improvement over the modified LeNet. I would have enjoyed trying out different Inception architectures (such as Google's v3 and v4), but the model ran so slowly, I didn't have enough time to try more.

I initially, had dropout on all layers of the network, including the convolution layers, but later changed to just having dropout on the 3 fully connected layers. My reasons for doing so, were to speed up training and because I realized that the convolution layers were already an aggregation of different nodes with the potential for redundancy. It seemed superfluous to add dropout on top of it. Switching to dropout on just the fully connected layers, did not markedly affect the accuracy of the model.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I happen to live in Germany. So instead of searching the web for traffic signs, I just went for a walk and took pictures of some. I ended up taking pictures of signs that both were and were not covered by the classes in the training set. I thought it would be interesting to see what my classifier thought of completely unknown signs. One picture, in fact, is actually the logo of a chain drug store in Germany and not a traffic sign at all!

#### Known traffic signs (covered by the training set)

![alt text][known]

#### Unknown traffic signs (the model has never encountered them before)

![alt text][unknown]

I think the six known signs should be fairly straight forward to classify, as the lighting is good and they are fairly clear. Perhaps the children crossing would be the most difficult to recognize due to lost detail when the image is down sampled to 32x32 pixels. I believe that the model will be very confident in its predictions (high softmax value for the correct sign).

Obviously the model will not be able to classify the unknown signs. My hope, though, is that the softmax values will show the uncertainty.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 20th, 21st, and 22nd cell of the notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn left ahead      		| Turn left ahead   									| 
| Priority road     			| Priority road 										|
| No entry					| No entry											|
| Children crossing	      		| Children crossing					 				|
| Speed limit (30km/h)			| Speed limit (30km/h)      							|
| Go straight or right			| Go straight or right     							|

![alt text][known0_pred]
![alt text][known1_pred]
![alt text][known2_pred]
![alt text][known3_pred]
![alt text][known4_pred]
![alt text][known5_pred]

The model was able to correctly classify all 6 of the known traffic signs for an accuracy of 100% this is compared to 97.22% for the test set. Granted, this was a small sample size, but I would have expected between 5 and 6 of the new images to be correctly classified and that is correct.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 20th-25th cells of the notebook.

#### Known signs

![alt text][known0_softmax]
![alt text][known1_softmax]
![alt text][known2_softmax]
![alt text][known3_softmax]
![alt text][known4_softmax]
![alt text][known5_softmax]

I was very surprised to see that the model was 100% sure of itself of the first 4 images. I imaged it would be confident, but not 100% confident. This was nice to see.

For the 5th image, it was still very sure of itself that it was a 30km/h speed limit sign with 91.97% certainty. What's also nice to see that it was 99.99% certain it was a speed limit sign of some type.

The 6th image (go straight or right) was still correct, but with the least amount of certainty (62.17%). I have to assume this is due to the stickers someone put on the sign near the right arrow and at the base of the arrow. It might have been enough to decrease the model's certainty. It is interesting to see that with 77.55% certainty, the model thought that the sign was either **go straight and right**, **turn right ahead**, or **ahead only**, all of which are related.

#### Unknown signs

![alt text][unknown0_softmax]
![alt text][unknown1_softmax]
![alt text][unknown2_softmax]
![alt text][unknown3_softmax]
![alt text][unknown4_softmax]
![alt text][unknown5_softmax]

I was happy to see that for some of the unknown signs, the model seemed very unsure. The first parking sign, the pedestrian crossing sign, and the dead end sign all had their top class at under 50% certainty. 

The other three, however, the model was more certain of.

The 2nd parking sign, the model was convinced was a bicycle crossing sign with 99.41% certainty. I'm not sure why this is.

The pedestrian zone sign was classified as a children crossing sign. This actually makes some sense, as the children crossing sign has a large and a small person on it, just as this one.

The Rossmann sign was hilariously classified as a mandatory roundabout sign. They both are round, I suppose.

The misclassification of signs with high certainty could be problematic for a self driving car. Seeing something that's not there could be just as dangerous as missing something important. For instance, what would my self driving car do if it came to a Rossmann's store? It might try to perform some sort of circular maneuver despite the fact that it might not be able to.
