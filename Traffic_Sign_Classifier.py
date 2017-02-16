
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree
# 
# ## Deep Learning
# 
# ## Project: Build a Traffic Sign Recognition Classifier
# 
# In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 
# 
# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 
# 
# In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.
# 
# The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.
# 
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ---
# ## Step 0: Load The Data

# In[ ]:

# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file =   "../data/traffic-signs-data/train.p"
validation_file = "../data/traffic-signs-data/valid.p"
testing_file =    "../data/traffic-signs-data/test.p"
### Preprocessed file
training_pp_file = "../data/traffic-signs-data/train_pp.p"
### Preprocessed balanced file
training_pp_bal_file = "../data/traffic-signs-data/train_pp_bal.p"

with open(training_pp_bal_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


# ---
# 
# ## Step 1: Dataset Summary & Exploration
# 
# The pickled data is a dictionary with 4 key/value pairs:
# 
# - `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
# - `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
# - `'sizes'` is a list containing tuples, (width, height) representing the the original width and height the image.
# - `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**
# 
# Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

# ### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas

# In[ ]:

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(y_train)

# TODO: Number of testing examples.
n_test = len(y_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# ### Include an exploratory visualization of the dataset

# Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc.
# 
# The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.
# 
# **NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections.

# In[16]:

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
# Visualizations will be shown in the notebook.


# ----
# 
# ## Step 2: Design and Test a Model Architecture
# 
# Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).
# 
# There are various aspects to consider when thinking about this problem:
# 
# - Neural network architecture
# - Play around preprocessing techniques (normalization, rgb to grayscale, etc)
# - Number of examples per label (some have more than others).
# - Generate fake data.
# 
# Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.
# 
# **NOTE:** The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play!

# ### Pre-process the Data Set (normalization, grayscale, etc.)

# Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.

# In[4]:

### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.

n_examples = len(X_train)

# ### Model Architecture

# In[ ]:

### Define your architecture here.
### Feel free to use as many code cells as needed.

### TO TRY:
### 1) sermanet architecture
### 2) Replace pooling with dropout

import tensorflow as tf

# Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
mu = 0
sigma = 0.1

# Activation function to use
activation = tf.nn.relu

def linear(x, W, b):
    return tf.add(tf.matmul(x, W), b)

def conv2d(x, W, b, strides=1):
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID')
    return tf.nn.bias_add(x, b)

def dropout(x, keep_prob):
    return tf.nn.dropout(x, keep_prob)

def maxpool2d(x, k=2):
    return tf.nn.max_pool(
        x,
        ksize=[1, k, k, 1],
        strides=[1, k, k, 1],
        padding='SAME')

def gen_keep_prob():
    return tf.placeholder(tf.float32, name="keep_prob")

def LeNet(x):

    weights = {
        'conv1' : tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma)),
        'conv2' : tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma)),
        'flat3' : tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma)),
        'flat4' : tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma)),
        'flat5' : tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    }
    biases = {
        'conv1' : tf.Variable(tf.zeros(6)),
        'conv2' : tf.Variable(tf.zeros(16)),
        'flat3' : tf.Variable(tf.zeros(120)),
        'flat4' : tf.Variable(tf.zeros(84)),
        'flat5' : tf.Variable(tf.zeros(43))
    }

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1 = conv2d(x, weights['conv1'], biases['conv1'])

    # Activation
    conv1 = activation(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = maxpool2d(conv1, 2)

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2 = conv2d(conv1, weights['conv2'], biases['conv2'])

    # Activation.
    conv2 = activation(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = maxpool2d(conv2, 2)

    # Flatten. Input = 5x5x16. Output = 400.
    flat3 = tf.reshape(conv2, [-1, weights['flat3'].get_shape().as_list()[0]])

    # Layer 3: Fully Connected. Input = 400. Output = 120.
    flat3 = linear(flat3, weights['flat3'], biases['flat3'])

    # Activation.
    flat3 = activation(flat3)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    flat4 = linear(flat3, weights['flat4'], biases['flat4'])

    # Activation.
    flat4 = activation(flat4)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    logits = linear(flat4, weights['flat5'], biases['flat5'])

    return logits

def LeNet_dropout(x, keep_prob):

    weights = {
        'conv1' : tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma)),
        'conv2' : tf.Variable(tf.truncated_normal(shape=(9, 9, 6, 16), mean = mu, stddev = sigma)),
        'flat3' : tf.Variable(tf.truncated_normal(shape=(1600, 400), mean = mu, stddev = sigma)),
        'flat4' : tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma)),
        'flat5' : tf.Variable(tf.truncated_normal(shape=(120, 43), mean = mu, stddev = sigma))
    }
    biases = {
        'conv1' : tf.Variable(tf.zeros(6)),
        'conv2' : tf.Variable(tf.zeros(16)),
        'flat3' : tf.Variable(tf.zeros(400)),
        'flat4' : tf.Variable(tf.zeros(120)),
        'flat5' : tf.Variable(tf.zeros(43))
    }

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1 = conv2d(x, weights['conv1'], biases['conv1'])

    # Activation
    conv1 = activation(conv1)

    # Dropout
    conv1 = dropout(conv1, keep_prob)

    # Layer 2: Convolutional. Input = 28x28x6. Output = 10x10x16.
    conv2 = conv2d(conv1, weights['conv2'], biases['conv2'], strides=2)

    # Activation.
    conv2 = activation(conv2)

    # Pooling
    conv2 = dropout(conv2, keep_prob)

    # Flatten. Input = 10x10x16. Output = 1600.
    flat3 = tf.reshape(conv2, [-1, weights['flat3'].get_shape().as_list()[0]])

    # Layer 3: Fully Connected. Input = 1600. Output = 400.
    flat3 = linear(flat3, weights['flat3'], biases['flat3'])

    # Activation.
    flat3 = activation(flat3)

    # Layer 4: Fully Connected. Input = 400. Output = 120.
    flat4 = linear(flat3, weights['flat4'], biases['flat4'])

    # Activation.
    flat4 = activation(flat4)

    # Layer 5: Fully Connected. Input = 120. Output = 43.
    logits = linear(flat4, weights['flat5'], biases['flat5'])

    return logits


def Multi_Scale_LeNet(x):

    weights = {
        'conv1' : tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma)),
        'conv2' : tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma)),
        'flat3' : tf.Variable(tf.truncated_normal(shape=(1576, 400), mean = mu, stddev = sigma)),
        'flat4' : tf.Variable(tf.truncated_normal(shape=(400, 100), mean = mu, stddev = sigma)),
        'flat5' : tf.Variable(tf.truncated_normal(shape=(100, 43), mean = mu, stddev = sigma))
    }
    biases = {
        'conv1' : tf.Variable(tf.zeros(6)),
        'conv2' : tf.Variable(tf.zeros(16)),
        'flat3' : tf.Variable(tf.zeros(400)),
        'flat4' : tf.Variable(tf.zeros(100)),
        'flat5' : tf.Variable(tf.zeros(43))
    }

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1 = conv2d(x, weights['conv1'], biases['conv1'])

    # Activation
    conv1 = activation(conv1)

    # Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = maxpool2d(conv1, 2)

    # Layer 2: Convolutional. Output = 10x10x16.
    conv2 = conv2d(conv1, weights['conv2'], biases['conv2'])

    # Activation.
    conv2 = activation(conv2)

    # Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = maxpool2d(conv2, 2)

    # Flatten. Input = 14x14x6 + 5x5x16. Output = 1576.
    flat_conv1 = tf.reshape(conv1, [-1, 14*14*6])
    flat_conv2 = tf.reshape(conv2, [-1, 5*5*16])
    flat3 = tf.concat(1, [flat_conv1, flat_conv2])

    # Layer 3: Fully Connected. Input = 1576. Output = 400.
    flat3 = linear(flat3, weights['flat3'], biases['flat3'])

    # Activation.
    flat3 = activation(flat3)

    # Layer 4: Fully Connected. Input = 400. Output = 100.
    flat4 = linear(flat3, weights['flat4'], biases['flat4'])

    # Activation.
    flat4 = activation(flat4)

    # Layer 5: Fully Connected. Input = 100. Output = 43.
    logits = linear(flat4, weights['flat5'], biases['flat5'])

    return logits

def Multi_Scale_LeNet_dropout(x, keep_prob):

    weights = {
        'conv1' : tf.Variable(tf.truncated_normal(shape=(5, 5, 3, 6), mean = mu, stddev = sigma)),
        'conv2' : tf.Variable(tf.truncated_normal(shape=(9, 9, 6, 16), mean = mu, stddev = sigma)),
        'flat3' : tf.Variable(tf.truncated_normal(shape=(6304, 1260), mean = mu, stddev = sigma)),
        'flat4' : tf.Variable(tf.truncated_normal(shape=(1260, 252), mean = mu, stddev = sigma)),
        'flat5' : tf.Variable(tf.truncated_normal(shape=(252, 43), mean = mu, stddev = sigma))
    }
    biases = {
        'conv1' : tf.Variable(tf.zeros(6)),
        'conv2' : tf.Variable(tf.zeros(16)),
        'flat3' : tf.Variable(tf.zeros(1260)),
        'flat4' : tf.Variable(tf.zeros(252)),
        'flat5' : tf.Variable(tf.zeros(43))
    }

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    conv1 = conv2d(x, weights['conv1'], biases['conv1'])

    # Activation
    conv1 = activation(conv1)

    # Dropout
    conv1 = dropout(conv1, keep_prob)

    # Layer 2: Convolutional. Input = 28x28x6. Output = 10x10x16.
    conv2 = conv2d(conv1, weights['conv2'], biases['conv2'], strides=2)

    # Activation.
    conv2 = activation(conv2)

    # Dropout
    conv2 = dropout(conv2, keep_prob)

    # Flatten. Input = 28x28x6 + 10x10x16. Output = 6304.
    flat_conv1 = tf.reshape(conv1, [-1, 28*28*6])
    flat_conv2 = tf.reshape(conv2, [-1, 10*10*16])
    flat3 = tf.concat(1, [flat_conv1, flat_conv2])

    # Layer 3: Fully Connected. Input = 6304. Output = 1260.
    flat3 = linear(flat3, weights['flat3'], biases['flat3'])

    # Activation.
    flat3 = activation(flat3)

    # Layer 4: Fully Connected. Input = 1260. Output = 252.
    flat4 = linear(flat3, weights['flat4'], biases['flat4'])

    # Activation.
    flat4 = activation(flat4)

    # Layer 5: Fully Connected. Input = 252. Output = 43.
    logits = linear(flat4, weights['flat5'], biases['flat5'])

    return logits



# ### Train, Validate and Test the Model

# A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
# sets imply underfitting. A high accuracy on the test set but low accuracy on the validation set implies overfitting.

# In[1]:

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

### TO TRY:
### 1) Decay the learning rate
### 2) Tweak epoch and batch_size

import math
import sys

from sklearn.utils import shuffle
from tqdm import tqdm

x = tf.placeholder(tf.float32, (None,) + image_shape)
y = tf.placeholder(tf.int32, (None))

keep_prob = tf.placeholder(tf.float32)

def evaluate(X_data, y_data, batch_size, accuracy_operation):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = X_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


def train_network(logits, model_file, rate=0.001, epochs=100, batch_size=128, keep_prob_val=1.0):
    one_hot_y = tf.one_hot(y, n_classes)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
    loss_operation = tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate = rate)
    training_operation = optimizer.minimize(loss_operation)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
    accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        #sess.run(tf.global_variables_initializer())

        batch_count = int(math.ceil(n_examples/batch_size))

        print("Training {}...".format(model_file))
        print()
        sys.stdout.flush()

        epoch_accuracy = []

        for i in range(epochs):
            local_X_train, local_y_train = shuffle(X_train, y_train)

            # Progress bar
            batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(i+1, epochs), unit='batches')

            for batch_i in batches_pbar:
                offset = batch_i * batch_size
                end = offset + batch_size
                batch_x, batch_y = local_X_train[offset:end], local_y_train[offset:end]
                sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: keep_prob_val})

            validation_accuracy = evaluate(X_valid, y_valid, batch_size, accuracy_operation)
            epoch_accuracy.append("{}: {:.3f}".format(i+1, validation_accuracy))

        ### Save model and epoch accuracy information
        saver.save(sess, './{}'.format(model_file))

        with open('./{}.accuracy'.format(model_file), mode='w') as f:
            f.write('----------------------\n')
            f.write('      Parameters\n')
            f.write('----------------------\n')
            f.write('Learn Rate: {}\n'.format(rate))
            f.write('Epochs:     {}\n'.format(epochs))
            f.write('Batch Size: {}\n'.format(batch_size))
            f.write('Keep Prob:  {}\n'.format(keep_prob_val))
            f.write('-----------------------\n')
            f.write('\n'.join(epoch_accuracy))
            f.write('\n')

        print("Accuracy: {} -> {}".format(epoch_accuracy[0], epoch_accuracy[-1]))
        print("Model saved: {}".format(model_file))


train_network(LeNet(x), "LeNet_balancedjitter_e20", epochs=20)
train_network(LeNet_dropout(x, keep_prob), "LeNet_dropout_balancedjitter_e100_kp0_5", epochs=100, keep_prob_val=0.5)
train_network(Multi_Scale_LeNet(x), "Multi_Scale_LeNet_balancedjitter_e20", epochs=20)
train_network(Multi_Scale_LeNet_dropout(x, keep_prob), "Multi_Scale_LeNet_dropout_balancedjitter_e100_kp0_5", epochs=100, keep_prob_val=0.5)


# ---
# 
# ## Step 3: Test a Model on New Images
# 
# To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.
# 
# You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.

# ### Load and Output the Images

# In[ ]:

### Load the images and plot them here.
### Feel free to use as many code cells as needed.


# ### Predict the Sign Type for Each Image

# In[3]:

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.


# ### Analyze Performance

# In[4]:

### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.


# ### Output Top 5 Softmax Probabilities For Each Image Found on the Web

# For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 
# 
# The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.
# 
# `tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.
# 
# Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tk.nn.top_k` is used to choose the three classes with the highest probability:
# 
# ```
# # (5, 6) array
# a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
#          0.12789202],
#        [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
#          0.15899337],
#        [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
#          0.23892179],
#        [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
#          0.16505091],
#        [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
#          0.09155967]])
# ```
# 
# Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:
# 
# ```
# TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
#        [ 0.28086119,  0.27569815,  0.18063401],
#        [ 0.26076848,  0.23892179,  0.23664738],
#        [ 0.29198961,  0.26234032,  0.16505091],
#        [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
#        [0, 1, 4],
#        [0, 5, 1],
#        [1, 3, 5],
#        [1, 4, 3]], dtype=int32))
# ```
# 
# Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.

# In[6]:

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.


# > **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the IPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run. You can then export the notebook by using the menu above and navigating to  \n",
#     "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 

# ### Project Writeup
# 
# Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 