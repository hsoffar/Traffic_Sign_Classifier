# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/sample.png "sample"
[image2]: ./images/graph.png "graph"
[image3]: ./images/samplenet.jpg "net"



### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.


* The size of training set is ?
* The size of the validation set is ?
* The size of test set is ?
      Number of training examples = 34799
      Number of testing examples = 12630
* The shape of a traffic sign image is ?
      Image data shape = (32, 32, 3)
* The number of unique classes/labels in the data set is ?
      Number of classes = 43



#### 2. Include an exploratory visualization of the dataset.

Here is a sample of data set

![alt text][image1]

Here is a graph visulazation of the dataset
![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

my preprocessign consisted of normalizign the data
-for some reason using gray scale didnt work for me , hence i have ignored it

I belive i should do more of data augumentation.
for example my idea was to get the original image, tilt it abit , darken it abit , hence i"ll be having mnore data
and i belive my model will work better



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

    # Layer 1: Convolutional. Input = 32x32x3. Output = 28x28x6.
    # Activation.
    I have choosed to use the droupout activation , it gave me better results than the relau
    # Pooling. Input = 28x28x6. Output = 14x14x6.
    # Layer 2: Convolutional. Output = 10x10x16.
    # Activation.
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    # Flatten. Input = 5x5x16. Output = 400.
    # Layer 3: Fully Connected. Input = 400. Output = 120.
    # Activation.
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    # Activation.
    # Layer 5: Fully Connected. Input = 84. Output = 43.




#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The model is using Adam optimizer to minimize loss function.
below are the settings i have used
EPOCHS = 50
BATCH_SIZE = 128
rate = 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
I have run 50 epochs
i have go around 95%
* validation set accuracy of ? 
 Validation Accuracy = 0.951

* test set accuracy of ?
accuracy in test set:  0.931512272179

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
-I have tried the LeNet arch

* What were some problems with the initial architecture?
-just the input images have to be resized, yet the output was bad.
also the LeNET was using single channel this traffic classifier uses three channels so that was adapted

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

-When i tried the Lenet it gave bad performnace i had to tune abit the architecture applying the preprocessing and changing the activation layer gave me great results i belive

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?



If a well known architecture was chosen:
* What architecture was chosen?
-LeNet

* Why did you believe it would be relevant to the traffic sign application?
-the traffic light signs has shapped and letters which the Lenet was working great with it

 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are several traffic sign which i choosed from the web
![alt text][image3]
 

 
#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

Prediction   Actual class
----------   ------------
     2             2     
    17            17     
     4             4     
    11            11     
    27            27     
    28            28     
    17            17     
    14            14     
    17            17     
    12            12     
    14            14     
    35            33     
    14            14     
    18            18     
    33            33     
    38            38     
    13            13  
Acuracy 94.117647 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the Step 3: Test a Model on New Images, Top 5 Softmax Probabilities For Each Image Found on the Web subsection.

     Actual values       :  2.0
-------prediction stats---------------
  1.000000000000000000000000000000% - [2]
  0.000000000005767225672720366347% - [1]
  0.000000000000000000000000027601% - [5]
  0.000000000000000000000000000000% - [6]
  0.000000000000000000000000000000% - [27]


        Actual values       :  17
-------prediction stats---------------
  1.000000000000000000000000000000% - [17]
  0.000000000000000001988355042964% - [0]
  0.000000000000000000013245489668% - [14]
  0.000000000000000000001618190754% - [3]
  0.000000000000000000000046976462% - [29]


        Actual values       :  4.0
-------prediction stats---------------
  0.999999880790710449218750000000% - [4]
  0.000000144463285778329009190202% - [1]
  0.000000000883284168029518923504% - [0]
  0.000000000000000490470087063748% - [3]
  0.000000000000000000000000000006% - [5]


        Actual values       :  11
-------prediction stats---------------
  1.000000000000000000000000000000% - [11]
  0.000000000000000001528729343864% - [30]
  0.000000000000000000102725246752% - [24]
  0.000000000000000000000042090639% - [27]
  0.000000000000000000000036798477% - [20]


        Actual values       :  27
-------prediction stats---------------
  0.999999165534973144531250000000% - [27]
  0.000000855038535974017577245831% - [24]
  0.000000000504238706344750653443% - [18]
  0.000000000000189131438917307382% - [11]
  0.000000000000000105793495287221% - [21]


        Actual values       :  28
-------prediction stats---------------
  1.000000000000000000000000000000% - [28]
  0.000000000000144306970344641738% - [29]
  0.000000000000000000733850651219% - [11]
  0.000000000000000000456406260457% - [24]
  0.000000000000000000049707286336% - [36]


        Actual values       :  17
-------prediction stats---------------
  0.999999880790710449218750000000% - [17]
  0.000000125970430531197052914649% - [3]
  0.000000000000075520027785578453% - [14]
  0.000000000000000000000105156320% - [10]
  0.000000000000000000000000031980% - [1]


        Actual values       :  14
-------prediction stats---------------
  1.000000000000000000000000000000% - [14]
  0.000000000000000000324872471394% - [1]
  0.000000000000000000000799542208% - [15]
  0.000000000000000000000091515930% - [5]
  0.000000000000000000000000007923% - [32]


        Actual values       :  17
-------prediction stats---------------
  1.000000000000000000000000000000% - [17]
  0.000000000000000000028374266027% - [10]
  0.000000000000000000000452378988% - [3]
  0.000000000000000000000028657048% - [12]
  0.000000000000000000000018405232% - [23]


        Actual values       :  12
-------prediction stats---------------
  1.000000000000000000000000000000% - [12]
  0.000000000000000000002589059565% - [14]
  0.000000000000000000000002370067% - [10]
  0.000000000000000000000000001850% - [17]
  0.000000000000000000000000000288% - [11]


        Actual values       :  14
-------prediction stats---------------
  1.000000000000000000000000000000% - [14]
  0.000000000000000000001438754660% - [17]
  0.000000000000000000000012351411% - [1]
  0.000000000000000000000000859266% - [15]
  0.000000000000000000000000529225% - [3]


        Actual values       :  33
-------prediction stats---------------
  0.556170880794525146484375000000% - [35]
  0.433882236480712890625000000000% - [33]
  0.009298550896346569061279296875% - [34]
  0.000648330140393227338790893555% - [40]
  0.000000000358079760287566273291% - [41]


        Actual values       :  14
-------prediction stats---------------
  1.000000000000000000000000000000% - [14]
  0.000000000000000000002586482906% - [17]
  0.000000000000000000000008029157% - [1]
  0.000000000000000000000000850854% - [15]
  0.000000000000000000000000388079% - [3]


        Actual values       :  18
-------prediction stats---------------
  1.000000000000000000000000000000% - [18]
  0.000000000000000000000000000006% - [27]
  0.000000000000000000000000000000% - [26]
  0.000000000000000000000000000000% - [0]
  0.000000000000000000000000000000% - [1]


        Actual values       :  33
-------prediction stats---------------
  1.000000000000000000000000000000% - [33]
  0.000000000000000000000008928277% - [39]
  0.000000000000000000000000010803% - [35]
  0.000000000000000000000000000992% - [37]
  0.000000000000000000000000000000% - [12]


        Actual values       :  38
-------prediction stats---------------
  1.000000000000000000000000000000% - [38]
  0.000000000000000000000000000000% - [36]
  0.000000000000000000000000000000% - [40]
  0.000000000000000000000000000000% - [0]
  0.000000000000000000000000000000% - [1]


        Actual values       :  13
-------prediction stats---------------
  1.000000000000000000000000000000% - [13]
  0.000000000000000000000000000000% - [0]
  0.000000000000000000000000000000% - [1]
  0.000000000000000000000000000000% - [2]
  0.000000000000000000000000000000% - [3]

