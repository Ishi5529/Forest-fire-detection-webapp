forest fire detection webapp
ABSTRACT
This project deals with the detection and management of forest fire with the 
combined technology. Forest fire are very common which is a massive 
disaster to the environment and wildlife. In order to protect these, there 
need to be taken early caution measures to control the spreading fire. 
Usually it requires massive dependency of man power, transportation 
facility and lagging to trace true area will leads to delay in taking actions. 
Through this look up we have come up with the solution for this by 
implementing the Machine Learning algorithms. Where various sensors 
detects the fluctuation in the temperature, oxygen and humidity 
continuously. 
Python language is incredibly easy to use and learn for newcomers. The 
python language is one of the most accessible programming language 
available because it has simplified syntax & not complicated , which gives 
more emphasis on natural language. By using Streamlit module we have 
created a webapp that will prevent the forest fire. By this we can avoid 
major loss and spreading of fire to large area at its early stage.
Keywords:Forest fire, Python, Machine Learning, 

CONTENTS
S.NO. TOPIC PAGE NO.
1 BASICS OF PYTHON 1
2 ADVANCE PYTHON 38
3 INTRODUCTION OF THE PROJECT – FOREST FIRE PREVENTION 
WEBAPP
55
4 CAUSES OF FOREST FIRE 56
5 SOFTWARE AND LIBRARIES USED 57
6 MECHANISM INVOLVED IN MACHINE LEARNING PREDICTION 
OF FOREST FIRE
60
7 PROGRAM CODE 64
8 RESULT 66
9 ADVANTAGES AND DISADVANTAGES 69
10 FUTURE SCOPE 70
11 CONCLUSION 70
12 BIBLIOGRAPHY 

ADAVANCED PYTHON:
NUMPY LIBRARY
NumPy is a general-purpose array-processing package. It provides a 
high-performance multidimensional array object, and tools for working 
with these arrays.
It is the fundamental package for scientific computing with Python. It 
contains various features including these important ones:
• A powerful N-dimensional array object
• Sophisticated (broadcasting) functions
• Useful linear algebra, Fourier transform, and random number 
capabilities
Besides its obvious scientific uses, NumPy can also be used as an 
efficient multi-dimensional container of generic data.
Arbitrary data-types can be defined using NumPy which allows NumPy to 
seamlessly and speedily integrate with a wide variety of databases.
Arrays in NumPy:
NumPy’s main object is the homogeneous multidimensional array.
• It is a table of elements (usually numbers), all of the same type, 
indexed by a tuple of positive integers.
• In NumPy dimensions are called axes. The number of axes is rank.
• NumPy’s array class is called ndarray. It is also known by the 
alias array.
Array creation:
There are various ways to create arrays in NumPy.
• For example, you can create an array from a regular Python 
list or tuple using the array function. The type of the resulting array is 
deduced from the type of the elements in the sequences.
• Often, the elements of an array are originally unknown, but its size is 
known. Hence, NumPy offers several functions to create arrays 
with initial placeholder content. These minimize the necessity of 
growing arrays, an expensive operation.
For example: np.zeros, np.ones, np.full, np.empty, etc.
• To create sequences of numbers, NumPy provides a function 
analogous to range that returns arrays instead of lists.
• arange: returns evenly spaced values within a given 
interval. step size is specified.
• linspace: returns evenly spaced values within a given 
interval. num no. of elements are returned.
• Reshaping array: We can use reshape method to reshape an array. 
Consider an array with shape (a1, a2, a3, …, aN). We can reshape 
and convert it into another array with shape (b1, b2, b3, …, bM). The 
only required condition is:
a1 x a2 x a3 … x aN = b1 x b2 x b3 … x bM . (i.e. original size of array 
remains unchanged.)
• Flatten array: We can use flatten method to get a copy of array 
collapsed into one dimension. It accepts order argument. Default 
value is ‘C’ (for row-major order). Use ‘F’ for column major order.
Array Indexing:
Knowing the basics of array indexing is important for analysing and 
manipulating the array object. NumPy offers many ways to do array 
indexing.
• Slicing: Just like lists in python, NumPy arrays can be sliced. As 
arrays can be multidimensional, you need to specify a slice for each 
dimension of the array.
• Integer array indexing: In this method, lists are passed for indexing 
for each dimension. One to one mapping of corresponding elements is 
done to construct a new arbitrary array.
• Boolean array indexing: This method is used when we want to pick 
elements from array which satisfy some condition.
Basic operations:
Plethora of built-in arithmetic functions are provided in NumPy.
• Operations on single array: We can use overloaded arithmetic 
operators to do element-wise operation on array to create a new array. 
In case of +=, -=, *= operators, the existing array is modified.
1. Unary operators: Many unary operations are provided as a method 
of ndarray class. This includes sum, min, max, etc. These functions 
can also be applied row-wise or column-wise by setting an axis 
parameter.
2. Binary operators: These operations apply on array elementwise and 
a new array is created. You can use all basic arithmetic operators like 
+, -, / etc. In case of +=, -=, = operators, the existing array is modified.
Sorting array:
There is a simple np. sort method for sorting NumPy arrays

FIRE FOREST PREVENTION WEBAPP
INTRODUCTION:
• Wildfire, also called forest, bush or vegetation fire, can be 
described as any uncontrolled and non -prescribed combustion or 
burning of plants in a natural setting such as a forest, grassland, 
brush land or tundra, which consumes the natural fuels and 
spreads based on environmental conditions (e.g. wind, 
topography). 
• Depending on the type of vegetation present, a wildfire can also be 
classified more specifically as a forest fire, brush fire, bushfire (in 
Australia), desert fire, grass fire, hill fire, peat fire, prairie fire, 
vegetation fire, or veld fire.
• The occurrence of wildfires throughout the history of terrestrial life 
invites conjecture that fire must have had pronounced evolutionary 
effects on most ecosystems' flora and fauna. Earth is an 
intrinsically flammable planet owing to its cover of carbon-rich 
vegetation, seasonally dry climates, atmospheric oxygen, and 
widespread lightning and volcanic ignitions.

SOFTWARE AND LIBRARIES USED:
SOFTWARE USED:
PYCHARM:
• PyCharm is an integrated development environment (IDE) used 
in computer programming, specifically for the Python language.
• It is developed by the Czech company JetBrains. It provides code 
analysis, a graphical debugger, an integrated unit tester, 
integration with version control systems (VCSes), and supports 
web development with Django as well as data 
science with Anaconda.
• PyCharm is cross-platform with Windows, MacOS and linux version.


LIABRARIES USED:
1. NUMPY
NumPy, which stands for Numerical Python, is a library consisting of 
multidimensional array objects and a collection of routines for processing 
those arrays. NumPy is a Python library used for working with arrays. It 
was created in 2005 by Travis Oliphant.
It is a Python Extension whose purpose is to provide functions and 
capability to transform arrays. NumPy handles large datasets effectively 
and efficiently.
2. PANDAS
Pandas is an open-source, BSD-licensed Python library providing highperformance, easy-to-use data structures and data analysis tools for the
Python programming language. The name Pandas is derived from the 
word Panel Data – an Econometrics from Multidimensional data.
Pandas is used to analyze data. Pandas allows importing data from 
various file formats such as comma-separated values, SQL, Microsoft 
Excel. 
Using Pandas, we can accomplish five typical steps in the processing 
and analysis of data, regardless of the origin of data — load, prepare, 
manipulate, model, and analyze. Pandas allows various data 
manipulation operations such as merging, reshaping, selecting, as well 
as data cleaning.
3. PICKLE
Pickle in Python is primarily used in serializing and deserializing a 
Python object structure. In other words, it’s the process of converting a 
Python object into a byte stream to store it in a file/database, maintain 
program state across sessions, or transport data over the network. 
Pickle module accepts any Python object and converts it into a string 
representation and dumps it into a file by using dump function. This 
process is called pickling. 
The pickled byte stream can be used to re-create the original object 
hierarchy by unpickling the stream. The process of retrieving original 
Python objects from the stored string representation is called unpickling.
4. STREAMLIT
Now, with an app framework called Streamlit, data scientists can build 
machine learning tools right from the beginning of the project. Utilising 
Streamlit, data scientists can visualise their code output while analysing 
data. They can build ML tools that can be utilised to analyse data
through clicks and sliding bars.
We have used sublime to write the Python code and used the anaconda 
terminal to run the Python file using Streamlit run app.py. This will open 
a server on your browser, where you can interact with the user interface.
5. SKLEARN
Scikit-learn is probably the most useful library for machine learning in 
Python. The sklearn library contains a lot of efficient tools for machine 
learning and statistical modeling including classification, regression, 
clustering and dimensionality reduction.
Sklearn is used to build machine learning models.
Scikit-learn provides a range of supervised and unsupervised learning 
algorithms via a consistent interface in Python.
Groups of models provided by scikit-learn include :
• Clustering: for grouping unlabelled data such as KMeans.
• Cross Validation: for estimating the performance of supervised 
models on unseen data.
• Datasets: for test datasets and for generating datasets with 
specific properties for investigating model behaviour.
• Dimensionality Reduction: for reducing the number of attributes in 
data for summarization, visualization and feature selection such as 
Principal component analysis.
• Ensemble methods: for combining the predictions of multiple 
supervised models.
• Feature extraction: for defining attributes in image and text data.
• Feature selection: for identifying meaningful attributes from which 
to create supervised models.
• Parameter Tuning: for getting the most out of supervised models.
• Manifold Learning: For summarizing and depicting complex multidimensional data.
• Supervised Models: a vast array not limited to generalized linear 
models, discriminate analysis, naive bayes, lazy methods, neural 
networks, support vector machines and decision trees.
6. WARNINGS
• The warning module is actually a subclass of Exception which is a 
built-in class in Python. The warnings module was introduced 
in PEP 230 as a way to warn programmers about changes in 
language or library features in anticipation of backwards 
incompatible changes coming with Python 3.0.
• The warn() function defined in the ‘warning’ module is used to
show the warning messages. The warning filter in 
Python handles warnings (presented, disregarded or raised to 
exceptions). A filter consists of 5 parts, 
the action, message, category, module, and line number. When a 
warning is generated, it is compared against all of the registered 
filters. The first filter that matches controls the action taken for the 
warning. If no filter matches, the default action is taken.

MECHANISM INVOLVED IN MACHINE 
LEARNING PREDICTION OF FOREST FIRE:
DETAILED DESCRIPTION OF STEPS INVOLVED:
1) Create a Machine Learning model and drain it.
STEPS FOR CREATING MACHINE LEARNING MODEL:
We can define the machine learning workflow in 5 stages.
1. Gathering data
2. Data pre-processing
3. Researching the model that will be best for the type of data
4. Training and testing the model
5. Evaluation
STEP 1-DATA GATHERING:
The data set can be collected from various sources such as a file, 
database, sensor and many other such sources but the collected data 
cannot be used directly for performing the analysis process as there 
might be a lot of missing data, extremely large values, unorganized text 
data or noisy data. Therefore, to solve this problem Data Preparation is 
done. We have collected our data set from the website called Kaggle. 
Kaggle is one of the most visited websites that is used for practicing 
machine learning algorithms, they also host competitions in which people 
can participate and get to test their knowledge of machine learning.
STEP 2-DATA PRE-PROCESSING:
Data pre-processing is one of the most important steps in machine 
learning. It is the most important step that helps in building machine 
learning models more accurately. Data pre-processing is a process of 
cleaning the raw data i.e. the data is collected in the real world and is 
converted to a clean data set. In other words, whenever the data is 
gathered from different sources it is collected in a raw format and this 
data isn’t feasible
for the analysis. Therefore, certain steps are executed to convert the data 
into a small clean data set, this part of the process is called as data preprocessing. The duplicate elements, errors and bias needs to be 
removed.
As we know that data pre-processing is a process of cleaning the raw 
data into clean data, so that can be used to train the model. So, we 
definitely need data pre-processing to achieve good results from the
applied model in machine learning and deep learning projects. The 
cleaned dataset is then split into training and test data sets. The training 
set is used to train the model, while the test data is used to validate the 
model. The typical default is a 70/30 split between training and test sets.
STEP 3-RESEARCHING THE MODEL THAT WILL BE BEST FOR THE 
TYPE OF DATA:
Our main goal is to train the best performing model possible, using the 
pre-processed data.
Supervised Learning:
In Supervised learning, an AI system is presented with data which is 
labelled, which means that each data tagged with the correct label.
The supervised learning is categorized into 2 other categories which are 
“Classification” and “Regression”. We have used Regression in our 
project. These some most used regression algorithms :Linear 
Regression, Support Vector Regression, Decision Tress/Random Forest 
and Logistic Regression. In our project we have used Logistic 
Regression due to its highest accuracy which is 91.7%.
LOGISTIC REGRESSION:
Logistic regression is one of the most popular Machine Learning 
algorithms, which comes under the Supervised Learning technique. It is 
used for predicting the categorical dependent variable using a given set 
of independent variables.
Logistic regression predicts the output of a categorical dependent 
variable. It can be either Yes or No, 0 or 1, true or False, etc. but 
instead of giving the exact value as 0 and 1, it gives the probabilistic 
values which lie between 0 and 1.
The Logistic Regression uses a more complex cost function, this cost 
function can be defined as the ‘Sigmoid function’ or also known as the 
‘logistic function’ instead of a linear function.
The sigmoid function is used to map predictions to probabilities.
STEP 4-TRAIN AND TESTING THE MODEL
The training dataset is given to the chosen classification model (i.e. 
Logistic Regression model here) for learning. The data set connects to 
an algorithm, and the algorithm leverages sophisticated mathematical 
modelling to learn and develop predictions. 
Significance of Training : Training data is the main and most important 
data which helps machines to learn and make the predictions. For 
machine learning models to understand how to perform various actions, 
training datasets must first be fed into the machine learning algorithm. 
Training the model with the best data that are precisely annotated and 
labelled can help the model to achieve the best level of accuracy at 
affordable cost.
After well training and testing of the model the classification model 
becomes fully trained. The trained model is then tested against 
previously unseen data which is the representative of model 
performance in the real world and also helps in tuning the model. ML 
model validation is important to ensure the accuracy of model prediction 
to develop a right application. Testing the model helps to know whether 
model can correctly identify the new examples or not
STEP 5-EVALUATION
Model Evaluation is an integral part of the model development process. It 
helps to find the best model that represents our data and how well the 
chosen model will work.
2) After creating the model we dump it into a pickle file.
3) Then we build the frontend of web application using Streamlit and 
we take the inputs from the web application and supply it to the 
pickle file which contains the model.
4) Then we get the output back and display that onto the webpag

ADVANTAGES AND DISADVANTAGES
ADVANTAGES 
• Mapping the prediction of wildfire susceptibility is an essential 
component of emergency land management, wildfire prevention 
and the mitigation of fire impacts by on-time responses and 
recovery management. 
• Wildfire susceptibility maps have often been used to prioritize 
investments in the prevention of this hazard. 
• Regression has an effective method for estimating missing data 
and maintains accuracy when a large proportion of the data are 
missing. 
• It also runs efficiently on large datasets. It gives estimates of what 
variables are important in the classification.
DISADVANTAGES
• Fire spread by spotting (flying embers or firebrands) is not 
accounted for. 
• Vertical and horizontal fire whirlwinds are not modelled 
• The degree of accuracy in model predictions of wildland fire 
behaviour characteristics are dependent on the model’s 
applicability to a given situation

CONCLUSION:
• With forest fires ravaging the wildlands of Australia and more 
recently Simlipal National Reserve, the need for forest fire 
prediction has become ever more important. 
• The advancement of technology has enabled us to predict with 
great accuracy the occurrences of forest fires and take the 
necessary steps to prevent them from occurring.
• During this internship, the knowledge gained will be beneficial in 
implementing projects in various domains such as cyclone 
prediction, rain prediction, etc. as datasheets are widely available. 
• As engineers, this will help us to make products which will ensure 
that the environment is safe from hazards.


FUTURE SCOPE:
• Emergency system to notify authorities directly as soon as a fire 
emerges.
• Using sensors in the forest to get real time data. 
• Making a dedicated dataset for others to us

BIBLIOGRAPHY:
1) https://medium.com/analytics-vidhya/machine-learning-web-app-withstreamlit-7864e04c1fbf
2)https://towardsdatascience.com/towards-more-accurate-firepredictions-using-ai-ee6d5c26955c
3)https://medium.com/jovianml/forest-fire-spread-using-linearregression-with-pytorch-217e9309e016
4)https://medium.com/swlh/build-a-ml-web-app-for-stock-marketprediction-from-daily-news-with-streamlit-and-python-7c4cf918d9b4
5) https://dokumen.pub/deploy-machine-learning-models-to-productionwith-flask-streamlit-docker-and-kubernetes-on-google-cloud-platform1st-ed-9781484265451-9781484265468.htm

