---
layout: page
title: Guides
permalink: /guides/
---
# <center><a href="https://www.huffingtonpost.com/entry/what-is-machine-learning_us_5959a981e4b0f078efd98b33">What Is Machine Learning?</a></center>
<center>What is machine learning?</center>

# <center><a href="http://kseow.com/machinelearningsetup">Machine Learning Setup</a></center>
<center>Background information on machine learning and its nomenclature.</center>

# <center><a href="http://kseow.com/knn">K-Nearest Neighbors</a></center>
<center>K-Nearest Neighbors (KNN) is one of the easiest (but effective) algorithms in machine learning used for classification and regression. When I have a new piece of data that I want to classify, I simply take that new piece of data and compare it to every piece of data in my training set. I then store K (K can be set to any number) examples from the training set most similar to the new data point, and predict the majority label from those K examples. The similarity metric is usually computed by the Euclidean distance. There is no need to train or create a model. All I need is data and to set K.</center>

# <center><a href="http://kseow.com/kmeans">K-Means Clustering</a></center>
<center>K-Means is an unsupervised learning algorithm, meaning that you don’t have the label. The purpose of the algorithm is to partition the data into K (K can be set to any number) different groups so you can find some unknown structure and learn what they belong to. First, you start off with K clusters, randomly placed. Then you assign each data point to the cluster it’s closest to. You then take the average of each cluster and move each centroid (center of each cluster) to their averages. This process continues until convergence, successfully grouping the data and finding structure.</center>

# <center><a href="http://kseow.com/decisiontree">Decision Tree Learning</a></center>
<center>Decision Trees are supervised learning methods used for classification and regression. In a tree structure, nodes represent attributes of the data, branches represent conjunctions of features, and the leaves represent the labels. Using the C4.5 tree generation algorithm, at each node, C4.5 chooses the attribute of the data that most effectively splits its set of data so that each split’s labels are as homogenous as possible. The C4.5 algorithm then recurs on the smaller splits until convergence. The goal is to build rules to reach a conclusion (label), i.e. if color is orange and shape is round then it is an orange.</center>

# <center><a href="http://kseow.com/naivebayes">Naive Bayes</a></center>
<center>Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem. It’s actually fairly intuitive and simple to compute. Instead of looking for \(p(y|x)\)
 (probability of the label, given the data) directly like discriminative models, we calculate \(p(y)\)
 (probability of the label) and \(p(x|y)\)
 (probability of the data, given the label) to find \(p(y|x)\)
. This is known as a generative model. The naive part comes from the naive assumption that all data features are conditionally independent.</center>

# <center><a href="http://kseow.com/linearregression">Linear Regression</a></center>
<center>Linear regression is the most basic type of regression and commonly used in predictive analysis. Unlike the previous algorithms, linear regression can only be used for regression as it returns a real predicted value, i.e. 567 dollars per share, or predicting your son grows to be 6ft4.  It models the relationship between dependent variable \(y\)
 and one or more independent variables \(x\)
 by fitting a linear equation to observed data. Linear regression finds the linear trend within the data with weights (parameters), and uses that to predict real values. We use gradient descent or normal equations to find the proper weights.</center>

# <center><a href="http://kseow.com/logisticregression">Logistic Regression</a></center>
<center>Logistic regression, despite its name, is a linear model for classification rather than regression. Logistic regression is a supervised classification algorithm where the dependent variable \(y\)
 (label) is categorical, i.e. yes or no. It takes a linear combination of features with weights (parameters) and feeds it though a nonlinear squeezing function (sigmoid) that returns a probability between 0-1 of which class it belongs to. We learn the model, finding the proper weights, by using gradient descent. Those newfound weights will then be able to classify new data.</center>

# <center><a href="http://kseow.com/perceptron">Perceptron</a></center>
<center>The perceptron is a supervised learning algorithm that only works on linearly separable data, as its goal is to find a hyperplane to separate different classes. It is known as the single-layer neural network and is the most basic version of an artificial neuron. The perceptron takes in inputs with weights and runs it though a step-up function (instead of sigmoid like logistic regression) to see whether it should fire or not (return 1 or 0). Whenever the perceptron makes a mistake, the weights are updated on the spot to correct the mistake, progressively updating its hyperplane until it converges (finds separating hyperplane).</center>

# <center><a href="http://kseow.com/nn">Neural Networks</a></center>
<center>Artificial neural networks are powerful computational models based off of biological neural networks. The neural network has an input layer, hidden layer, and output layer, all connected by weights. The inputs, with interconnection weights, are fed into the hidden layer. Within the hidden layer, they get summed and processed by a nonlinear function. The outputs from the hidden layer, along with different interconnection weights, are processed one last time in the output layer to produce the neural network output. It learns its weights through backpropagation and is able to model any continuous non-linear function.</center>

# <center><a href="http://kseow.com/svm">Support Vector Machines</a></center>
<center>Recall that algorithms like the perceptron look for a separating hyperplane. However, there are many separating hyperplanes and none of them are unique. Intuitively, for an optimal answer, you want an algorithm that finds a hyperplane that maximizes the margin between different classes so that it can perform better on new data. Support Vector Machines (SVM) accomplishes this and finds a unique hyperplane by solving the primal and dual problem. SVMs can also be extended to classify non-linearly separable data by using soft margin classification and implicitly mapping inputs into high-dimensional feature spaces to classify them easier using the kernel trick.</center>

# <center><a href="http://kseow.com/ensemblemethods">Ensemble Methods</a></center>
<center>Ensemble methods is the art of combining diverse set of ‘weak’ classifiers together to create a strong classifier. They use multiple learning algorithms to obtain a better predictive performance than what could be obtained from any of the constituent learning algorithms alone. It means you build several different models then “fuse” the prediction of each model. The intuition is that the diversity of models can capture more aspects of the problem and gives better performance. We can create different models with only one set of training data by slightly adjusting the data for each model. We do that by using methods like boosting, or bagging.</center>

# <center><a href="http://kseow.com/rl">Reinforcement Learning</a></center>

# <center><a href="http://kseow.com/cnn">Convolutional Neural Networks</a></center>

# <center><a href="http://kseow.com/dbn">Deep Belief Networks</a></center>

# <center><a href="http://kseow.com/rnnlstm">Recurrent Neural Networks and Long Short Term Memory</a></center>

# <center><a href="http://kseow.com/gans">Generative Adversarial Networks</a></center>