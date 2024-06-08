import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

colnames = ['Feature_1','Feature_2','Feature_3','Feature_4','class_labels']

X_train_data = pd.read_csv('train.data',names=colnames, header = None)
X_test_data = pd.read_csv('test.data',names = colnames, header = None)

X_train = X_train_data.drop(['class_labels'],axis = 1)

X_test = X_test_data.drop(['class_labels'], axis = 1)

def bold_underline(text):
    print("\033[4m\033[1m" + text)
    print("\033[0m")

def bold(text):
    print("\033[1m"+text+"\033[0m")

#
class Perceptron:

    def __init__(self,epochs):
        self.weights = None
        self.bias = None
        self.epochs = epochs
    
    def fit(self,X,y):
        n_samples, n_features = X.shape

        # Initializing the parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # looping over number of iterations
        for i in range(self.epochs):
            # looping over all values in our dataset
            for i in range(n_samples):
                # Calculating the activation function
                activation = np.dot(X.iloc[i],self.weights) + self.bias
                # Updating weight and bias
                if np.dot(y[i],activation) <= 0:
                    self.weights += np.dot(y[i],X.iloc[i])
                    self.bias += y[i]
        return self.weights,self.bias
        
    def predict(self,X):
        y_pred = np.dot(X,self.weights) + self.bias
        return np.sign(y_pred)
    
    def accuracies(self,y_actual,y_pred):
        accuracy = np.sum(y_actual == y_pred)/len(y_actual)
        return accuracy*100

#
print()
bold_underline("\033[1;31m" + "Results for Question 3")
# Data for training classifier class_1 and class_2
X_training_data_1_2 = X_train_data[X_train_data.class_labels != 'class-3']
X_train_1_2 = X_training_data_1_2.drop(['class_labels'],axis = 1)
y_train_1_2 = np.where(X_training_data_1_2.class_labels == 'class-1',1,-1)

# Data for testing classifier class_1 and class_2
X_testing_data_1_2 = X_test_data[X_test_data.class_labels != 'class-3']
X_test_1_2 = X_testing_data_1_2.drop(['class_labels'],axis = 1)
y_test_1_2 = np.where(X_testing_data_1_2.class_labels == 'class-1',1,-1)

# Perceptron Algorithm Implementation for classification of class_1 and class_2
bold_underline("Binary Classifier for class 1 and class 2")
classifier_1_2 = Perceptron(20)
classifier_1_2.fit(X_train_1_2,y_train_1_2)

# Predicting the test data 
y_predict_1_2 = classifier_1_2.predict(X_test_1_2)

# Accuracy of model on both train data and test data
train_accuracy = classifier_1_2.accuracies(y_train_1_2,classifier_1_2.predict(X_train_1_2))
print(f"Training Accuracy for class 1 and class 2: {train_accuracy}%")
test_accuracy = classifier_1_2.accuracies(y_test_1_2,y_predict_1_2)
print(f"Testing Accuracy for class 1 and class 2: {test_accuracy}%")

# Data for training classifier class_2 and class_3
X_training_data_2_3 = X_train_data[X_train_data.class_labels != 'class-1']
X_train_2_3 = X_training_data_2_3.drop(['class_labels'],axis = 1)
y_train_2_3 = np.where(X_training_data_2_3.class_labels == 'class-2',1,-1)

# Data for testing classifier class_2 and class_3
X_testing_data_2_3 =  X_test_data[X_test_data.class_labels != 'class-1']
X_test_2_3 = X_testing_data_2_3.drop(['class_labels'],axis = 1)
y_test_2_3 = np.where(X_testing_data_2_3.class_labels == 'class-2',1,-1)

# Perceptron Algorithm Implementation for classification of class_2 and class_3
print()
bold_underline("Binary Classifier for class 2 and class 3")
classifier_2_3 = Perceptron(20)
classifier_2_3.fit(X_train_2_3,y_train_2_3)

# Predicting the test data 
y_predict_2_3 = classifier_2_3.predict(X_test_2_3)

# Accuracy of model on both train data and test data
train_accuracy = classifier_2_3.accuracies(y_train_2_3,classifier_2_3.predict(X_train_2_3))
print(f"Training Accuracy for class 2 and class 3: {train_accuracy}%")
test_accuracy = classifier_2_3.accuracies(y_test_2_3,y_predict_2_3)
print(f"Testing Accuracy for class 2 and class 3: {test_accuracy}%")

#Data for training classifier class_1 and class_3
X_training_data_1_3 = X_train_data[X_train_data.class_labels != 'class-2']
X_train_1_3 = X_training_data_1_3.drop(['class_labels'],axis = 1)
y_train_1_3 = np.where(X_training_data_1_3.class_labels == 'class-1',1,-1)

# Data for testing classifier class_1 and class_3
X_testing_data_1_3 = X_test_data[X_test_data.class_labels != 'class-2']
X_test_1_3 = X_testing_data_1_3.drop(['class_labels'],axis = 1)
y_test_1_3 = np.where(X_testing_data_1_3.class_labels == 'class-1',1,-1)

# Perceptron Algorithm Implementation for classification of class_1 and class_3
print()
bold_underline("Binary Classifier for class 3 and class 1")
classifier_1_3 = Perceptron(20)
classifier_1_3.fit(X_train_1_3,y_train_1_3)

# Predicting the test data 
y_predict_1_3 = classifier_1_3.predict(X_test_1_3)

# Accuracy of model on both train data and test data
train_accuracy = classifier_1_3.accuracies(y_train_1_3,classifier_1_3.predict(X_train_1_3))
print(f"Training Accuracy for class 1 and class 3: {train_accuracy}%")
test_accuracy = classifier_1_3.accuracies(y_test_1_3,y_predict_1_3)
print(f"Testing Accuracy for class 1 and class 3: {test_accuracy}%")


# 
print()
bold_underline("\033[1;31m" + "Results for Question 4")
class Multi_Class_Perceptron:

    def __init__(self,epochs):
        self.weights = None
        self.bias = None
        self.epochs = epochs
    
    def fit(self,X,y):

        #Initializing the weights and bias
        self.n_samples, self.n_features = X.shape

        # Initializing the parameters
        self.weights = np.zeros(self.n_features)
        self.bias = 0

        # Initializing the list that will store all the weights and bias
        self.all_weights = []
        self.all_bias = []

        # Identify the classes
        classes = np.unique(y)

        # Training binary classifier for each class
        for c in classes:
            y_binary = np.where(y == c, 1,-1)

            # getting weights and bias for each class
            w, b = self.training(X,y_binary)

            # storing the values
            self.all_weights.append(w.copy())
            self.all_bias.append(b)
    
    def training(self,X,y):
        # looping over number of iterations
        for i in range(self.epochs):
            # looping over all values in our dataset
            for i in range(self.n_samples):
                # Calculating the activation function
                activation = np.dot(X.iloc[i],self.weights) + self.bias
                # Updating weight and bias
                if np.dot(y[i],activation) <= 0:
                    self.weights += np.dot(y[i],X.iloc[i])
                    self.bias += y[i]
        return self.weights,self.bias
    
    def predict(self,X ):
        prediction = np.dot(X,np.array(self.all_weights).T) + self.all_bias
        return np.argmax(prediction,axis = 1) + 1
    
    def accuracy(self,true,prediction):
        accuracy = np.sum(true == prediction)/len(true)
        return accuracy*100


# Preparing target dataset for training
y_train = np.where(X_train_data.class_labels == 'class-1',1,np.where(X_train_data.class_labels == 'class-2',2,3))

# Preparing target dataset for testing
y_test = np.where(X_test_data.class_labels == 'class-1',1,np.where(X_test_data.class_labels == 'class-2',2,3))


# Training Perceptron Algorithm
multiclass_classifier = Multi_Class_Perceptron(20)
multiclass_classifier.fit(X_train,y_train)

# Training Accuracy
training_prediction = multiclass_classifier.predict(X_train)
training_accuracy = multiclass_classifier.accuracy(y_train,training_prediction)
print("Training Accuracy", training_accuracy)

# Testing Accuracy
testing_prediction = multiclass_classifier.predict(X_test)
testing_accuracy = multiclass_classifier.accuracy(y_test,testing_prediction)
print("Testing Accuracy", testing_accuracy)


# 
print()
bold_underline("\033[1;31m" + "Results for Question 5")

class Multi_Class_Regularised_Perceptron:

    def __init__(self,epochs,l2):
        self.weights = None
        self.bias = None
        self.epochs = epochs
        self.l2 = l2
    
    def fit(self,X,y):

        #Initializing the weights and bias
        self.n_samples, self.n_features = X.shape

        # Initializing the parameters
        self.weights = np.zeros(self.n_features)
        self.bias = 0

        # Initializing the list that will store all the weights and bias
        self.all_weights = []
        self.all_bias = []

        # Identify the classes
        classes = np.unique(y)

        # Training binary classifier for each class
        for c in classes:
            y_binary = np.where(y == c, 1,-1)

            # getting weights and bias for each class
            w, b = self.training(X,y_binary)

            # storing the values
            self.all_weights.append(w.copy())
            self.all_bias.append(b.copy())
        
    def training(self,X,y):
        # looping over number of iterations
        for i in range(self.epochs):
            # looping over all values in our dataset
            for i in range(self.n_samples):
                # Calculating the activation function
                activation = np.dot(X.iloc[i],self.weights) + self.bias
                # Updating weight and bias
                if np.dot(y[i],activation) <= 0:
                    self.weights = np.dot((1-(2*self.l2)),self.weights) + np.dot(y[i],X.iloc[i])
                    self.bias += y[i]
                else:
                    self.weights = np.dot((1 - (2*self.l2)),self.weights)
        return self.weights,self.bias
    
    def predict(self,X ):
        prediction = np.dot(X,np.array(self.all_weights).T) + self.all_bias
        return np.argmax(prediction,axis = 1) + 1
    
    def accuracy(self,true,prediction):
        accuracy = np.sum(true == prediction)/len(true)
        return accuracy*100

bold_underline('Regularized Multi-Class Classifier')
for l2 in [0.01,0.1,1.0,10.0,100.0]:
    print()
    bold('L2 regularization coefficient: '+str(l2))
    print()
    multiclass_regularised_classifier = Multi_Class_Regularised_Perceptron(20,l2)
    multiclass_regularised_classifier.fit(X_train,y_train)

    # Training Accuracy
    training_prediction = multiclass_regularised_classifier.predict(X_train)
    training_accuracy = multiclass_regularised_classifier.accuracy(y_train,training_prediction)
    print("Training Accuracy", training_accuracy)

    # Testing Accuracy
    testing_prediction = multiclass_regularised_classifier.predict(X_test)
    testing_accuracy = multiclass_regularised_classifier.accuracy(y_test,testing_prediction)
    print("Testing Accuracy", testing_accuracy)


