#!/usr/bin/env python
# coding: utf-8

# <h1><center>Machine Learning Algorithms</center></h1>
# 
# <h1><center>By Felicia Fryer</center></h1>

# Python packages and modules

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import pydotplus
import matplotlib.image as mpimg
import itertools
import pylab as pl
import scipy.optimize as opt
import time

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.externals.six import StringIO
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score


get_ipython().run_line_magic('matplotlib', 'inline')


# Use popular machine learning algorithms: Decision Tree, Support Vector Machines, Neural Networks, k-Nearest Neighbors, and Boosting. I  will use these classification algorithms to build a model from historical data. Then you use the trained machine elarning algorithm to predict the target variable

# <div id="Medication dataset">
#     <h2>About the dataset</h2>
#     Imagine that you are a medical researcher compiling data for a study. You have collected data about a set of patients, all of whom suffered from the same illness. During their course of treatment, each patient responded to one of 5 medications, Drug A, Drug B, Drug c, Drug x and y. 
#     <br>
#     <br>
#     Part of your job is to build a model to find out which drug might be appropriate for a future patient with the same illness. The feature sets of this dataset are Age, Sex, Blood Pressure, and Cholesterol of patients, and the target is the drug that each patient responded to.
#     <br>
#     <br>
#     It is a sample of binary classifier, and you can use the training part of the dataset 
#     to build a decision tree, and then use it to predict the class of a unknown patient, or to prescribe it to a new patient.
# </div>
# 

# Question: Can we predict what drug a patient should take base on their features.

# now, read data using pandas dataframe:

# In[2]:


my_data = pd.read_csv("drug200.csv", delimiter=",")
my_data[0:5]


# ## Data Exploration

# In[3]:


# Data Size
my_data.shape


# In[4]:


ax = my_data[my_data['Drug'] == 'drugY'][:].plot(kind='scatter', x='Age', y='Na_to_K', color='DarkBlue', label='Drug Y');
my_data[my_data['Drug'] == 'drugX'][:].plot(kind='scatter', x='Age', y='Na_to_K', color='Yellow', label='Drug X', ax=ax);
my_data[my_data['Drug'] == 'drugA'][:].plot(kind='scatter', x='Age', y='Na_to_K', color='Red', label='Drug A', ax=ax);
my_data[my_data['Drug'] == 'drugB'][:].plot(kind='scatter', x='Age', y='Na_to_K', color='Green', label='Drug B', ax=ax);
my_data[my_data['Drug'] == 'drugC'][:].plot(kind='scatter', x='Age', y='Na_to_K', color='Orange', label='Drug C', ax=ax);


plt.title ("Distribution of the Classes based on Na_to_K and Age")
plt.show()


# In[5]:


df_p = my_data['Drug'].value_counts()


# In[6]:


print (df_p)


# In[7]:


ax = df_p.plot(kind='bar');
plt.title("Count of each Drug Class")
plt.show()


# In[8]:


ax = my_data[my_data['Drug'] == 'drugY'][:].plot(kind='bar', x='BP', y='Na_to_K', color='DarkBlue', label='Drug Y');
my_data[my_data['Drug'] == 'drugX'][:].plot(kind='bar', x='BP', y='Na_to_K', color='Yellow', label='Drug X', ax=ax);
my_data[my_data['Drug'] == 'drugA'][:].plot(kind='bar', x='BP', y='Na_to_K', color='Red', label='Drug A', ax=ax);
my_data[my_data['Drug'] == 'drugB'][:].plot(kind='bar', x='BP', y='Na_to_K', color='Green', label='Drug B', ax=ax);
my_data[my_data['Drug'] == 'drugC'][:].plot(kind='bar', x='BP', y='Na_to_K', color='Orange', label='Drug C', ax=ax);


plt.title ("Distribution of the Classes based on Blood Pressure and Na_to_K")
plt.show()


# In[9]:


#Frequency count of Drug Column
my_data['Drug'].value_counts()


# <div href="pre-processing">
#     <h2>Pre-processing</h2>
# </div>

# Using <b>my_data</b> as the Drug.csv data read by pandas, declare the following variables: <br>
# 
# <ul>
#     <li> <b> X </b> as the <b> Feature Matrix </b> (data of my_data) </li>
#     <li> <b> y </b> as the <b> response vector (target) </b> </li>
# </ul>

# Remove the column containing the target name since it doesn't contain numeric values.

# In[10]:


X = my_data[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]


# As you may figure out, some features in this dataset are categorical such as __Sex__ or __BP__. Unfortunately, Sklearn Decision Trees do not handle categorical variables. But still we can convert these features to numerical values. __pandas.get_dummies()__
# Convert categorical variable into dummy/indicator variables.

# In[11]:


le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1]) 


le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3]) 

X[0:5]


# Now we can fill the target variable.

# In[12]:


y = my_data["Drug"]
y[0:5]


# In[13]:


le_Drug = preprocessing.LabelEncoder()
le_Drug.fit(['drugA', 'drugB', 'drugC','drugX', 'drugY'])
y1 = le_Drug.transform(my_data['Drug'])
y1[:10]


# ## One Hot Encoding
# Use the get_dummies function in numpy to one-hot encode the data.

# In[31]:


my_data[0:5]


# In[32]:


# Make dummy variables for Sex, BP, Cholesterol, and Drug
one_hot_data = pd.concat([my_data, pd.get_dummies(my_data['Sex'], prefix='Sex'), 
                                   pd.get_dummies(my_data['BP'], prefix='BP'), 
                                   pd.get_dummies(my_data['Cholesterol'], prefix='Cholesterol')],
                                   axis=1)

# Drop the previous rank column
one_hot_data = one_hot_data.drop(['Sex', 'BP', 'Cholesterol'], axis=1)

# Print the first 10 rows of our data
one_hot_data[:10]


# In[33]:


# Drop Drug Column
X1 = one_hot_data.drop(['Drug'], axis=1)


# In[34]:


X1[:10]


# In[35]:


scaler = StandardScaler()

# Fit only to X1 - One Hot Encoding data
#scaler.fit(X_train)
scaler.fit(X1)

# Now apply the transformations to the data:
X1 = scaler.transform(X1)
#X_train = scaler.transform(X_train)
#X_test = scaler.transform(X_test)
X1[:10]


# In[36]:


X1[:10]


# ## Cross-Validation for Decision Tree 

# In[14]:


# 10-fold cross-validation with Decision Tree (before pruning)
dt = DecisionTreeClassifier(max_depth = 5)
scores = cross_val_score(dt, X, y1, cv=10, scoring='accuracy')
print(scores)
print(np.std(scores))
# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# In[15]:


# 10-fold cross-validation with Decision Tree (after pruning)
dt1 = DecisionTreeClassifier(max_depth = 4)
scores = cross_val_score(dt1, X, y1, cv=10, scoring='accuracy')
print(scores)

# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# In[16]:


# 10-fold cross-validation with Decision Tree (after pruning)
dt2 = DecisionTreeClassifier(max_depth = 3)
scores = cross_val_score(dt2, X, y1, cv=10, scoring='accuracy')
print(scores)

# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# In[17]:


# 10-fold cross-validation with Decision Tree (after pruning)
dt3 = DecisionTreeClassifier(max_leaf_nodes = 4)
scores = cross_val_score(dt3, X, y1, cv=10, scoring='accuracy')
print(scores)

# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# ## Cross Validation for KNN

# In[18]:


# search for an optimal value of K for KNN
k_range = list(range(1, 31))
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y1, cv=11, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)


# In[19]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(k_range, k_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-Validated Accuracy')
plt.title('Value of k for kNN vs 11-Fold Cross-Validation Accuracy')


# In[20]:


# 10-fold cross-validation with the best KNN model
knn = KNeighborsClassifier(n_neighbors=3)
print(cross_val_score(knn, X, y1, cv=11, scoring='accuracy').mean())


# In[21]:


# 15-fold cross-validation with the best KNN model
knn = KNeighborsClassifier(n_neighbors=3)
print(cross_val_score(knn, X, y1, cv=10, scoring='accuracy').mean())


# In[22]:


# 5-fold cross-validation with the best KNN model
knn = KNeighborsClassifier(n_neighbors=3)
print(cross_val_score(knn, X, y1, cv=12, scoring='accuracy').mean())


# ## Cross-validation for SVM

# In[23]:


# 11-fold cross-validation with SVM
sv = svm.SVC(kernel='rbf', gamma='auto')
scores = cross_val_score(sv, X, y1, cv=11, scoring='accuracy')
print(scores)
print(np.std(scores))
# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# In[24]:


# 11-fold cross-validation with SVM
sv1 = svm.SVC(kernel='linear', gamma='auto')
scores = cross_val_score(sv1, X, y1, cv=11, scoring='accuracy')
print(scores)
print(np.std(scores))
# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# In[25]:


# 11-fold cross-validation with SVM
sv2 = svm.SVC(kernel='poly', gamma='auto')
scores = cross_val_score(sv2, X, y1, cv=11, scoring='accuracy')
print(scores)
print(np.std(scores))
# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# ## Cross-Validation for Boosting

# In[26]:


# 10-fold cross-validation with Boosting
boost = AdaBoostClassifier(DecisionTreeClassifier(criterion = "gini", max_leaf_nodes = 5), 
                             n_estimators=200, algorithm="SAMME.R", learning_rate = 0.5)
scores = cross_val_score(boost, X, y1, cv=10, scoring='accuracy')
print(scores)
print(np.std(scores))
# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# In[27]:


# 10-fold cross-validation with Boosting
boost1 = AdaBoostClassifier(DecisionTreeClassifier(criterion = "gini", max_leaf_nodes = 5), 
                             n_estimators=200, algorithm="SAMME.R", learning_rate = 1.0)
scores = cross_val_score(boost1, X, y1, cv=10, scoring='accuracy')
print(scores)
print(np.std(scores))
# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# In[28]:


# 10-fold cross-validation with Boosting
boost2 = AdaBoostClassifier(DecisionTreeClassifier(criterion = "gini", max_leaf_nodes = 5), 
                             n_estimators=200, algorithm="SAMME.R", learning_rate = 0.25)
scores = cross_val_score(boost2, X, y1, cv=10, scoring='accuracy')
print(scores)
print(np.std(scores))
# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# In[29]:


# 10-fold cross-validation with Boosting
boost3 = AdaBoostClassifier(DecisionTreeClassifier(criterion = "gini", max_leaf_nodes = 5), 
                             n_estimators=200, algorithm="SAMME.R", learning_rate = 0.10)
scores = cross_val_score(boost3, X, y1, cv=10, scoring='accuracy')
print(scores)
print(np.std(scores))
# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# ## Cross-Validation for Neural Networks

# In[59]:


# 10-fold cross-validation with Neural Networks - 1 Layers
ann1 = MLPClassifier(hidden_layer_sizes=(9),max_iter=1500, solver='lbfgs')
scores = cross_val_score(ann1, X1, y1, cv=10, scoring='accuracy')
print(scores)
print(np.std(scores))
# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# In[60]:


# 10-fold cross-validation with Neural Networks - 2 Layers
ann2 = MLPClassifier(hidden_layer_sizes=(9,9),max_iter=1500, solver='lbfgs')
scores = cross_val_score(ann2, X1, y1, cv=10, scoring='accuracy')
print(scores)
print(np.std(scores))
# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# In[62]:


# 10-fold cross-validation with Neural Networks - 3 Layers
ann3 = MLPClassifier(hidden_layer_sizes=(9,9,9),max_iter=1500, solver='lbfgs')
scores = cross_val_score(ann3, X1, y1, cv=5, scoring='accuracy')
print(scores)
print(np.std(scores))
# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# In[76]:


# 10-fold cross-validation with Neural networks - 4 Layers
ann4 = MLPClassifier(hidden_layer_sizes=(9,9,9,9),max_iter=1500, solver='lbfgs')
scores = cross_val_score(ann4, X1, y1, cv=10, scoring='accuracy')
print(scores)
print(np.std(scores))
# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# In[74]:


# 10-fold cross-validation with Neural Networks - 3 Layers
ann5 = MLPClassifier(hidden_layer_sizes=(9),max_iter=3000, solver='lbfgs')
scores = cross_val_score(ann5, X1, y1, cv=10, scoring='accuracy')
print(scores)
print(np.std(scores))
# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# In[77]:


#from sklearn.linear_model import LogisticRegression
# 10-fold cross-validation with SoftMax 2750 Iterations to Converge
ann3 = LogisticRegression(multi_class="multinomial", solver='lbfgs', max_iter=2000)
scores = cross_val_score(ann3, X1, y1, cv=10, scoring='accuracy')
print(scores)
print(np.std(scores))
# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# In[78]:


#from sklearn.linear_model import LogisticRegression
# 10-fold cross-validation with SoftMax 2750 Iterations to Converge
ann4 = LogisticRegression(multi_class="multinomial", solver='lbfgs', max_iter=2000)
scores = cross_val_score(ann4, X, y1, cv=10, scoring='accuracy')
print(scores)
print(np.std(scores))
# use average accuracy as an estimate of out-of-sample accuracy
print(scores.mean())


# # Learning Curve

# In[81]:


train_sizes=np.linspace(0.1, 1.0, 10)

# SVC is more expensive so we do a lower number of CV iterations:
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)


# In[86]:


#Learning Curve Function
#The intention is to use mean squared error(MSE) metric, we use a proxy, negative MSE
#CV represents the k-fold cross-validation

def learning_curves(estimator, features, target, train_sizes, cv):
    train_sizes, train_scores, validation_scores = learning_curve(estimator, features, target, train_sizes = train_sizes, cv = cv, error_score = np.nan)

    train_scores_mean = train_scores.mean(axis = 1)
    ymax = train_scores.max()
    ylim = ymax * 1.5

    validation_scores_mean = validation_scores.mean(axis = 1)
    plt.plot(train_sizes, train_scores_mean, label = 'Training Score')
    plt.plot(train_sizes, validation_scores_mean, label = 'Cross-validation Score')
    plt.ylabel('Score', fontsize = 14)
    plt.xlabel('Training Set Size', fontsize = 14)
    title = 'Learning Curve for a ' + str(estimator).split('(')[0] + ' Model'
    plt.title(title, fontsize = 18, y = 1.03)
    plt.legend()
    plt.ylim(0,ylim)
    


# In[83]:


# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
#cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

# SVC is more expensive so we do a lower number of CV iterations:
#cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

#train_sizes, train_scores, validation_scores = learning_curve(svm.SVC(kernel = 'rbf', gamma = 0.001), X_train, y_train, train_sizes = np.linspace(0.1,1.0,10), cv = cv)


# In[87]:


#Plotting the Learning Curves for Decision Tree(Pruning), Neural Networks, Boosting, Support Vector Machines, and k-Nearest Neighbors
plt.figure(figsize = (16,5))

for model,i in [(DecisionTreeClassifier(criterion = 'gini', max_depth = 4), 1),(svm.SVC(kernel='linear', gamma = 'auto'), 2)]:
    plt.subplot(1,2,i)
    learning_curves(model, X, y1, train_sizes, cv)


# In[88]:


plt.figure(figsize = (16,5))
for model,i in [(AdaBoostClassifier(DecisionTreeClassifier(criterion = "gini", max_leaf_nodes = 5), 
                             n_estimators=200, algorithm="SAMME.R", learning_rate = 0.5), 1)]:
    plt.subplot(1,2,i)
    
    learning_curves(model, X, y1, train_sizes, cv)


# In[89]:


plt.figure(figsize = (16,10))
for model,i in [(MLPClassifier(hidden_layer_sizes=(9),max_iter=3000, solver='lbfgs'), 1), 
                (MLPClassifier(hidden_layer_sizes=(9),max_iter=1500, solver='lbfgs'), 2),
                (MLPClassifier(hidden_layer_sizes=(9,9),max_iter=1500, solver='lbfgs'), 3),
                (MLPClassifier(hidden_layer_sizes=(9,9),max_iter=3000, solver='lbfgs'), 4)]:
    plt.subplot(2,2,i)
    plt.subplots_adjust(hspace = 0.4)
    learning_curves(model, X1, y1, train_sizes, cv)


# In[91]:


plt.figure(figsize = (16,5))
for model,i in [(LogisticRegression(multi_class='multinomial',max_iter=2000, solver='lbfgs'), 1), 
                (LogisticRegression(multi_class='multinomial',max_iter=3000, solver='lbfgs'), 2)]:
    plt.subplot(1,2,i)
    plt.subplots_adjust(hspace = 0.4)
    learning_curves(model, X1, y1, train_sizes, cv)


# In[93]:


plt.figure(figsize = (16,10))
for model,i in [(KNeighborsClassifier(n_neighbors = 1), 1), 
                (KNeighborsClassifier(n_neighbors = 2), 2),
                (KNeighborsClassifier(n_neighbors = 3), 3),
                (KNeighborsClassifier(n_neighbors = 4), 4)]:
    plt.subplot(2,2,i)
    plt.subplots_adjust(hspace = 0.6)
    learning_curves(model, X1, y1, train_sizes, cv)  


# <hr>
# 
# <div id="setting_up_tree">
#     <h2>Setting up the Decision Tree</h2>
#     We will be using <b>train/test split</b> on our <b>decision tree</b>. Let's import <b>train_test_split</b> from <b>sklearn.cross_validation</b>.
# </div>

# Now <b> train_test_split </b> will return 4 different parameters. We will name them:<br>
# X_trainset, X_testset, y_trainset, y_testset <br> <br>
# The <b> train_test_split </b> will need the parameters: <br>
# X, y, test_size=0.3, and random_state=3. <br> <br>
# The <b>X</b> and <b>y</b> are the arrays required before the split, the <b>test_size</b> represents the ratio of the testing dataset, and the <b>random_state</b> ensures that we obtain the same splits.

# In[103]:


X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)


# <h3>Practice</h3>
# Print the shape of X_trainset and y_trainset. Ensure that the dimensions match

# In[104]:


print ('Train set:', X_trainset.shape,  y_trainset.shape)


# Print the shape of X_testset and y_testset. Ensure that the dimensions match

# In[105]:


print ('Test set:',X_testset.shape, y_testset.shape)


# <hr>
# 
# <div id="modeling">
#     <h2>Modeling</h2>
#     We will first create an instance of the <b>DecisionTreeClassifier</b> called <b>drugTree</b>.<br>
#     Inside of the classifier, specify <i> criterion="entropy" </i> so we can see the information gain of each node.
# </div>

# In[131]:


start_time = time.time()
drugTree = DecisionTreeClassifier(max_depth = 4)
#drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = None)
drugTree.fit(X_trainset,y_trainset)
print("---%s seconds ---" % (time.time() - start_time))


# In[132]:


drugTree


# <hr>
# 
# <div id="prediction">
#     <h2>Prediction</h2>
#     Let's make some <b>predictions</b> on the testing dataset and store it into a variable called <b>predTree</b>.
# </div>

# In[108]:


predTree = drugTree.predict(X_testset)


# You can print out <b>predTree</b> and <b>y_testset</b> if you want to visually compare the prediction to the actual values.

# In[109]:


print (predTree [0:5])
print (y_testset [0:5])


# <hr>
# 
# <div id="evaluation">
#     <h2>Evaluation</h2>
#     Next, let's import <b>metrics</b> from sklearn and check the accuracy of our model.
# </div>

# In[110]:


print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))


# In[111]:


print("Decision Tree before pruning F1_Score:", f1_score(y_testset, predTree, average='weighted'))


# In[112]:


print("Decision Tree before pruning Jaccard Score:", jaccard_similarity_score(y_testset, predTree))


# __Accuracy classification score__ computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.  
# 
# In multilabel classification, the function returns the subset accuracy. If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.
# 

# In[113]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# In[114]:


# Compute confusion matrix
dt_cnf_matrix = confusion_matrix(y_testset, predTree, labels=['drugA','drugB','drugC','drugX','drugY'])
np.set_printoptions(precision=2)

print (classification_report(y_testset, predTree))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(dt_cnf_matrix, classes=['DrugA','DrugB','DrugC', 'DrugX', 'DrugY'],normalize= False,  title='Decision Tree Before Pruning Confusion Matrix')


# <hr>
# 
# <div id="visualization">
#     <h2>Visualization</h2>
#     Lets visualize the tree
# </div>

# In[125]:


dot_data = StringIO()
filename = "drugtree.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')


# <hr>
# 
# <div id="pruning">
#     <h2>Pruning</h2>
#     Lets prune the decision tree to avoid overfitting
# </div>

# We can avoid overfitting by changing the parameters.
# 
# Pruning Parameters:
# <ul>
#     <li> <b>max_leaf_nodes - reduce the number of leaf nodes</b> </li>
#     <li> <b>min_samples_leaf - restrict the size of sample leaf</b> </li>
#     <li> <b>max_depth - reduce the depth of the tree to build a generalized tree</b> </li>
# </ul>

# In[133]:


#We will rebuild a new tree by using above data and see how it works by tweeking the parameteres
start_time = time.time()
drugTree2 = DecisionTreeClassifier(criterion = "gini", max_depth = 3)
drugTree2.fit(X_trainset,y_trainset)
print("---%s seconds ---" % (time.time() - start_time))


# In[134]:


drugTree2


# In[128]:


#Predict Prune Tree
predTree2 = drugTree2.predict(X_testset)


# <hr>
# 
# <div id="evaluation">
#     <h2>Evaluation</h2>
#     Next, let's import <b>metrics</b> from sklearn and check the accuracy of our prune model.
# </div>

# In[129]:


print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree2))


# In[130]:


print("Decision Tree after pruning F1_Score:", f1_score(y_testset, predTree2, average = 'weighted'))


# In[131]:


print("Decision Tree after pruning Jaccard Score:", jaccard_similarity_score(y_testset, predTree2))


# In[132]:


# Compute confusion matrix
dtp_cnf_matrix = confusion_matrix(y_testset, predTree2, labels=['drugA','drugB','drugC','drugX','drugY'])
np.set_printoptions(precision=2)

print (classification_report(y_testset, predTree2))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(dtp_cnf_matrix, classes=['DrugA','DrugB','DrugC', 'DrugX', 'DrugY'],normalize= False,  title='Decision Tree After Pruning Confusion Matrix')


# In[35]:


print (np.unique(y_trainset))


# <h3>Visualize the Prune Tree</h3>

# In[133]:


plt.clf
dot_data = StringIO()
filename = "drugtree2.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree2,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')


# 
# 
# <h1><center>Neural Networks</center></h1>

# ### Splitting the dataset into training and test

# In[120]:


X_train, X_test, y_train, y_test = train_test_split(X, y1, test_size=0.3, random_state=3)

print (X_train[0:5])
print (y_train[0:5])


# In[121]:


scaler = StandardScaler()

# Fit only to the training data
scaler.fit(X_train)
#scaler.fit(X)

#Now apply the transformations to the data:
#X_N = scaler.transform(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[122]:


print (X_train[0:5])
print (y_train[0:5])


# ## Training the 2-layer Neural Network
# The following function trains the 2-layer neural network. First, we'll write some helper functions.

# In[179]:


start_time = time.time()
mlp = MLPClassifier(hidden_layer_sizes=(9,9),max_iter=500, solver='lbfgs')
mlp.fit(X_train,y_train)
print("---%s seconds ---" % (time.time() - start_time))


# In[158]:


start_time = time.time()
softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", max_iter = 2750)
softmax_reg.fit(X_train, y_train)
print("---%s seconds ---" % (time.time() - start_time))


# In[170]:


mlp.fit(X_train,y_train)


# In[187]:


softmax_reg.fit(X_train,y_train)


# In[180]:


predictions = mlp.predict(X_test)


# In[188]:


predictions_soft = softmax_reg.predict(X_test)


# In[181]:


print(confusion_matrix(y_test,predictions))


# In[189]:


print (confusion_matrix(y_test, predictions_soft))


# In[182]:


print(classification_report(y_test,predictions))


# In[190]:


print(classification_report(y_test, predictions_soft))


# In[183]:


print("ANN Accuracy: ", metrics.accuracy_score(y_test, predictions))


# In[191]:


print("Softmax Accuracy: ", metrics.accuracy_score(y_test, predictions_soft))


# In[184]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, predictions, labels=[0,1,2,3,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, predictions))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['DrugA(0)','DrugB(1)','DrugC(2)', 'DrugX(3)', 'DrugY(4)'],normalize= False,  title='ANN Confusion Matrix')


# In[192]:


# Compute confusion matrix
cnf_matrix1 = confusion_matrix(y_test, predictions_soft, labels=[0,1,2,3,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, predictions_soft))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix1, classes=['DrugA(0)','DrugB(1)','DrugC(2)', 'DrugX(3)', 'DrugY(4)'],normalize= False,  title='Softmax Confusion Matrix')


# In[185]:


f1_score(y_test, predictions, average='weighted') 


# In[186]:


jaccard_similarity_score(y_test, predictions)


# In[193]:


jaccard_similarity_score(y_test, predictions_soft)


# In[ ]:





# 
# 
# <h1><center>Support Vector Machine</center></h1>
# 
# 

# The SVM algorithm offers a choice of kernel functions for performing its processing. Basically, mapping data into a higher dimensional space is called kernelling. The mathematical function used for the transformation is known as the kernel function, and can be of different types, such as:
# 
#     1.Linear
#     2.Polynomial
#     3.Radial basis function (RBF)
#     4.Sigmoid
# Each of these functions has its characteristics, its pros and cons, and its equation, but as there's no easy way of knowing which function performs best with any given dataset, we usually choose different functions in turn and compare the results. Let's just use the default, RBF (Radial Basis Function) for this lab.

# In[194]:


#Different Kernels
clf_radial = svm.SVC(kernel='rbf', gamma = 'auto')
clf_radial.fit(X_train, y_train) 


# In[135]:


start_time = time.time()
clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(X_train, y_train) 
print("---%s seconds ---" % (time.time() - start_time))


# In[196]:


clf_poly = svm.SVC(kernel='poly')
clf_poly.fit(X_train, y_train) 


# In[197]:


clf_sig = svm.SVC(kernel='sigmoid')
clf_sig.fit(X_train, y_train) 


# In[198]:


yhat_radial = clf_radial.predict(X_test)


# In[199]:


f1_score(y_test, yhat_radial, average='weighted') 


# In[200]:


yhat_linear = clf_linear.predict(X_test)


# In[201]:


f1_score(y_test, yhat_linear, average='weighted') 


# In[202]:


yhat_poly = clf_poly.predict(X_test)


# In[203]:


f1_score(y_test, yhat_poly, average='weighted') 


# In[204]:


yhat_sig = clf_sig.predict(X_test)


# In[205]:


f1_score(y_test, yhat_sig, average='weighted') 


# In[206]:


# Compute confusion matrix
cnf_matrix_linear = confusion_matrix(y_test, yhat_linear, labels=[0,1,2,3,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat_linear))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_linear, classes=['DrugA(0)','DrugB(1)','DrugC(2)', 'DrugX(3)', 'DrugY(4)'],normalize= False,  title='SVM Linear Confusion Matrix')


# In[207]:


jaccard_similarity_score(y_test, yhat_linear)


# <h1><center>k-Nearest Neighbors</center></h1>

# #### Different K values
# K in KNN, is the number of nearest neighbors to examine. It is supposed to be specified by the User. So, how can we choose right value for K?
# The general solution is to reserve a part of your data for testing the accuracy of the model. Then chose k =1, use the training part for modeling, and calculate the accuracy of prediction using all samples in your test set. Repeat this process, increasing the k, and see which k is the best for your model.
# 
# We can calculate the accuracy of KNN for different Ks.

# In[123]:


Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc


# #### Plot  model accuracy  for Different number of Neighbors 

# In[124]:


plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Nabors (K)')
plt.tight_layout()
plt.show()


# In[125]:


print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


# ### Training
# 
# k = 1 has the best accuracy value

# In[136]:


k = 1
start_time = time.time()
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
print("---%s seconds ---" % (time.time() - start_time))
print (neigh)


# In[127]:


yhat_k1 = neigh.predict(X_test)


# In[128]:


f1_score(y_test, yhat_k1, average='weighted') 


# In[129]:


jaccard_similarity_score(y_test, yhat_k1)


# In[130]:


# Compute confusion matrix
cnf_matrix_k1 = confusion_matrix(y_test, yhat_k1, labels=[0,1,2,3,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat_k1))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_k1, classes=['DrugA(0)','DrugB(1)','DrugC(2)','DrugX(3)','DrugY(4)'],normalize= False,  title='KNN K=1 Confusion Matrix')


# <h1><center>Boosting</center></h1>

# In[137]:


start_time = time.time()
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(criterion = "gini", max_leaf_nodes = 5), 
                             n_estimators=200, algorithm="SAMME.R", learning_rate = 0.5)
ada_clf.fit(X_train, y_train)
print("---%s seconds ---" % (time.time() - start_time))


# In[138]:


ada_clf


# In[86]:


yhat_ada = ada_clf.predict(X_test)


# In[87]:


f1_score(y_test, yhat_ada, average='weighted') 


# In[88]:


jaccard_similarity_score(y_test, yhat_ada)


# In[89]:


# Compute confusion matrix
cnf_matrix_ada = confusion_matrix(y_test, yhat_ada, labels=[0,1,2,3,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat_ada))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix_ada, classes=['DrugA(0)','DrugB(1)','DrugC(2)','DrugX(3)','DrugY(4)'],normalize= False,  title='AdaBoost Confusion Matrix')


# # The Second DataSet

# <h2 id="load_dataset">Load the Cancer data</h2>
# The example is based on a dataset that is publicly available from the UCI Machine Learning Repository (Asuncion and Newman, 2007)[http://mlearn.ics.uci.edu/MLRepository.html]. The dataset consists of several hundred human cell sample records, each of which contains the values of a set of cell characteristics. The fields in each record are:
# 
# |Field name|Description|
# |--- |--- |
# |ID|Identifier|
# |Clump|Clump thickness|
# |UnifSize|Uniformity of cell size|
# |UnifShape|Uniformity of cell shape|
# |MargAdh|Marginal adhesion|
# |SingEpiSize|Single epithelial cell size|
# |BareNuc|Bare nuclei|
# |BlandChrom|Bland chromatin|
# |NormNucl|Normal nucleoli|
# |Mit|Mitoses|
# |Class|Benign or malignant|
# 
# <br>
# <br>

# In[90]:


cancer_data = pd.read_csv("cancer.csv", delimiter=",")
cancer_data[0:5]


# The ID field contains the patient identifiers. The characteristics of the cell samples from each patient are contained in fields Clump to Mit. The values are graded from 1 to 10, with 1 being the closest to benign.
# 
# The Class field contains the diagnosis, as confirmed by separate medical procedures, as to whether the samples are benign (value = 2) or malignant (value = 4).
# 
# Lets look at the distribution of the classes based on Clump thickness and Uniformity of cell size:

# In[91]:


cancer_data.shape


# In[92]:


ax = cancer_data[cancer_data['Class'] == 4][:].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cancer_data[cancer_data['Class'] == 2][:].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.title ("Distribution of the Classes based on Clump Thickness and Uniformity of Cell Size")
plt.show()


# ## Data Preprocessing and Selection

# In[93]:


cancer_data.dtypes


# It looks like the __BareNuc__ column includes some values that are not numerical. We can drop those rows:

# In[94]:


cancer_data = cancer_data[pd.to_numeric(cancer_data['BareNuc'], errors='coerce').notnull()]
cancer_data['BareNuc'] = cancer_data['BareNuc'].astype('int')
cancer_data.dtypes


# In[95]:


feature_df = cancer_data[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
X[0:5]


# In[96]:


cancer_data['Class'] = cancer_data['Class'].astype('int')
y = np.asarray(cancer_data['Class'])
y [0:5]


# ## Train/Test Dataset

# In[97]:


X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[ ]:





# <hr>
# 
# <div id="modeling">
#     <h2>Modeling</h2>
#     We will first create an instance of the <b>DecisionTreeClassifier</b> called <b>drugTree</b>.<br>
#     Inside of the classifier, specify <i> criterion="entropy" </i> so we can see the information gain of each node.
# </div>

# In[98]:


cancerTree = DecisionTreeClassifier()
cancerTree # it shows the default parameters


# Next, we will fit the data with the training feature matrix <b> X_trainset </b> and training  response vector <b> y_trainset </b>

# In[99]:


cancerTree.fit(X_train,y_train)


# <hr>
# 
# <div id="prediction">
#     <h2>Prediction</h2>
#     Let's make some <b>predictions</b> on the testing dataset and store it into a variable called <b>predTree</b>.
# </div>

# In[100]:


predTreeCancer = cancerTree.predict(X_test)


# You can print out <b>predTree</b> and <b>y_testset</b> if you want to visually compare the prediction to the actual values.

# In[101]:


print (predTreeCancer [0:5])
print (y_test [0:5])


# <hr>
# 
# <div id="evaluation">
#     <h2>Evaluation</h2>
#     Next, let's import <b>metrics</b> from sklearn and check the accuracy of our model.
# </div>

# In[104]:


print("DecisionTrees's before pruning Accuracy: ", metrics.accuracy_score(y_test, predTreeCancer))


# In[105]:


print("Decision Tree before pruning F1_Score:", f1_score(y_test, predTreeCancer, average='weighted'))


# In[106]:


print("Decision Tree before pruning Jaccard Score:", jaccard_similarity_score(y_test, predTreeCancer))


# __Accuracy classification score__ computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true.  
# 
# In multilabel classification, the function returns the subset accuracy. If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.
# 

# In[110]:


# Compute confusion matrix
dt_cancer_matrix = confusion_matrix(y_test, predTreeCancer, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, predTreeCancer))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(dt_cancer_matrix, classes=['Benign(2)', 'Malignant(4)'],normalize= False,  title='Decision Tree Before Pruning Confusion Matrix')


# <hr>
# 
# <div id="visualization">
#     <h2>Visualization</h2>
#     Lets visualize the tree
# </div>

# In[117]:


plt.clf()
dot_data = StringIO()
filename = "cancertreebp.png"
featureNames = cancer_data.columns[1:10]
#targetNames = cancer_data["Class"].unique().tolist()
targetNames = [2,4]
out=tree.export_graphviz(cancerTree,feature_names=featureNames, out_file=dot_data, class_names= ['Benign','Malignant'], filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')


# <hr>
# 
# <div id="pruning">
#     <h2>Pruning</h2>
#     Lets prune the decision tree to avoid overfitting
# </div>

# We can avoid overfitting by changing the parameters.
# 
# Pruning Parameters:
# <ul>
#     <li> <b>max_leaf_nodes - reduce the number of leaf nodes</b> </li>
#     <li> <b>min_samples_leaf - restrict the size of sample leaf</b> </li>
#     <li> <b>max_depth - reduce the depth of the tree to build a generalized tree</b> </li>
# </ul>

# In[30]:


#We will rebuild a new tree by using above data and see how it works by tweeking the parameteres
drugTree2 = DecisionTreeClassifier(criterion = "gini", max_leaf_nodes = 4)
drugTree2


# Next, we will fit the data with the training feature matrix <b> X_trainset </b> and training  response vector <b> y_trainset </b> to the new prune decision tree

# In[31]:


drugTree2.fit(X_trainset,y_trainset)


# In[32]:


#Predict Prune Tree
predTree2 = drugTree2.predict(X_testset)


# <hr>
# 
# <div id="evaluation">
#     <h2>Evaluation</h2>
#     Next, let's import <b>metrics</b> from sklearn and check the accuracy of our prune model.
# </div>

# In[33]:


print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree2))


# In[38]:


print("Decision Tree after pruning F1_Score:", f1_score(y_testset, predTree2, average = 'weighted'))


# In[39]:


print("Decision Tree after pruning Jaccard Score:", jaccard_similarity_score(y_testset, predTree2))


# In[43]:


# Compute confusion matrix
dtp_cnf_matrix = confusion_matrix(y_testset, predTree2, labels=['drugA','drugB','drugC','drugX','drugY'])
np.set_printoptions(precision=2)

print (classification_report(y_testset, predTree2))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(dtp_cnf_matrix, classes=['DrugA','DrugB','DrugC', 'DrugX', 'DrugY'],normalize= False,  title='Decision Tree After Pruning Confusion Matrix')


# In[ ]:





# <h3>Visualize the Prune Tree</h3>

# In[34]:


plt.clf
dot_data = StringIO()
filename = "drugtree2.png"
featureNames = my_data.columns[0:5]
targetNames = my_data["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree2,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)  
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')


# 
# 
# <h1><center>Neural Networks</center></h1>
