# fetching dataset
from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.model_selection import cross_val_score
import warnings
from sklearn.exceptions import ConvergenceWarning
from PIL import Image
warnings.filterwarnings(action='ignore', category=ConvergenceWarning)

def show(img):
  some_digit_image=img.reshape(28,28)
  plt.imshow(some_digit_image, cmap=matplotlib.cm.binary,interpolation='nearest')
  plt.axis("off")
  plt.show()

print("Fetching dataset")
mnist = fetch_openml('mnist_784')
x, y = mnist['data'], mnist['target']

some_digit = x.to_numpy()[36001]
show(some_digit)

x_train, x_test = x[:60000], x[60000:70000]
y_train, y_test = y[:60000], y[60000:70000]

shuffle_index = np.random.permutation(6000)
x_train, y_train = x_train.iloc[shuffle_index], y_train[shuffle_index]

y_train = y_train.astype(np.int8)
y_test = y_test.astype(np.int8)

print("Training Logistic Regression")
clf = LogisticRegression(tol=0.1)
t=time.time()
clf.fit(x_train.values, y_train)
a = cross_val_score(clf, x_train.values, y_train, cv=3, scoring="accuracy")
print(f"Training time: {time.time()-t}")
print(f"Mean of CV of Linear Regression: {a.mean()}")
print(f"Accuracy of Linear regression  : {clf.score(x_test.values,y_test)}")

print("Training Naive Bayes")
model = GaussianNB()
t=time.time()
model.fit(x_train.values,y_train)
a = cross_val_score(model, x_train.values, y_train, cv=3, scoring="accuracy")
print(f"Training time: {time.time()-t}")
print(f"Mean of CV of Naive Bayes      : {a.mean()}")
print(f"Accuracy of Naive Bayes        : {model.score(x_test.values,y_test)}")

print("Training KNN")
neigh = KNeighborsClassifier(n_neighbors=3)
t=time.time()
neigh.fit(x_train.values,y_train)
a = cross_val_score(neigh, x_train.values, y_train, cv=3, scoring="accuracy")
print(f"Training time: {time.time()-t}")
print(f"Mean of CV of Kneighbors       : {a.mean()}")
print(f"Accuracy of KNeighbors         : {neigh.score(x_test.values,y_test)}")

print("Training Decision Tree")
tree = DecisionTreeClassifier(random_state=0)
t=time.time()
tree.fit(x_train.values, y_train)
a = cross_val_score(tree, x_train.values, y_train, cv=3, scoring="accuracy")
print(f"Training time: {time.time()-t}")
print(f"Mean of CV of Decision Tree    : {a.mean()}")
print(f"Accuracy of Decision Tree      : {tree.score(x_test.values, y_test)}")

print("Training SVM")
svc = svm.SVC(kernel='rbf')
t=time.time()
svc.fit(x_train.values, y_train)
a = cross_val_score(svc, x_train.values, y_train, cv=3, scoring="accuracy")
print(f"Training time: {time.time()-t}")
print(f"Mean of CV of SVM              : {a.mean()}")
print(f"Accuracy of SVM                : {svc.score(x_test.values, y_test)}")


