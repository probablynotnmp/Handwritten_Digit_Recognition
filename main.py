from functions import *
import cv2

Print("Testing")
img_array = cv2.imread('/content/6.jfif', cv2.IMREAD_GRAYSCALE)
img_array = cv2.bitwise_not(img_array)
print(f"Prediction by SVM {pred_svc(img_array,True)}")
print(f"Prediction by KNN {pred_knn(img_array,True)}")
print(f"Prediction by Decision Tree {pred_dt(img_array,True)}")
print(f"Prediction by Logistic Regression {pred_lr(img_array,True)}")
print(f"Prediction by Naive Bayes {pred_nb(img_array,True)}")


recognise('/content/test.png',plot)