from models import svc, neigh, tree, clf, model
from google.colab.patches import cv2_imshow
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def pred_svc(img_array, plot):
  img_pil = Image.fromarray(img_array)
  img_28x28 = np.array(img_pil.resize((28, 28), Image.ANTIALIAS))
  img_array = (img_28x28.flatten())
  img_array = img_array.reshape(1, -1)
  p = svc.predict(img_array)[0]
  if plot:
    show(img_array)
  return p


def pred_knn(img_array, plot):
  img_pil = Image.fromarray(img_array)
  img_28x28 = np.array(img_pil.resize((28, 28), Image.ANTIALIAS))
  img_array = (img_28x28.flatten())
  img_array = img_array.reshape(1, -1)
  p = neigh.predict(img_array)[0]
  if plot:
    show(img_array)
  return p


def pred_dt(img_array, plot):
    img_pil = Image.fromarray(img_array)
    img_28x28 = np.array(img_pil.resize((28, 28), Image.ANTIALIAS))
    img_array = (img_28x28.flatten())
    img_array = img_array.reshape(1, -1)
    p = tree.predict(img_array)[0]
    if plot:
        show(img_array)
    return p


def pred_lr(img_array, plot):
    img_pil = Image.fromarray(img_array)
    img_28x28 = np.array(img_pil.resize((28, 28), Image.ANTIALIAS))
    img_array = (img_28x28.flatten())
    img_array = img_array.reshape(1, -1)
    p = clf.predict(img_array)[0]
    if plot:
        show(img_array)
    return p


def pred_nb(img_array, plot):
    img_pil = Image.fromarray(img_array)
    img_28x28 = np.array(img_pil.resize((28, 28), Image.ANTIALIAS))
    img_array = (img_28x28.flatten())
    img_array = img_array.reshape(1, -1)
    p = model.predict(img_array)[0]
    if plot:
        show(img_array)
    return p


def recognise(img,plot):
    img = cv2.imread(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    # threshold
    _, thresh = cv2.threshold(gray, 100, 200, cv2.THRESH_BINARY)

    # get contours
    result = img.copy()
    contours = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        if h < 10 or w < 5:
          continue
        w = max(w, 28)
        h = max(h, 28)
        if w < h:
            x -= (h-w)//2
            w = h
        else:
            y -= (w-h)//2
            h = w
        new = gray.copy()[y:y+h, x:w+x]
        c = pred_svc(new, plot)
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(result, str(c), (x+w//2-2, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        print("x,y,w,h:", x, y, w, h, c)
    cv2_imshow(result)
