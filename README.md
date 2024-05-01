# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
```
import the standard libraries.
Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
Import LabelEncoder and encode the dataset.
Import LogisticRegression from sklearn and apply the model on the dataset.
Predict the values of array.
Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
Apply new unknown value
```

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Vasanth P
RegisterNumber: 212222240113
```

```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]
```

## Output:
Original data(first five columns)


![image](https://user-images.githubusercontent.com/94911373/167058479-427a6a41-89fe-44c7-9a72-8f00f8ec5763.png)


Data after dropping unwanted columns(first five):

![image](https://user-images.githubusercontent.com/94911373/167058586-f0f196f1-3063-4804-8387-82c06bae9f14.png)

Checking the presence of null values:

![image](https://user-images.githubusercontent.com/94911373/167058653-b0956bfe-a817-4408-922d-d290a39da450.png)

Checking the presence of duplicated values

![image](https://user-images.githubusercontent.com/94911373/167058762-1bb8b273-d7df-4bb0-a89c-9d2f9a61a2a6.png)


Data after Encoding

![image](https://user-images.githubusercontent.com/94911373/167058820-de576627-b66c-4602-abe9-f1e243bb2436.png)


X Data

![image](https://user-images.githubusercontent.com/94911373/167058938-a6c3369d-f48c-4964-a12d-454a4f9eaba4.png)

Y Data

![image](https://user-images.githubusercontent.com/94911373/167058975-3596007b-850d-4bde-82f0-1beb536a9284.png)

Predicted Values

![image](https://user-images.githubusercontent.com/94911373/167059026-18ddbf18-ac09-4295-81cb-1836a3c054d3.png)

Accuracy Score

![image](https://user-images.githubusercontent.com/94911373/167059074-b2562d39-d22c-4b03-8b4d-361d2d5e41fc.png)


Confusion Matrix

![image](https://user-images.githubusercontent.com/94911373/167059101-94640a14-1869-464f-8070-e1423e7e8fb4.png)

Classification Report

![image](https://user-images.githubusercontent.com/94911373/167059133-eab54da6-b8a4-447d-9add-981e8014e9aa.png)


Predicting output from Regression Model

![image](https://user-images.githubusercontent.com/94911373/167059177-8266cd3b-22a9-404d-984f-c487f7834523.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
