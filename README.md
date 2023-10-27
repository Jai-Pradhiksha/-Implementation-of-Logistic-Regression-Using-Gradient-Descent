# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: JAI PRADHIKSHA D P
RegisterNumber:  212221040062
*/
```
```
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1 (1).txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

print("Array of X") 
X[:5]

print("Array of y") 
y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
print("Exam 1- score Graph")
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

 plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
print("Sigmoid function graph")
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad
    
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print("X_train_grad value")
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print("Y_train_grad value")
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad 
   
 X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(" Print res.x")
print(res.fun)
print(res.x)   
    
 def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()
  
print("Decision boundary - graph for exam score")
plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1, 45, 85]),res.x))
print("Proability value ")
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
   
 print("Prediction value of mean")
np.mean(predict(res.x,X) == y)  

```
## Output:
1. Array of X
![image](https://github.com/Jai-Pradhiksha/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/100289733/df3da620-b64f-459d-9140-b130b2105464)
2. Array of Y
![image](https://github.com/Jai-Pradhiksha/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/100289733/4e71ce41-ef0e-4d77-9c28-e47bf33af5db)
3. Score Graph
![image](https://github.com/Jai-Pradhiksha/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/100289733/cf8b5db0-7127-41c3-aa39-397fceb3b01d)
4. Sigmoidal function graph
![image](https://github.com/Jai-Pradhiksha/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/100289733/14841f14-a279-49ef-860a-6b07e3d0a420)
5. X train grad value
![image](https://github.com/Jai-Pradhiksha/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/100289733/f367e90d-5bd6-42ca-9e10-99b0b2753b38)
6. Y train grad value
![image](https://github.com/Jai-Pradhiksha/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/100289733/f58dc85a-bba0-4926-9120-37a233d02720)
7. Print res.x
![image](https://github.com/Jai-Pradhiksha/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/100289733/11a65a8a-7ce2-496b-9f4a-9635f7da8d66)
8. Decision boundary - graph for exam score
![image](https://github.com/Jai-Pradhiksha/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/100289733/565bbab9-80d2-42ba-8eaa-dcdfebe6985d)
9. Probability value
![image](https://github.com/Jai-Pradhiksha/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/100289733/5263eda1-0adc-4476-a176-027454009933)
10. Prediction value of mean
![image](https://github.com/Jai-Pradhiksha/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/100289733/b272b09a-b3e3-4fb0-883c-1d2b1bf72c8b)



## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

