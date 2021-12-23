from django.http import HttpResponse
from django.shortcuts import render
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#from pandas_profiling import ProfileReport


def result(request):
    data=pd.read_csv(r'C:\Users\hp\Downloads\Iris.csv')
    data=data.drop('Id',axis=1)

    X = data.iloc[:,:-1].values
    Y = data.iloc[:,-1].values


    

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

    model = LogisticRegression()
    model.fit(X_train,Y_train)
    
    val1=float(request.GET['n1'])
    val2=float(request.GET['n2'])
    val3=float(request.GET['n3'])
    val4=float(request.GET['n4'])
    

    pred=model.predict([[val1,val2,val3,val4]])
    result1=pred[0]

    return render(request,'predict.html', {'result2':result1})



def predict(request):
    return render(request,'predict.html')


def about(request):
    
    

    return render(request,'test.html', context={'job':5})

    