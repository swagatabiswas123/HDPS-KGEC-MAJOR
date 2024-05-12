from django.shortcuts import render,redirect
from sklearn import linear_model
from .forms import Parameters
from .regressor import LogitRegression
import pandas as pd
import numpy as np
from . import regressor
from . import svc
from . import randomforest
from . import bayes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression 
from django.contrib.auth.models import User,auth
from django.core.mail import send_mail
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



def quickpredict(request):
    
    if request.method=='POST':

        form=Parameters(request.POST)
        if form.is_valid():
            age=form.cleaned_data['age']
            sex=form.cleaned_data['sex']
            cp=form.cleaned_data['cp']
            trestbps=form.cleaned_data['trestbps']
            chol=form.cleaned_data['chol']
            fbs=form.cleaned_data['fbs']
            restcg=form.cleaned_data['restcg']
            thalach=form.cleaned_data['thalach']
            exang=form.cleaned_data['exang']
            oldpeak=form.cleaned_data['oldpeak']
            slope=form.cleaned_data['slope']
            ca=form.cleaned_data['ca']
            thal=form.cleaned_data['thal']
            
            
            #support vector machines
            
            X, Y = svc.find()
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
            model = SVC (kernel='linear',max_iter=1000, C=10, probability=True) 
            model.fit(X_train, Y_train)
            outputsvc = model.predict(np.array([age, sex, cp, trestbps, chol, fbs, restcg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1))
            print('-----------------------------------------')
            print(outputsvc)
            outputsvc = outputsvc[0]
        
            #Logistic Regression
            
            X,Y=regressor.find()
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0 )
            model = LogisticRegression(max_iter=1000, random_state=1,solver='liblinear', penalty='l1')
            model.fit(X_train, Y_train)
            output1 = model.predict(np.array([age,sex,cp,trestbps,chol,fbs,restcg,thalach,exang,oldpeak,slope,ca,thal]).reshape(1,-1))
            print('-----------------------------------------')
            print(output1)
            output1=output1[0]
            
            #RandomForestClassifier
            
            X, Y = randomforest.find()
            model = RandomForestClassifier(n_estimators=1000, random_state=1, max_leaf_nodes=20, min_samples_split=15)
            model.fit(X_train, Y_train)
            outputran = model.predict(np.array([age, sex, cp, trestbps, chol, fbs, restcg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1,-1))
            print('-----------------------------------------')
            print(outputran)
            outputran = outputran[0]
            
           
           #Gaussian Naive Bayes
           
            X, Y = bayes.find()
            model = GaussianNB(var_smoothing=0.1)
            model.fit(X_train, Y_train)
            outputnav = model.predict(np.array([age, sex, cp, trestbps, chol, fbs, restcg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1,-1))
            print('-----------------------------------------')
            print(outputnav)
            outputnav = outputnav[0]
            
            #K-Nearest Neighbors (KNN) classifier
            X, Y = bayes.find()
            model = KNeighborsClassifier(n_neighbors=3)  
            model.fit(X_train, Y_train)
            outputknn = model.predict(np.array([age, sex, cp, trestbps, chol, fbs, restcg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1,-1))
            print('-----------------------------------------')
            print(outputknn)
            outputknn = outputknn[0]
            
            

            #Disicion tree
            
            X, Y = bayes.find()
            model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, criterion='entropy', min_samples_split=5,splitter='random', random_state=1)  
            model.fit(X_train, Y_train)
            outputdt = model.predict(np.array([age, sex, cp, trestbps, chol, fbs, restcg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1,-1))
            print('-----------------------------------------')
            print(outputdt)
            outputdt = outputdt[0]

            return render(request, 'output.html', {'output1': output1 * 100, 'outputsvc': outputsvc * 100, 'outputran': outputran * 100, 'outputnav': outputnav * 100, 'outputknn': outputknn * 100, 'outputdt': outputdt * 100})
            
            

        


        else:
            print('The form was not valid.')
            return redirect('/')
        
        
    else:
        form=Parameters()
        return render(request,'quickpredict.html',{'form':form})

def index(request):
    if request.user.is_authenticated:
        if request.method=='POST':
    
            form=Parameters(request.POST)
            if form.is_valid():
                age=form.cleaned_data['age']
                sex=form.cleaned_data['sex']
                cp=form.cleaned_data['cp']
                trestbps=form.cleaned_data['trestbps']
                chol=form.cleaned_data['chol']
                fbs=form.cleaned_data['fbs']
                restcg=form.cleaned_data['restcg']
                thalach=form.cleaned_data['thalach']
                exang=form.cleaned_data['exang']
                oldpeak=form.cleaned_data['oldpeak']
                slope=form.cleaned_data['slope']
                ca=form.cleaned_data['ca']
                thal=form.cleaned_data['thal']


                X,Y=regressor.find() 
                X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0 )
                model = LogitRegression(learning_rate=0.0001 , iterations=1000)
                model.fit(X_train, Y_train)
                output , output1 = model.predict(np.array([age,sex,cp,trestbps,chol,fbs,restcg,thalach,exang,oldpeak,slope,ca,thal]).reshape(1,-1))
                danger = 'high' if output == 1 else 'low'
                output1=output1[0]
                saved_data = HeartData(age=age ,
                sex = sex,
                cp = cp,
                trestbps = trestbps,
                chol = chol,
                fbs = fbs,
                restcg = restcg , 
                thalach = thalach , 
                exang = exang,
                oldpeak = oldpeak,
                slope = slope,
                ca = ca,
                thal = thal,
                owner = request.user,
                probability = output1
                )
                saved_data.save()
                return render(request , 'output2.html',{'output1':output1 , 'danger':danger})


        form = Parameters()
        return render(request , 'user.html', {'form':form})
    return render(request , 'index.html')



def record(request): 
    if request.user.is_authenticated:
        record_data = HeartData.objects.filter(owner = request.user) 
        return render(request , 'record.html' , {'record_data':record_data})
    return redirect('/')

def heartdetail(request): 
    if request.user.is_authenticated:
        record_data = HeartData.objects.filter(owner = request.user)
        return render(request , 'heartdetail.html' , {'record_data':record_data})
    return redirect('/')

def symptoms(request): 
    if request.user.is_authenticated:
        record_data = HeartData.objects.all()
        return render(request , 'symptoms.html')
    return redirect('/')

def prevention(request): 

        return render(request , 'prevention.html')
    

def doctorhospital(request): 
    if request.user.is_authenticated:
        datas = DoctorHospital.objects.all()
        return render(request , 'doctorshospitals.html',{'datas':datas})
    return render('/')

def contact(request): 
    if request.method == 'POST':
        name = request.POST.get('name')
        email = request.POST.get('email')
        title = request.POST.get('title1')
        message = request.POST.get('message')
        
        send_mail(title , message+'\n'+'From : '+name+'\n'+'Email : '+email ,from_email=email, recipient_list=['focusus1@gmail.com']) #Sends mail to HDPS
    return render(request , 'contact.html')



def about(request):
    return render(request , 'about.html')







def signin(request): # For the user to sign in.
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']

        user = auth.authenticate(username=username,password=password)
        if user is not None:
            auth.login(request,user)
            return redirect('/')
        else:
            messages.warning(request,'Invalid Credentials')
            return redirect('signin')

        
    else:
        return render(request,'signin.html')


def signup(request): #For the user to resister or sign up.

    if request.method == 'POST':
        first_name = request.POST['first_name']
        last_name = request.POST['last_name']
        username = request.POST['username']
        password = request.POST['password']
        email = request.POST['email']

        
        if User.objects.filter(username=username).exists():
            messages.info(request,'Username taken')
            return redirect('signup')
        elif User.objects.filter(email=email).exists():
            messages.info(request,'Email taken')
            return redirect('signup')
        else:
            user = User.objects.create_user(username=username, password=password,email=email,first_name = first_name,last_name=last_name)
            
            user.save()
            
            messages.success(request,f"User {username} created!")
            return redirect('signin')
        #return redirect('/')
    else:   
        return render(request,'signup.html')


def signout(request): # In order to logout from the website
    auth.logout(request)
    return redirect('/')


# End login