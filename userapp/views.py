from django.shortcuts import render, redirect
from userapp.models import *
from django.contrib import messages
import urllib.request
import urllib.parse
import random 
import time
from keras.models import load_model
from time import sleep
from tensorflow.keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
# Create your views here.
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score,f1_score, recall_score, precision_score
from imblearn import under_sampling 
from imblearn import over_sampling
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re
from django.shortcuts import render
from nltk.stem import WordNetLemmatizer
import pickle    
import re
from nltk.stem import PorterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
from adminapp.models import * 
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, classification_report

from imblearn import under_sampling 
from imblearn import over_sampling
from imblearn.over_sampling import SMOTE
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from django.core.mail import send_mail
from django.conf import settings

from adminapp.models import Upload_dataset_model, Logistic
from django.contrib import messages
from django.http import HttpResponse


nltk.download('wordnet')


# User Register details
def register(req):
    if req.method == 'POST' :
        name = req.POST.get('myName')
        age = req.POST.get('myAge')
        password = req.POST.get('myPwd')
        phone = req.POST.get('myPhone')
        email = req.POST.get('myEmail')
        address = req.POST.get("address")
        image = req.FILES['image']
        number = random.randint(1000,9999)
        mail_message = f'Registration Successfull\n Your Login OTP is\n {number}'
        # print(mail_message)
        send_mail("Login OTP", mail_message , settings.EMAIL_HOST_USER, [email])

        print(number)
        try:
            user_data = User_details.objects.get(Email = email)
            messages.warning(req, 'Email was already registered, choose another email..!')
            return redirect("register")
        except:
            User_details.objects.create(Full_name = name, Image = image, Age = age, Password = password, Address = address, Email = email, Phone_Number = phone, Otp_Num = number)
            messages.success(req, 'Your account was created..')
            return redirect('register')
 
    return render(req, 'user/user-register.html')

# User Login 
def login(req):
    if req.method == 'POST':
        user_email = req.POST.get('uemail')
        user_password = req.POST.get('upwd')

        try:
            user_data = User_details.objects.get(Email = user_email)
            if user_data.Password == user_password:
                if user_data.User_Status == 'accepted' and user_data.Otp_Status == 'verified':
                    req.session['User_id'] = user_data.User_id
                    user_data.No_Of_Times_Login += 1
                    user_data.save()
                    # print(user_data.No_Of_Times_Login,'no of logins')
                    messages.success(req, 'You are logged in..')
                    return redirect('userdashboard')
                elif user_data.Otp_Status == 'pending':
                    req.session['User_id'] = user_data.User_id
                    messages.warning(req, 'Your OTP Verification was in Penidng. Submit the OTP for verification..!')
                    return redirect('otpverify')
                elif user_data.User_Status == 'pending' and user_data.Otp_Status == 'verified':
                    req.session['User_id'] = user_data.User_id
                    messages.warning(req, 'Your Request was in pending. Please Try after some time. Thank You..!')
                    return redirect('login')
            else:
                messages.warning(req, 'Password was incorrect..!')
                return redirect('login')
        except:
            # if user_data.Email == user_email and user_data.Password != user_password:
            #     messages.info(req, 'Password was incorrect..!')
            #     return redirect('login')
            messages.warning(req, 'Once Check your passowrd and mail id, or you did not have an account please register..!')
            return redirect('login')
    return render(req, 'main/main-user.html')

# OTP Verification 
def otpverify(req):
    user_id = req.session['User_id']
    user_o = User_details.objects.get(User_id = user_id)
    print(user_o.Otp_Num)
    if req.method == 'POST':
        user_otp = req.POST.get('otp')
        u_otp = int(user_otp)
        if u_otp == user_o.Otp_Num:
            user_o.Otp_Status = 'verified'
            user_o.save()
            messages.success(req, 'OTP verification was Success. Now you can continue to login..!')
            return redirect('home')
        else:
            messages.error(req, 'OTP verification was Faild. You entered invalid OTP..!')
            return redirect('otpverify')
    return render(req, 'user/user-otpverify.html')

# user-dashboard Function
def userdashboard(req):
    prediction_count =  User_details.objects.all().count()
    user_id = req.session["User_id"]
    user = User_details.objects.get(User_id = user_id)
    return render(req, 'user/user-dashboard.html', {'predictions' : prediction_count, 'la' : user})


# user-profile Function
def profile(req):
    user_id = req.session["User_id"]
    user = User_details.objects.get(User_id = user_id)
    if req.method == 'POST':
        user_name = req.POST.get('userName')
        user_age = req.POST.get('userAge')
        user_phone = req.POST.get('userPhNum')
        user_email = req.POST.get('userEmail')
        user_address = req.POST.get("userAddress")
        # user_img = request.POST.get("userimg")

        user.Full_name = user_name
        user.Age = user_age
        user.Address = user_address
        user.Phone_Number = user_phone

        if len(req.FILES) != 0:
            image = req.FILES['profilepic']
            user.Image = image
            user.Full_name = user_name
            user.Age = user_age
            user.save()
        else:
            user.Full_name = user_name
            user.Age = user_age
            user.save()
        # print(user_name, user_age, user_phone, user_email, user_address)
    context = {"i":user}
    return render(req, 'user/user-profile.html', context)

def depfacial(request):
    return render(request, 'user/user-depression-facial.html')

def depchatbot(request):
    return render(request, 'user/user-depression-chatbot.html')

# predict button count
# def predict_count (req, id):
#     predict_counts = User_details.objects.get(User_id = id)
#     predict_counts.No_Of_Predictions += 1
#     return render(req, 'user/user-predict.html')

# Result function
def result(req):
    return render(req, 'user/user-result.html')

# User Logout
def userlogout(req):
    user_id = req.session["User_id"]
    user = User_details.objects.get(User_id = user_id) 
    t = time.localtime()
    user.Last_Login_Time = t
    current_time = time.strftime('%H:%M:%S', t)
    user.Last_Login_Time = current_time
    current_date = time.strftime('%Y-%m-%d')
    user.Last_Login_Date = current_date
    user.save()
    messages.info(req, 'You are logged out..')
    # print(user.Last_Login_Time)
    # print(user.Last_Login_Date)
    return redirect('login')





# from flask import Flask, render_template,request

wo = WordNetLemmatizer()

def preprocess(data):
    # Preprocess
    a = re.sub('[^a-zA-Z]', ' ', data)
    a = a.lower()
    a = a.split()
    a = [wo.lemmatize(word) for word in a]
    a = ' '.join(a)
    return a

tfidf_vectorizer = pickle.load(open('userapp/vectorizer.pkl', 'rb'))
model = pickle.load(open('userapp/prediction.pkl', 'rb'))

def predict(request):
    if request.method == 'POST':
        msg = request.POST.get('mood_pred', '')
        a = preprocess(msg)

        pred = model.predict(tfidf_vectorizer.transform([a]))[0]
        result = pred

        if result == 0:
            return render(request, 'user/user-depression-chatbot.html', {'final_result': "Not Depressed", 'quote': "Be Happy, Be Bright, Be You"})

        elif result == 1:
            return render(request, 'user/user-depression-chatbot.html', {'final_result': "Depressed", 'quote': "Consult Your Doctor"})

    # Handle GET requests or other cases
    return render(request, 'user/user-depression-chatbot.html')





def depressionfacial(request):
    face_classifier = cv2.CascadeClassifier('userapp\haarcascade_frontalface_default.xml')
    classifier =load_model('userapp\model.h5')

    emotion_labels = ['Depressed(angry)', 'Disgust', 'Depressed(fear)', 'Not Depressed(happy)', 'Neutral', 'Depressed(sad)', 'Not Depressed(surprise)']

    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y - 10)

                if label == 'Depressed(angry)' or label=='Depressed(fear)' or label== 'Depressed(sad)':
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Change color to red
                else:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255,0), 2)  # Default color

                cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Emotion Detector', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return render(request, 'user/user-depression-facial.html')

def naviesbyesbtn(req):
    data = Upload_dataset_model.objects.last()
    file_path = f'./media/{data.Dataset}'

    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')

    if df.shape[1] < 2:
        return HttpResponse("❌ Dataset must have at least 2 columns!", status=400)

    print("Available columns:", df.columns)

    # Pick first text-like column for input
    text_column = None
    for col in df.columns:
        if df[col].dtype == 'object':
            text_column = col
            break

    if not text_column:
        text_column = df.columns[0]

    # Pick second column for label
    label_column = df.columns[1] if df.shape[1] > 1 else df.columns[0]

    wo = WordNetLemmatizer()
    corpus = []

    for text in df[text_column]:
        message = re.sub('[^a-zA-Z]', ' ', str(text))
        message = message.lower().split()
        message = [wo.lemmatize(word) for word in message]
        corpus.append(' '.join(message))

    X_train, X_test, y_train, y_test = train_test_split(corpus, df[label_column], test_size=0.25, random_state=42)

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_features=15000)
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    x_resample, y_resample = SMOTE().fit_resample(X_train_vect, y_train)

    mnb = MultinomialNB()
    mnb.fit(x_resample, y_resample)

    y_pred = mnb.predict(X_test_vect)

    accuracy = round(accuracy_score(y_test, y_pred) * 100, 3)
    precision = round(precision_score(y_test, y_pred, average='macro') * 100, 3)
    f1_Score = round(f1_score(y_test, y_pred, average='macro') * 100, 3)
    recall = round(recall_score(y_test, y_pred, average='macro') * 100, 3)

    Naive_bayes.objects.create(Accuracy=accuracy, Precession=precision, F1_Score=f1_Score, Recall=recall, Name='Naive Bayes')

    data = Naive_bayes.objects.last()
    messages.success(req, '✅ Naive Bayes executed successfully.')
    return render(req, 'admin/admin-xgboost-algorithm.html', {'i': data})

def logisticbtn(req):
    data = Upload_dataset_model.objects.last()
    file_path = f'./media/{data.Dataset}'

    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')

    if df.shape[1] < 2:
        return HttpResponse("❌ Dataset must have at least 2 columns!", status=400)

    print("Available columns:", df.columns)

    # Pick first text-like column for input
    text_column = None
    for col in df.columns:
        if df[col].dtype == 'object':
            text_column = col
            break

    if not text_column:
        # If no text column found, fallback to first column forcibly
        text_column = df.columns[0]

    # Pick second column for label
    label_column = df.columns[1] if df.shape[1] > 1 else df.columns[0]

    wo = WordNetLemmatizer()
    corpus = []

    for text in df[text_column]:
        message = re.sub('[^a-zA-Z]', ' ', str(text))
        message = message.lower().split()
        message = [wo.lemmatize(word) for word in message]
        corpus.append(' '.join(message))

    X_train, X_test, y_train, y_test = train_test_split(corpus, df[label_column], test_size=0.25, random_state=42)

    vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='english', max_features=15000)
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    x_resample, y_resample = SMOTE().fit_resample(X_train_vect, y_train)

    clf = LogisticRegression(solver='lbfgs', max_iter=500)
    clf.fit(x_resample, y_resample)

    y_pred = clf.predict(X_test_vect)

    acc = round(accuracy_score(y_test, y_pred) * 100, 3)
    prec = round(precision_score(y_test, y_pred, average='macro') * 100, 3)
    f1 = round(f1_score(y_test, y_pred, average='macro') * 100, 3)
    rec = round(recall_score(y_test, y_pred, average='macro') * 100, 3)

    Logistic.objects.create(Accuracy=acc, Precession=prec, F1_Score=f1, Recall=rec, Name='Logistic Regression')
    result = Logistic.objects.last()

    messages.success(req, '✅ Logistic Regression executed successfully.')
    return render(req, 'admin/admin-anm-algorithm.html', {'i': result})

def Feedback(request):
    if request.method == 'POST':
        review = request.POST.get('review')
        rating = request.POST.get('rating')
        # print(review, rating)
        feed_id = request.session["User_id"]
        user_id = User_details.objects.get(User_id = feed_id)
        sid = SentimentIntensityAnalyzer()
        score = sid.polarity_scores(review)
        sentiment = None
        if score['compound'] >= 0.5:
            sentiment = 'Very Positive'
        elif score['compound'] >= 0:
            sentiment = 'Positive'
        elif score['compound'] >- 0.5:
            sentiment = 'Neutral' 
        elif score['compound'] >- 1:
            sentiment = 'Negative'
        else:
            sentiment = 'Very negative'
        FeedbackModel.objects.create(Rating = rating, Review = review, Sentiment = sentiment, Reviewer = user_id)
        # print(sentiment, rating)
        messages.success(request, 'Feedback was Submitted')
        return redirect("feedback")
    return render(request, 'user/user-feedback.html')

def forgotpwd(req):
    return render(req, 'user/user-forgot-passwrod.html')