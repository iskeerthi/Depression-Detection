import pandas as pd
from adminapp.models import *
from django.shortcuts import render, redirect
from adminapp.models import  *
from userapp.models import *
from django.contrib import messages
from django.core.paginator import Paginator
from .models import Upload_dataset_model
# import os
# import socket
# Admin logout
def adminlogout(req):
    messages.info(req, 'You are logged out..!')
    return redirect('admin')

#Admin Dashboard index.html
def admindashboard(req):
    all_users_count =  User_details.objects.all().count()
    pending_users_count = User_details.objects.filter(User_Status = 'Pending').count()
    rejected_users_count = User_details.objects.filter(User_Status = 'removed').count()
    accepted_users_count = User_details.objects.filter(User_Status = 'accepted').count()
    datasets_count = Upload_dataset_model.objects.all().count()
    no_of_predicts = Predict_details.objects.all().count()
    return render(req, 'admin/admin-dashboard.html',{'a' : pending_users_count, 'b' : all_users_count, 'c' : rejected_users_count, 'd' : accepted_users_count, 'e' : datasets_count, 'f' : no_of_predicts})

# Admin pending users
def pendingusers(req):
    pending = User_details.objects.filter(User_Status = 'pending')
    paginator = Paginator(pending, 5) 
    page_number = req.GET.get('page')
    post = paginator.get_page(page_number)
    return render(req, 'admin/admin-pending-users.html', { 'user' : post})

# Admin all users
def allusers(req):
    all_users = User_details.objects.all()
    paginator = Paginator(all_users, 5)
    page_number = req.GET.get('page')
    post = paginator.get_page(page_number)
    return render(req, 'admin/admin-all-users.html', {"allu" : all_users, 'user' : post})

#Deleet user button in allusers
def delete_user(req, id):
    User_details.objects.get(User_id = id).delete()
    messages.warning(req, 'User was Deleted..!')
    return redirect('allusers')

# Acept users button
def accept_user(req, id):
    status_update = User_details.objects.get(User_id = id)
    status_update.User_Status = 'accepted'
    status_update.save()
    messages.success(req, 'User was accepted..!')
    return redirect('pendingusers')

# Remove user button
def reject_user(req, id):
    status_update2 = User_details.objects.get(User_id = id)
    status_update2.User_Status = 'removed'
    status_update2.save()
    messages.warning(req, 'User was Rejected..!')
    return redirect('pendingusers')

# change status
# def change_status(req, id):
#     status_update = User_details.objects.get(User_id = id)
#     if (status_update.User_Status == 'accepted'):
#         status_update.User_Status = 'rejected'
#     else:
#         status_update.User_Status = 'accepted'

#     status_update.save()
#     return redirect('allusers')

# Admin upload dataset
def uploaddataset(req):
    if req.method == 'POST':
        file = req.FILES['data_file']
        # print(file)
        file_size = str((file.size)/1024) +' kb'
        # print(file_size)
        Upload_dataset_model.objects.create(File_size = file_size, Dataset = file)
        messages.success(req, 'Your dataset was uploaded..')
    return render(req, 'admin/admin-upload-dataset.html')

# Admin view dataset
def viewdataset(request):
    data = Upload_dataset_model.objects.last()
    if not data:
        return render(request, 'admin/admin-view-dataset.html', {'t': 'No dataset found.'})

    file_path = f'./media/{data.Dataset}'
    
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='ISO-8859-1')  # Fallback encoding

    table = df.to_html(table_id='data_table')
    return render(request, 'admin/admin-view-dataset.html', {'t': table})

# Admin delete dataset button
def delete_dataset(req, id):
    dataset = Upload_dataset_model.objects.get(User_id = id).delete()
    messages.warning(req, 'Dataset was deleted..!')
    return redirect('viewdataset')

# Admin ANM Alogorithm
def anmalgm(req):
    return render(req, 'admin/admin-anm-algorithm.html')

# Admin XGBOOST Algorithm
def xgbalgm(req):
    return render(req, 'admin/admin-xgboost-algorithm.html')

# Admin ADA Boost Algorithm
def adabalgm(req):
    return render(req, 'admin/admin-adaboost-algorithm.html')

# Admin KNN Algorithm
def knnalgm(req):
    return render(req, 'admin/admin-knn-algorithm.html')

# Admin SXM Algorithm
def sxmalgm(req):
    return render(req, 'admin/admin-sxm-algorithm.html')

# Admin Decission tree Algorithm
def dtalgm(req):
    return render(req, 'admin/admin-decission-algorithm.html')

# Admin Comparison graph
def cgraph(req):
    details = Logistic.objects.last()
    a = details.Accuracy
    deatails1 = Naive_bayes.objects.last()
    b = deatails1.Accuracy
    return render(req, 'admin/admin-graph-analysis.html', {'log':a, 'nav':b})

def usersfeedback(request):
    feed = FeedbackModel.objects.all()
    return render(request, 'admin/admin-user-feedback.html', {'feed':feed})