from django.db import models

# Create your models here.

class All_users_model(models.Model):
    User_id = models.AutoField(primary_key = True)
    user_Profile = models.FileField(upload_to = 'images/')
    User_Email = models.EmailField(max_length = 50)
    User_Status = models.CharField(max_length = 10)
    
    class Meta:
        db_table = 'all_users'

# Dataset
class Upload_dataset_model(models.Model):
    User_id = models.AutoField(primary_key = True)
    Dataset = models.FileField(upload_to='dataset/', null=True)
    File_size = models.CharField(max_length = 100) 
    Date_Time = models.DateTimeField(auto_now = True)
    
    class Meta:
        db_table = 'upload_dataset'

class Naive_bayes(models.Model):
    NB_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length=100, null=True)
    class Meta:
        db_table = 'naive_bayes'

class Logistic(models.Model):
    LR_ID = models.AutoField(primary_key = True)
    Accuracy = models.TextField(max_length = 100)
    Precession = models.TextField(max_length = 100) 
    F1_Score = models.TextField(max_length = 100)
    Recall = models.TextField(max_length = 100)
    Name = models.TextField(max_length=100, null=True)

    class Meta:
        db_table = 'logistic'
        