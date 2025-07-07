from django.db import models

# Create your models here.


class User_details(models.Model):
    User_id = models.AutoField(primary_key = True)
    Full_name = models.CharField(max_length = 100)
    Image = models.FileField(upload_to='images/')
    Email = models.EmailField(max_length = 50)
    Address = models.CharField(max_length = 200, null = True)
    # Last_name = models.CharField(max_length=150,null=True)
    Age = models.IntegerField(null = True)
    Phone_Number = models.CharField(max_length=10, null = True)
    Password = models.CharField(max_length = 15, null = True)
    Date_Time = models.DateTimeField(auto_now = True, null = True)
    User_Status = models.CharField(default = 'pending', max_length=50, null = True)
    Otp_Num = models.IntegerField(null = True)
    Otp_Status = models.CharField(default = 'pending', max_length = 60, null = True)
    Last_Login_Time = models.TimeField(null = True)
    Last_Login_Date = models.DateField(null = True)
    No_Of_Times_Login = models.IntegerField(default = 0, null = True)

    class Meta:
        db_table = 'user_detail'

class Predict_details(models.Model):
    predict_id = models.AutoField(primary_key=True)
    Field_1 = models.CharField(max_length = 60, null = True)
    Field_2 = models.CharField(max_length = 60, null = True)
    Field_3 = models.CharField(max_length = 60, null = True)
    Field_4 = models.CharField(max_length = 60, null = True)
    Field_5 = models.CharField(max_length = 60, null = True)
    Field_6 = models.CharField(max_length = 60, null = True)
    Field_7 = models.CharField(max_length = 60, null = True)
    Field_8 = models.CharField(max_length = 60, null = True)
    Field_9 = models.CharField(max_length = 60, null = True)
    Field_10 = models.CharField(max_length = 60, null = True)
    
    class Meta:
        db_table = "predict_detail"

class Last_login(models.Model):
    Id = models.AutoField(primary_key = True)
    Login_Time = models.DateTimeField(auto_now = True, null = True)

    class Meta:
        db_table = "last_login"

class FeedbackModel(models.Model):
    Feedback_Id = models.AutoField(primary_key=True)
    Review = models.CharField(max_length = 1000)
    Rating = models.CharField(max_length = 1000)
    Sentiment = models.CharField(max_length = 1000)
    Reviewer = models.ForeignKey(User_details, on_delete = models.CASCADE)    
    Feedback_Time = models.DateTimeField(auto_now = True)
    
    class Meta:
        db_table = 'user_feedback'