from django.shortcuts import render,redirect
from django.contrib import messages


# Home
def home(req):
    return render(req, 'main/main-home.html')

# Admin Function
def admin(req):
    admin_name = 'admin'
    admin_pwd = 'admin'
    if req.method == 'POST':
        admin_n = req.POST.get('adminName')
        admin_p = req.POST.get('adminPwd')
        if (admin_n == admin_name and admin_p == admin_pwd):
            messages.success(req, 'You are logged in..')
            return redirect('admindashboard')
        else:
            messages.error(req, 'You are trying to loging with wrong details..')
            return redirect('admin')
    
    return render(req, 'main/main-admin.html')

# about Function
def about(req):
    return render(req, 'main/main-about.html')

# Contact Function
def contact(req):
    if req.method == 'POST':
        name = req.POST.get('name')
        email = req.POST.get('email')
        subject = req.POST.get('subject')
        message = req.POST.get('message')

        # You can save the data to a database here or just print for now
        print(f"New Contact Form: {name}, {email}, {subject}, {message}")

        messages.success(req, 'Your message has been sent successfully!')
        return redirect('contact')

    return render(req, 'main/main-contact.html')

 