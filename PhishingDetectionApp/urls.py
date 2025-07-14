from django.urls import path

from . import views

urlpatterns = [path("", views.index, name="index"),
               path("AdminLogin.html", views.AdminLogin, name="AdminLogin"),
               path("UserLogin", views.UserLogin, name="UserLogin"),
               path("UserLoginAction", views.UserLoginAction, name="UserLoginAction"),
               path("AdminLoginAction", views.AdminLoginAction, name="AdminLoginAction"),
               path("RunLGBM", views.RunLGBM, name="RunLGBM"),
               path("RunSVM", views.RunSVM, name="RunSVM"),
               path("RunMLP", views.RunMLP, name="RunMLP"),
               path("Predict", views.Predict, name="Predict"),
               path("PredictAction", views.PredictAction, name="PredictAction"),
	       path("Predict.html", views.Predict, name="Predict"),	      
               path("PredictAction", views.PredictAction, name="PredictAction"),
               path("blockurl", views.blockurl, name="blockurl"),
               path("ViewBlockedUrls", views.ViewBlockedUrls,name="ViewBlockedUrls"),
]
