from django.urls import path

from . import views

urlpatterns = [path("index.html", views.index, name="index"),
			path("Login.html", views.Login, name="Login"),
			path("LoginAction", views.LoginAction, name="LoginAction"),
			path("Signup.html", views.Signup, name="Signup"),
			path("SignupAction", views.SignupAction, name="SignupAction"),	    
			path("PublishNews.html", views.PublishNews, name="PublishNews"),
			path("PublishNewsAction", views.PublishNewsAction, name="PublishNewsAction"),	  
			path("ViewNews", views.ViewNews, name="ViewNews"),
			path("loadDataset", views.loadDataset, name="loadDataset"),
			path("trainReinforceModel", views.trainReinforceModel, name="trainReinforceModel"),
]