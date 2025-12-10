from django.urls import path
from . import views

urlpatterns = [
    path("", views.login_page, name="login"),
    path("login-email/", views.login_email, name="login_email"),
    path("login-biometric/", views.login_biometric_page, name="login_biometric"),
    path("register/", views.register_page, name="register"),
   path("logout/", views.logout_user, name="logout"),
    # APIs biométricas (ESTO ES LO QUE FALTABA)
    path("api/register-biometric/", views.register_biometric, name="register_biometric"),
    path("api/verify-biometric/", views.verify_biometric, name="verify_biometric"),
 

    path("welcome/", views.welcome_page, name="welcome"),
]
