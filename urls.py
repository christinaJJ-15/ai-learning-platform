from django.urls import path

from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('survey/', views.survey_view, name='survey'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
    path('feedback/', views.feedback_view, name='feedback'),
    path('progress/', views.progress_view, name='progress'),
    path('admin-analytics/', views.admin_analytics_view, name='admin_analytics'),
    path('export-report/', views.export_pdf_report, name='export_report'),
    path('api/search-materials/', views.search_materials_api, name='api_search_materials'),
    path('api/recommendations/', views.recommendations_api, name='api_recommendations'),
    path('api/analytics/', views.analytics_api, name='api_analytics'),
]
