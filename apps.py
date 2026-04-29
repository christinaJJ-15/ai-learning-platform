from django.apps import AppConfig


class StudentAnalysisConfig(AppConfig):
    default_auto_field = 'django_mongodb_backend.fields.ObjectIdAutoField'
    name = 'student_analysis'
