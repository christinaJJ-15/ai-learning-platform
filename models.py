from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MaxValueValidator, MinValueValidator


class StudentProfile(models.Model):
    LOCATION_CHOICES = [('Rural', 'Rural'), ('Urban', 'Urban')]

    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='student_profile')
    full_name = models.CharField(max_length=120)
    location_type = models.CharField(max_length=10, choices=LOCATION_CHOICES)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'{self.full_name} ({self.location_type})'


class SurveyResponse(models.Model):
    PERFORMANCE_CHOICES = [
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='survey_responses')
    student_name = models.CharField(max_length=120, default='Student')
    age = models.PositiveSmallIntegerField(validators=[MinValueValidator(5), MaxValueValidator(30)], default=15)
    location_type = models.CharField(max_length=10, choices=StudentProfile.LOCATION_CHOICES)
    class_grade = models.CharField(max_length=20)
    subjects_studied = models.TextField(help_text='Comma-separated subjects')
    daily_study_hours = models.TextField(help_text='JSON map: subject -> hours')
    weak_topics = models.TextField(help_text='JSON map: subject -> list of weak topics')
    preferred_learning_method = models.CharField(max_length=20, choices=[('video', 'Video'), ('reading', 'Reading'), ('practice', 'Practice')])
    exam_score = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(100)])
    self_assessed_level = models.CharField(max_length=10, choices=PERFORMANCE_CHOICES, default='medium')
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.user.username} - {self.class_grade} - {self.created_at:%Y-%m-%d}'


class StudyRecommendation(models.Model):
    survey = models.ForeignKey(SurveyResponse, on_delete=models.CASCADE, related_name='recommendations')
    weak_subject = models.CharField(max_length=80)
    weak_topic = models.CharField(max_length=120)
    recommendation_type = models.CharField(max_length=20, choices=[('youtube', 'YouTube'), ('pdf', 'PDF Notes'), ('practice', 'Practice Strategy')])
    title = models.CharField(max_length=180)
    link = models.URLField(max_length=500, blank=True)
    rationale = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['weak_subject', 'recommendation_type']

    def __str__(self):
        return f'{self.weak_subject} - {self.recommendation_type}'


class ProgressSnapshot(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='progress_snapshots')
    subject = models.CharField(max_length=80)
    study_hours = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(24)])
    performance_score = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(100)])
    recorded_on = models.DateField()
    notes = models.CharField(max_length=250, blank=True)

    class Meta:
        ordering = ['recorded_on']

    def __str__(self):
        return f'{self.user.username} | {self.subject} | {self.recorded_on}'


class ChatMessage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='chat_messages')
    question = models.TextField()
    answer = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.user.username} chat @ {self.created_at:%Y-%m-%d %H:%M}'


class Feedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='feedback_entries')
    rating = models.PositiveSmallIntegerField(validators=[MinValueValidator(1), MaxValueValidator(5)])
    message = models.TextField(max_length=1000)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f'{self.user.username} feedback ({self.rating}/5)'
