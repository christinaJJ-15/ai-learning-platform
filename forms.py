import json

from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

from .models import Feedback, ProgressSnapshot, StudentProfile, SurveyResponse


class SignUpForm(UserCreationForm):
    email = forms.EmailField()

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ('email', 'password1', 'password2')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Username is required by Django's default User model,
        # but signup flow should be email-only.
        self.fields.pop('username', None)
        for field in self.fields.values():
            if isinstance(field.widget, forms.PasswordInput):
                field.widget.attrs['class'] = 'form-control'
            elif isinstance(field.widget, forms.EmailInput):
                field.widget.attrs['class'] = 'form-control'

    def clean_email(self):
        email = self.cleaned_data['email'].strip().lower()
        if User.objects.filter(username=email).exists():
            raise forms.ValidationError('An account with this email already exists.')
        return email

    def save(self, commit=True):
        user = super().save(commit=False)
        # Use email as username for a simpler student-first signup flow.
        user.username = self.cleaned_data['email']
        user.email = self.cleaned_data['email']
        if commit:
            user.save()
            StudentProfile.objects.update_or_create(
                user=user,
                defaults={
                    'full_name': self.cleaned_data['email'].split('@')[0].title(),
                    'location_type': 'Urban',
                },
            )
        return user


class SurveyForm(forms.ModelForm):
    subjects_studied = forms.CharField(
        widget=forms.TextInput(attrs={'placeholder': 'Math, Science, English'}),
        help_text='Comma-separated subjects'
    )
    daily_study_hours = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 3, 'placeholder': '{"Math": 2.0, "Science": 1.5}'}),
        help_text='JSON format: {"Subject": hours}'
    )
    weak_topics = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 4, 'placeholder': '{"Math": ["Algebra", "Trigonometry"]}'}),
        help_text='JSON format: {"Subject": ["weak topic 1", "weak topic 2"]}'
    )

    class Meta:
        model = SurveyResponse
        fields = [
            'student_name',
            'age',
            'location_type',
            'class_grade',
            'subjects_studied',
            'daily_study_hours',
            'weak_topics',
            'preferred_learning_method',
            'exam_score',
            'self_assessed_level',
        ]
        widgets = {
            'age': forms.NumberInput(attrs={'min': 5, 'max': 30}),
            'exam_score': forms.NumberInput(attrs={'min': 0, 'max': 100, 'step': 1}),
        }

    def clean_daily_study_hours(self):
        value = self.cleaned_data['daily_study_hours']
        try:
            payload = json.loads(value)
            if not isinstance(payload, dict) or not payload:
                raise forms.ValidationError('Provide a non-empty JSON object.')
            for subject, hours in payload.items():
                if not subject or float(hours) < 0:
                    raise forms.ValidationError('Hours must be numeric and >= 0 for all subjects.')
        except (ValueError, TypeError, json.JSONDecodeError):
            raise forms.ValidationError('Invalid JSON for daily study hours.')
        return value

    def clean_weak_topics(self):
        value = self.cleaned_data['weak_topics']
        try:
            payload = json.loads(value)
            if not isinstance(payload, dict):
                raise forms.ValidationError('Weak topics must be a JSON object.')
            for _, topics in payload.items():
                if not isinstance(topics, list):
                    raise forms.ValidationError('Each subject in weak topics must map to a list.')
        except (ValueError, TypeError, json.JSONDecodeError):
            raise forms.ValidationError('Invalid JSON for weak topics.')
        return value


class ProgressSnapshotForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, field in self.fields.items():
            existing = field.widget.attrs.get('class', '')
            base = 'form-control'
            if isinstance(field.widget, forms.Select):
                base = 'form-select'
            field.widget.attrs['class'] = f'{existing} {base}'.strip()

    class Meta:
        model = ProgressSnapshot
        fields = ['subject', 'study_hours', 'performance_score', 'recorded_on', 'notes']
        widgets = {
            'recorded_on': forms.DateInput(attrs={'type': 'date'}),
            'notes': forms.TextInput(attrs={'placeholder': 'Optional notes'}),
        }


class FeedbackForm(forms.ModelForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['rating'].label = 'Overall Rating'
        self.fields['message'].label = 'Message'

    class Meta:
        model = Feedback
        fields = ['rating', 'message']
        widgets = {
            'rating': forms.Select(
                choices=[(i, f'{i} / 5') for i in range(5, 0, -1)],
                attrs={'class': 'form-select'}
            ),
            'message': forms.Textarea(
                attrs={
                    'class': 'form-control',
                    'rows': 5,
                    'placeholder': 'Tell us what worked well and what should be improved...',
                }
            ),
        }
