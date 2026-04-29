# AI-Powered Personalized Learning Platform

End-to-end Django web app for analyzing and improving individual student study patterns with personalized recommendations.

## Tech Stack
- Frontend: HTML, CSS, JavaScript, Chart.js
- Backend: Django (Python)
- ML: scikit-learn (K-Means + RandomForestRegressor + RandomForestClassifier)
- Database: SQLite (default, PostgreSQL-ready)
- Reporting: PDF export (reportlab)

## Folder Structure

```text
ai_learning/
+-- manage.py
+-- requirements.txt
+-- db.sqlite3
+-- ai_learning/
¦   +-- settings.py
¦   +-- urls.py
¦   +-- ...
+-- student_analysis/
    +-- admin.py
    +-- forms.py
    +-- models.py
    +-- services.py
    +-- views.py
    +-- urls.py
    +-- ml_model.py
    +-- data/
    ¦   +-- sample_student_survey.csv
    +-- migrations/
    ¦   +-- 0001_initial.py
    ¦   +-- 0002_progresssnapshot_studentprofile_studyrecommendation_and_more.py
    ¦   +-- 0003_surveyresponse_age_surveyresponse_student_name_and_more.py
    +-- templates/
        +-- base.html
        +-- index.html
        +-- signup.html
        +-- login.html
        +-- survey.html
        +-- dashboard.html
        +-- progress.html
        +-- admin_analytics.html
```

## Database Schema

### StudentProfile
- user (OneToOne -> auth user)
- full_name
- location_type (Rural/Urban)
- created_at

### SurveyResponse
- user (FK)
- student_name
- age
- location_type
- class_grade
- subjects_studied (comma-separated)
- daily_study_hours (JSON string)
- weak_topics (JSON string)
- preferred_learning_method (video/reading/practice)
- exam_score
- self_assessed_level
- created_at

### StudyRecommendation
- survey (FK)
- weak_subject
- weak_topic
- recommendation_type (youtube/pdf/practice)
- title, link, rationale
- created_at

### ProgressSnapshot
- user (FK)
- subject
- study_hours
- performance_score
- recorded_on
- notes

### ChatMessage
- user (FK)
- question
- answer
- created_at

## Main Features
1. Secure authentication and session flow (signup/login/logout).
2. Interactive survey with checkboxes, sliders, dropdowns, and text entry.
3. ML analysis:
   - K-Means clustering for study behavior grouping.
   - Regression for score trend prediction.
   - Classification for predicted performance level.
4. Personalized AI recommendations and study schedule generation.
5. Dashboard analytics:
   - Subject-wise performance proxies.
   - Study hours vs improvement trend.
   - Weak-topic analysis.
   - Progress over time.
6. Study chatbot for topic guidance and schedule clarifications.
7. Study material search integrated with weak-topic context.
8. Admin dashboard for aggregate rural vs urban comparisons.
9. PDF report export.

## API Endpoints
- GET `/api/recommendations/`
- GET `/api/analytics/`
- POST `/api/chatbot/`
- GET `/api/search-materials/?q=<query>`

## Setup
```bash
pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser
python manage.py runserver
```

## User Flow
1. Open `/`
2. Sign up (`/signup/`)
3. Complete survey (`/survey/`)
4. Use home dashboard modules (`/dashboard/`)
5. Add progress snapshots (`/progress/`)
6. Export PDF (`/export-report/`)
7. Admin analytics (`/admin-analytics/`)

## Notes
- PDF export requires `reportlab`.
- Default DB is SQLite; switch to PostgreSQL in `settings.py` for production.
