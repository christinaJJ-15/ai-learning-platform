import json

from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required, user_passes_test
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_GET, require_http_methods

from .forms import FeedbackForm, ProgressSnapshotForm, SignUpForm, SurveyForm
from .models import Feedback, ProgressSnapshot, StudentProfile, StudyRecommendation, SurveyResponse
from .services import (
    build_personalized_detailed_summary,
    ensure_baseline_progress,
    run_ml_analysis,
    search_study_materials,
)


def home(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'index.html')


def signup_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    form = SignUpForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.save()
        login(request, user)
        messages.success(request, 'Account created successfully.')
        return redirect('survey')
    if request.method == 'POST' and not form.is_valid():
        messages.error(request, 'Signup failed. Please correct the highlighted fields.')

    return render(request, 'signup.html', {'form': form})


def login_view(request):
    if request.user.is_authenticated:
        return redirect('dashboard')

    error = None
    if request.method == 'POST':
        email = (request.POST.get('email') or '').strip().lower()
        password = request.POST.get('password') or ''
        user = authenticate(request, username=email, password=password)
        if user:
            login(request, user)
            messages.success(request, 'Logged in successfully.')
            return redirect('dashboard')
        error = 'Invalid email or password.'

    return render(request, 'login.html', {'error': error})


@login_required
def logout_view(request):
    logout(request)
    return redirect('home')


@login_required
def survey_view(request):
    profile = StudentProfile.objects.filter(user=request.user).first()
    initial = {
        'student_name': profile.full_name if profile else request.user.username,
        'location_type': profile.location_type if profile else 'Urban',
    }

    form = SurveyForm(request.POST or None, initial=initial)
    if request.method == 'POST' and form.is_valid():
        survey = form.save(commit=False)
        survey.user = request.user
        survey.save()

        analysis = run_ml_analysis(survey)
        # Old per-topic recommendation records are no longer generated.
        StudyRecommendation.objects.filter(survey=survey).delete()

        ensure_baseline_progress(request.user, survey)
        messages.success(request, 'Survey submitted and your personalized detailed summary is ready.')
        return redirect('dashboard')

    return render(request, 'survey.html', {'form': form})


@login_required
def dashboard_view(request):
    latest_survey = SurveyResponse.objects.filter(user=request.user).first()
    if not latest_survey:
        messages.info(request, 'Complete your study survey to unlock analytics.')
        return redirect('survey')

    analysis = run_ml_analysis(latest_survey)
    detailed = build_personalized_detailed_summary(latest_survey, analysis)

    context = {
        'survey': latest_survey,
        'analysis': analysis,
        'detailed_recommendation': detailed,
    }
    return render(request, 'dashboard.html', context)

@login_required
@require_http_methods(['GET', 'POST'])
def feedback_view(request):
    form = FeedbackForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            entry = form.save(commit=False)
            entry.user = request.user
            entry.save()
            messages.success(request, 'Thanks for your feedback.')
            return redirect('feedback')
        messages.error(request, 'Please fix feedback fields and submit again.')

    recent_feedback = Feedback.objects.filter(user=request.user)[:10]
    return render(request, 'feedback.html', {'form': form, 'recent_feedback': recent_feedback})


@login_required
@require_http_methods(['GET', 'POST'])
def progress_view(request):
    form = ProgressSnapshotForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        row = form.save(commit=False)
        row.user = request.user
        row.save()
        messages.success(request, 'Progress snapshot saved.')
        return redirect('progress')

    snapshots = ProgressSnapshot.objects.filter(user=request.user).order_by('-recorded_on')[:20]
    return render(request, 'progress.html', {'form': form, 'snapshots': snapshots})


@user_passes_test(lambda u: u.is_staff)
def admin_analytics_view(request):
    surveys = SurveyResponse.objects.all()
    rural = surveys.filter(location_type='Rural')
    urban = surveys.filter(location_type='Urban')

    def avg_score(qs):
        scores = [row.exam_score for row in qs]
        return round(sum(scores) / len(scores), 2) if scores else 0

    context = {
        'total_students': surveys.count(),
        'rural_students': rural.count(),
        'urban_students': urban.count(),
        'avg_rural_score': avg_score(rural),
        'avg_urban_score': avg_score(urban),
        'rural_urban_score_json': json.dumps([avg_score(rural), avg_score(urban)]),
    }
    return render(request, 'admin_analytics.html', context)


@login_required
@require_GET
def export_pdf_report(request):
    survey = SurveyResponse.objects.filter(user=request.user).first()
    if not survey:
        messages.error(request, 'No survey found to export.')
        return redirect('survey')

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
    except ImportError:
        return HttpResponse(
            'PDF export requires reportlab. Install with: pip install reportlab',
            content_type='text/plain',
            status=501,
        )

    analysis = run_ml_analysis(survey)
    detailed = build_personalized_detailed_summary(survey, analysis)

    response = HttpResponse(content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="study_report.pdf"'

    p = canvas.Canvas(response, pagesize=A4)
    _, height = A4
    y = height - 50

    p.setFont('Helvetica-Bold', 15)
    p.drawString(40, y, 'Student Study Analytics Report')
    y -= 26

    p.setFont('Helvetica', 10)
    lines = [
        f'Name: {survey.student_name}',
        f'Age: {survey.age}',
        f'Location: {survey.location_type}',
        f'Class/Grade: {survey.class_grade}',
        f'Exam Score: {survey.exam_score}',
        f'Predicted Score Trend: {analysis["predicted_score"]}',
        f'Predicted Level: {analysis["predicted_level"]}',
        f'Cluster ID: {analysis["cluster_id"]}',
        f'Consistency Insight: {analysis["consistency_issue"]}',
        '',
        'Personalized Detailed Summary:',
    ]

    lines.append((detailed.get('opener') or '')[:110])
    for item in (detailed.get('root_causes') or [])[:5]:
        lines.append(f'- Root cause: {item}'[:110])
    for area in (detailed.get('lagging_areas') or [])[:5]:
        topics = ', '.join(area.get('topics') or [])[:60]
        lines.append(f"- Lagging: {area.get('subject')} ({topics})"[:110])
        for step in (area.get('what_to_do') or [])[:2]:
            lines.append(f"  * {step}"[:110])

    for line in lines:
        p.drawString(40, y, line[:110])
        y -= 16
        if y < 60:
            p.showPage()
            p.setFont('Helvetica', 10)
            y = height - 50

    p.save()
    return response


@login_required
@require_GET
def search_materials_api(request):
    query = request.GET.get('q', '')
    survey = SurveyResponse.objects.filter(user=request.user).first()
    return JsonResponse({'results': search_study_materials(query, survey)})


@login_required
@require_GET
def recommendations_api(request):
    survey = SurveyResponse.objects.filter(user=request.user).first()
    if not survey:
        return JsonResponse({'error': 'survey_not_found'}, status=404)

    analysis = run_ml_analysis(survey)
    detailed = build_personalized_detailed_summary(survey, analysis)
    return JsonResponse({'recommendations': detailed})


@login_required
@require_GET
def analytics_api(request):
    survey = SurveyResponse.objects.filter(user=request.user).first()
    if not survey:
        return JsonResponse({'error': 'survey_not_found'}, status=404)

    analysis = run_ml_analysis(survey)
    progress = build_progress_chart_data(request.user)
    return JsonResponse({'analysis': analysis, 'progress': progress})
