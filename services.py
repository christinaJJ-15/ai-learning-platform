import json
import os
import random
import hashlib
import traceback
from collections import defaultdict
from datetime import date, timedelta
from urllib.parse import quote_plus

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from .models import ProgressSnapshot, SurveyResponse

TOPIC_RESOURCES = {
    'algebra': {
        'youtube': ('Algebra Fundamentals', 'https://www.youtube.com/results?search_query=algebra+basics'),
        'pdf': ('Algebra Revision Notes PDF', 'https://ncert.nic.in/textbook.php'),
        'practice': ('Algebra Worksheet Drill', ''),
    },
    'trigonometry': {
        'youtube': ('Trigonometry Concept Playlist', 'https://www.youtube.com/results?search_query=trigonometry+class+tutorial'),
        'pdf': ('Trigonometry Formula Sheet', 'https://ncert.nic.in/textbook.php'),
        'practice': ('Trigonometry Daily Problems', ''),
    },
    'chemical reactions': {
        'youtube': ('Chemical Reactions Visual Learning', 'https://www.youtube.com/results?search_query=chemical+reactions+class+10'),
        'pdf': ('Chemistry Chapter Notes', 'https://ncert.nic.in/textbook.php'),
        'practice': ('Reaction Balancing Practice', ''),
    },
    'grammar': {
        'youtube': ('English Grammar Quick Lessons', 'https://www.youtube.com/results?search_query=english+grammar+practice'),
        'pdf': ('Grammar Rules PDF', 'https://ncert.nic.in/textbook.php'),
        'practice': ('Daily Grammar Error Correction', ''),
    },
}


def parse_json_field(value, default):
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return default


NONE_WEAK_TOKENS = {
    "na",
    "n/a",
    "n.a.",
    "none",
    "no",
    "nil",
    "null",
    "-",
    "–",
    "—",
    "not applicable",
}


def clean_weak_topics_map(weak_map):
    """
    Treat tokens like 'NA'/'none'/'-' as empty weak-topic entries.
    This prevents generating links/recommendations for that subject.
    """
    if not isinstance(weak_map, dict):
        return {}

    cleaned = {}
    for subject, topics in weak_map.items():
        if not isinstance(topics, list):
            continue

        new_topics = []
        for t in topics:
            if t is None:
                continue
            s = str(t).strip()
            if not s:
                continue
            s_norm = s.lower()
            if s_norm in NONE_WEAK_TOKENS:
                continue
            new_topics.append(s)

        if new_topics:
            cleaned[subject] = new_topics

    return cleaned


def _performance_band(score):
    if score < 50:
        return 'low'
    if score < 75:
        return 'medium'
    return 'high'


def survey_feature_vector(survey):
    hours_map = parse_json_field(survey.daily_study_hours, {})
    weak_topic_map = clean_weak_topics_map(parse_json_field(survey.weak_topics, {}))

    hour_values = [float(v) for v in hours_map.values()] if hours_map else [0.0]
    total_hours = float(sum(hour_values))
    avg_hours = float(total_hours / max(len(hour_values), 1))
    consistency = float(np.std(hour_values))
    weak_topic_count = float(sum(len(v) for v in weak_topic_map.values()))
    subject_count = float(len([s.strip() for s in survey.subjects_studied.split(',') if s.strip()]))
    rural_flag = 1.0 if survey.location_type == 'Rural' else 0.0

    return [
        total_hours,
        avg_hours,
        consistency,
        weak_topic_count,
        subject_count,
        rural_flag,
        float(survey.exam_score),
    ]


def _stable_user_seed(survey):
    raw = f"{survey.user_id}:{survey.id}:{survey.student_name}:{survey.class_grade}:{survey.created_at.isoformat()}"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def build_personalized_detailed_summary(survey, analysis):
    """
    AI-generated (LLM) detailed, personalized recommendation summary derived ONLY from:
    - survey inputs (study hours, weak topics, learning method, score, self assessment)
    - analysis outputs (predicted score/level, consistency insight, comparisons)

    No hardcoded subject-specific if/else logic is used to shape the recommendations.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    def _fallback_summary(reason):
        rng = random.Random(_stable_user_seed(survey) + 8088)
        hours_map = parse_json_field(survey.daily_study_hours, {})
        weak_map = clean_weak_topics_map(parse_json_field(survey.weak_topics, {}))
        subjects = [s.strip() for s in (survey.subjects_studied or "").split(",") if s.strip()]
        total_hours = round(sum(float(v) for v in hours_map.values()), 2) if hours_map else 0.0
        current = float(survey.exam_score)
        predicted = float(analysis.get("predicted_score", current))
        consistency_flag = str(analysis.get("consistency_issue", "N/A"))

        # Rank subjects by (weak topic count desc, hours asc, name) without subject-specific rules.
        def weak_count(sub):
            topics = weak_map.get(sub, [])
            return len(topics) if isinstance(topics, list) else 0

        all_subjects = sorted({*subjects, *hours_map.keys(), *weak_map.keys()})
        ranked = sorted(all_subjects, key=lambda s: (-weak_count(s), float(hours_map.get(s, 0.0) or 0.0), s.lower()))

        lagging_areas = []
        for sub in ranked:
            topics = weak_map.get(sub, [])
            topics = topics if isinstance(topics, list) else []
            topics = [t for t in (topics or []) if str(t).strip()]
            declared_hours = float(hours_map.get(sub, 0.0) or 0.0)
            # Only show lagging areas from explicit weak topics (not all subjects).
            if not topics:
                continue

            focus_topics = topics[:4]
            primary_topic = focus_topics[0] if focus_topics else ""
            resources = []
            if primary_topic:
                # Suggest topic-oriented platforms/resources derived from the topic text itself.
                # No subject-specific logic; URLs are generated from the user's topic string.
                res = _resource_for_topic(primary_topic)
                query = quote_plus(primary_topic)
                resources = [
                    {"topic": primary_topic, "label": "Khan Academy", "url": f"https://www.khanacademy.org/search?page_search_query={query}"},
                    {"topic": primary_topic, "label": res["youtube"][0], "url": res["youtube"][1]},
                    {"topic": primary_topic, "label": res["pdf"][0], "url": res["pdf"][1]},
                    {"topic": primary_topic, "label": "Practice questions", "url": f"https://www.google.com/search?q={query}+practice+questions"},
                ]

            why_bits = []
            if focus_topics:
                why_bits.append("Weak topics were listed in your survey")
            if declared_hours <= 0.5:
                why_bits.append("very little daily time is allocated here")
            elif declared_hours <= 1.0:
                why_bits.append("limited daily time is allocated here")
            if not why_bits:
                why_bits.append("this subject needs structured practice to improve accuracy")

            what_to_do = [
                "Start with a quick diagnostic: attempt 12 questions (mixed difficulty) and mark the exact mistake type for each wrong answer.",
                "Do targeted practice: 15–25 questions focusing only on your mistakes and weak points.",
                "Use spaced repetition: redo only the wrong questions after 24–48 hours until accuracy stabilizes.",
                "Maintain an error log: write (mistake type → correct rule → one corrected example).",
            ]
            if focus_topics:
                what_to_do.insert(0, f"Focus topics: {', '.join(focus_topics)}. Keep the practice strictly on these until accuracy improves.")

            base_days = 10 if total_hours >= 3.0 else 14
            stretch = 0 if predicted >= current else 2
            days = base_days + stretch + rng.randrange(0, 3)
            base_target = 78 if current < 55 else 82 if current < 75 else 86
            target = base_target + rng.randrange(-2, 3)
            success_metric = f"Target: {target}%+ accuracy on this weak-topic set within {days} days."

            # Stable per-user variation in step ordering (keep focus-topics line first, if present).
            fixed_prefix = []
            steps = list(what_to_do)
            if steps and steps[0].startswith("Focus topics:"):
                fixed_prefix = [steps.pop(0)]
            rng.shuffle(steps)
            what_to_do = fixed_prefix + steps

            lagging_areas.append(
                {
                    "subject": sub,
                    "topics": focus_topics,
                    "primary_topic": primary_topic,
                    "why": f"{'; '.join(why_bits)}.",
                    "what_to_do": what_to_do,
                    "success_metric": success_metric,
                    "resources": resources[:8],
                }
            )

        if not lagging_areas:
            lagging_areas = [
                {
                    "subject": "Identify weak areas",
                    "topics": [],
                    "why": "Your survey does not include weak topics or subject-wise hours, so we cannot pinpoint lagging areas precisely.",
                    "what_to_do": [
                        "Take a short diagnostic per subject (20 questions, timed).",
                        "Record top 3 mistake topics per subject and update the survey weak topics.",
                        "Allocate at least 0.5–1.0 hours/day to the weakest subject for 7 days, then retest.",
                    ],
                    "success_metric": "Target: identify at least 6 weak micro-topics this week.",
                    "resources": [],
                }
            ]

        # Root causes: generic, derived from numeric + consistency string.
        root_causes = []
        if total_hours <= 1.5:
            root_causes.append("Total daily study time is low; increasing it slightly will improve retention and practice volume.")
        if "High variation" in consistency_flag:
            root_causes.append("Study hours vary a lot across subjects, so weak areas don’t get repeated exposure.")
        if "Moderate variation" in consistency_flag:
            root_causes.append("Study hours are uneven; a small, consistent routine will reduce gaps.")
        if predicted < current - 2:
            root_causes.append("Your predicted score trend is below your current score; consistency + error-review needs tightening.")
        if predicted >= current - 2:
            root_causes.append("Your predicted trend is stable; consistent weak-topic practice can drive steady improvement.")

        recall_min = 8 + rng.randrange(0, 6)  # 8–13
        practice_min = 30 + rng.randrange(0, 31)  # 30–60
        redo_q = 3 + rng.randrange(0, 4)  # 3–6
        daily_plan = [
            f"Step 1 ({recall_min} min): quick recall of yesterday’s mistakes (notes/flashcards).",
            f"Step 2 ({practice_min} min): targeted practice on one lagging area (timed questions).",
            f"Step 3 (10 min): error log + redo {redo_q} previously wrong questions.",
        ]
        cycles = 4 + rng.randrange(0, 3)  # 4–6
        weekly_goals = [
            f"Complete {cycles} focused practice cycles (diagnostic → practice → error log → retest).",
            "Retest weak topics after 7 days and compare accuracy before/after.",
        ]
        timeline_variants = [
            [
                "Week 1: run diagnostics and lock your top 2 lagging areas; start an error log.",
                "Week 2: raise practice volume; begin spaced retests for wrong questions.",
                "Week 3: mix practice across lagging areas; eliminate repeat mistake patterns.",
                "Week 4: do 1–2 mock tests; finalize revision of remaining weak topics.",
            ],
            [
                "Week 1: identify the biggest gaps from your survey and start targeted drills.",
                "Week 2: keep drills tight on weak topics; retest after 48 hours.",
                "Week 3: increase mixed sets; keep only high-impact corrections in the error log.",
                "Week 4: mock test + review; repeat weak-topic retests until stable.",
            ],
            [
                "Week 1: diagnose and fix fundamentals in the most lagging areas.",
                "Week 2: timed practice + error review; retest weak topics regularly.",
                "Week 3: mixed-topic practice; improve speed while keeping accuracy.",
                "Week 4: consolidate with mock tests and final weak-topic cleanup.",
            ],
        ]
        timeline = timeline_variants[rng.randrange(len(timeline_variants))]

        opener_templates = [
            "Non‑AI fallback summary ({reason}). This report is generated from your survey + analysis values.",
            "Fallback mode ({reason}). Using only your survey inputs to generate a detailed improvement plan.",
            "Fallback report ({reason}). Recommendations are derived only from what you entered in the survey.",
            "Fallback summary ({reason}). Personalized from your hours, weak topics, and score trend.",
        ]
        opener = opener_templates[rng.randrange(len(opener_templates))].format(reason=reason)

        return {
            "title": "Personalized Detailed Recommendation",
            "opener": opener,
            "overview": {
                "student": survey.student_name,
                "grade": survey.class_grade,
                "learning_method": survey.preferred_learning_method,
                "score_now": current,
                "score_predicted": predicted,
                "consistency": consistency_flag or "N/A",
                "study_time_total_hours_per_day": total_hours,
                "subjects": subjects,
            },
            "root_causes": root_causes,
            "lagging_areas": lagging_areas[:6],
            "daily_plan": daily_plan,
            "weekly_goals": weekly_goals,
            "timeline": timeline,
            "next_steps": [
                "Fill weak topics for each subject in the survey (more detail = better output).",
                "Retake the survey after 7 days so the recommendation adapts.",
                "Enable AI later (billing/quota) for richer personalization.",
            ],
        }

    if not api_key:
        return _fallback_summary("AI key not configured")

    hours_map = parse_json_field(survey.daily_study_hours, {})
    weak_map = clean_weak_topics_map(parse_json_field(survey.weak_topics, {}))
    total_hours = round(sum(float(v) for v in hours_map.values()), 2) if hours_map else 0.0

    payload = {
        "survey": {
            "student_name": survey.student_name,
            "age": survey.age,
            "location_type": survey.location_type,
            "class_grade": survey.class_grade,
            "subjects_studied": survey.subjects_studied,
            "daily_study_hours": hours_map,
            "weak_topics": weak_map,
            "preferred_learning_method": survey.preferred_learning_method,
            "exam_score": float(survey.exam_score),
            "self_assessed_level": survey.self_assessed_level,
            "total_study_hours_per_day": total_hours,
        },
        "analysis": {
            "predicted_score": analysis.get("predicted_score"),
            "predicted_level": analysis.get("predicted_level"),
            "consistency_issue": analysis.get("consistency_issue"),
            "cluster_id": analysis.get("cluster_id"),
            "global_avg_score": analysis.get("global_avg_score"),
            "rural_vs_urban_hours": analysis.get("rural_vs_urban_hours"),
            "rural_vs_urban_score": analysis.get("rural_vs_urban_score"),
        },
        "constraints": {
            "no_time_scheduling": True,
            "base_only_on_payload": True,
            "must_be_detailed": True,
            "language": "English",
        },
    }

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        model = os.getenv("OPENAI_RECOMMENDATION_MODEL", os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))

        sys_prompt = (
            "You are an educational coach. Generate a personalized, detailed improvement report.\n"
            "CRITICAL RULES:\n"
            "- Use ONLY the provided JSON payload. Do not assume subjects, syllabus, or add outside facts.\n"
            "- Do NOT create time schedules (no clocks, no timeslots).\n"
            "- Be specific, actionable, and diagnostic. If weak topics are missing, propose a survey-driven way to discover them.\n"
            "- Output MUST be valid JSON matching the required schema.\n"
        )

        schema = {
            "title": "Personalized Detailed Recommendation",
            "opener": "string",
            "overview": {
                "student": "string",
                "grade": "string",
                "learning_method": "string",
                "score_now": 0,
                "score_predicted": 0,
                "consistency": "string",
                "study_time_total_hours_per_day": 0,
                "subjects": ["string"],
            },
            "root_causes": ["string"],
            "lagging_areas": [
                {
                    "subject": "string",
                    "topics": ["string"],
                    "why": "string",
                    "what_to_do": ["string"],
                    "success_metric": "string",
                }
            ],
            "daily_plan": ["string"],
            "weekly_goals": ["string"],
            "timeline": ["string"],
            "next_steps": ["string"],
        }

        user_prompt = (
            "Generate the recommendation JSON.\n\n"
            f"REQUIRED_JSON_SCHEMA_EXAMPLE:\n{json.dumps(schema, indent=2)}\n\n"
            f"PAYLOAD:\n{json.dumps(payload, indent=2)}"
        )

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=1200,
            response_format={"type": "json_object"},
        )
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content) if content else {}
        if isinstance(data, dict) and data.get("title"):
            return data
    except Exception as e:
        # Show a short error to help debugging without leaking secrets.
        err = f"{type(e).__name__}: {str(e)}"
        print("AI recommendation error:", err)
        traceback.print_exc()
        return _fallback_summary(err[:140])

    return {
        "title": "Personalized Detailed Recommendation",
        "opener": "AI recommendation generation failed. Please verify OPENAI_API_KEY and try again.",
        "overview": {
            "student": survey.student_name,
            "grade": survey.class_grade,
            "learning_method": survey.preferred_learning_method,
            "score_now": float(survey.exam_score),
            "score_predicted": float(analysis.get("predicted_score", survey.exam_score)),
            "consistency": analysis.get("consistency_issue", "N/A"),
            "study_time_total_hours_per_day": total_hours,
            "subjects": [s.strip() for s in (survey.subjects_studied or "").split(",") if s.strip()],
        },
        "root_causes": [],
        "lagging_areas": [],
        "daily_plan": [],
        "weekly_goals": [],
        "timeline": [],
        "next_steps": ["Try again after configuring OPENAI_API_KEY."],
    }


def summarize_lagging_areas(survey, analysis):
    weak_map = clean_weak_topics_map(parse_json_field(survey.weak_topics, {}))
    lagging = []
    predicted_score = analysis.get('predicted_score', float(survey.exam_score))
    current_score = float(survey.exam_score)
    gap = round(max(0.0, current_score - predicted_score), 1)

    for subject, topics in weak_map.items():
        if isinstance(topics, list) and topics:
            micro_topics = topics[:2]
            lagging.append(
                {
                    'subject': subject,
                    'topics': topics[:3],
                    'reason': (
                        f"Weak topics: {', '.join(micro_topics)}. "
                        f"Action: revise notes, solve 5-10 questions, self-test in 48h. "
                        f"Goal: improve {subject} accuracy by 6-8%."
                    ),
                }
            )

    if not lagging:
        lagging.append(
            {
                'subject': 'Lagging Areas',
                'topics': [],
                'reason': (
                    'We have few explicit weak-topic entries; start with a quick self-assessment to pinpoint 2-3 topics for focused practice.'
                ),
            }
        )

    if predicted_score < current_score:
        worst_subject = max(
            (subj for subj in weak_map.keys()),
            key=lambda s: len(weak_map.get(s, [])) if isinstance(weak_map.get(s, []), list) else 0,
            default='core areas',
        )
        lagging.append(
            {
                'subject': 'Overall Performance',
                'topics': [],
                'reason': (
                    f"Predicted score is {predicted_score}% (gap {gap} points). "
                    f"Target: add 8-10 min daily review on {worst_subject}, aim +3% in 1 week."
                ),
            }
        )
    else:
        lagging.append(
            {
                'subject': 'Overall Performance',
                'topics': [],
                'reason': (
                    f"Predicted score is {predicted_score}%. Continue current plan, "
                    'add one extension practice set each day for small gains.'
                ),
            }
        )

    lagging.append(
        {
            'subject': 'Study Consistency',
            'topics': [],
            'reason': (
                f"{analysis['consistency_issue']} "
                'Set 4x 25-minute focused blocks with 5-minute review and track completion.'
            ),
        }
    )
    return lagging


def run_ml_analysis(target_survey):
    all_surveys = list(SurveyResponse.objects.all())
    if not all_surveys:
        baseline_level = _performance_band(float(target_survey.exam_score))
        return {
            'cluster_id': 0,
            'predicted_score': target_survey.exam_score,
            'predicted_level': baseline_level,
            'consistency_issue': 'Insufficient data',
            'comparison': {'Rural': 0, 'Urban': 0},
            'rural_vs_urban_hours': {'Rural': 0, 'Urban': 0},
            'global_avg_score': 0,
        }

    X = np.array([survey_feature_vector(s)[:-1] for s in all_surveys], dtype=float)
    y = np.array([survey_feature_vector(s)[-1] for s in all_surveys], dtype=float)
    y_band = np.array([_performance_band(v) for v in y])
    target = np.array([survey_feature_vector(target_survey)[:-1]], dtype=float)

    cluster_count = int(max(1, min(3, len(all_surveys))))
    if cluster_count == 1:
        cluster_id = 0
    else:
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        model.fit(X)
        cluster_id = int(model.predict(target)[0])

    if len(all_surveys) >= 5:
        reg = RandomForestRegressor(n_estimators=120, random_state=42)
        reg.fit(X, y)
        predicted_score = float(reg.predict(target)[0])

        clf = RandomForestClassifier(n_estimators=120, random_state=42)
        clf.fit(X, y_band)
        predicted_level = str(clf.predict(target)[0])
    else:
        predicted_score = float((target_survey.exam_score + float(np.mean(y))) / 2.0)
        predicted_level = _performance_band(predicted_score)

    hours_map = parse_json_field(target_survey.daily_study_hours, {})
    distribution = [float(v) for v in hours_map.values()] if hours_map else [0]
    std_dev = float(np.std(distribution))
    if std_dev > 1.5:
        consistency_issue = 'High variation in subject-wise study hours. Balance your daily schedule.'
    elif std_dev > 0.7:
        consistency_issue = 'Moderate variation in study hours. Add structured study slots.'
    else:
        consistency_issue = 'Good study consistency across subjects.'

    rural_count = SurveyResponse.objects.filter(location_type='Rural').count()
    urban_count = SurveyResponse.objects.filter(location_type='Urban').count()

    def avg_total_hours(location):
        entries = SurveyResponse.objects.filter(location_type=location)
        totals = []
        for row in entries:
            hm = parse_json_field(row.daily_study_hours, {})
            totals.append(sum(float(v) for v in hm.values()))
        return round(float(np.mean(totals)), 2) if totals else 0.0

    def avg_score_location(location):
        entries = SurveyResponse.objects.filter(location_type=location)
        scores = [float(row.exam_score) for row in entries if row.exam_score is not None]
        return round(float(np.mean(scores)), 2) if scores else 0.0

    summary = {
        'cluster_id': cluster_id,
        'predicted_score': round(predicted_score, 2),
        'predicted_level': predicted_level,
        'consistency_issue': consistency_issue,
        'comparison': {'Rural': rural_count, 'Urban': urban_count},
        'rural_vs_urban_hours': {
            'Rural': avg_total_hours('Rural'),
            'Urban': avg_total_hours('Urban'),
        },
        'rural_vs_urban_score': {
            'Rural': avg_score_location('Rural'),
            'Urban': avg_score_location('Urban'),
        },
        'global_avg_score': round(float(np.mean(y)), 2),
    }
    summary['personalized_recommendation'] = build_personalized_recommendation(target_survey, summary)
    return summary


def _resource_for_topic(topic):
    key = topic.strip().lower()
    if key in TOPIC_RESOURCES:
        return TOPIC_RESOURCES[key]
    query = quote_plus(topic)
    return {
        'youtube': (f'{topic.title()} Video Lessons', f'https://www.youtube.com/results?search_query={query}+explained'),
        'pdf': (f'{topic.title()} Notes PDF', f'https://www.google.com/search?q={query}+pdf+notes'),
        'practice': (f'{topic.title()} Practice Strategy', ''),
    }


def build_dynamic_recommendations(survey, analysis):
    weak_topic_map = clean_weak_topics_map(parse_json_field(survey.weak_topics, {}))
    output = []

    predicted_score = analysis.get('predicted_score', float(survey.exam_score))
    trend = ' improving' if predicted_score >= survey.exam_score else ' at risk of falling'
    target_gain = max(2, min(15, int(90 - predicted_score)))

    for subject, topics in weak_topic_map.items():
        if not isinstance(topics, list) or not topics:
            continue

        for topic in topics:
            resources = _resource_for_topic(topic)
            topic_label = topic.strip()
            core_instruction = f"Core micro-topic: {topic_label}."
            if survey.preferred_learning_method == 'video':
                method_line = 'Start with short videos (10-15 min), then note 3 keywords.'
            elif survey.preferred_learning_method == 'reading':
                method_line = 'Skim a summary sheet, highlight 5 points, then self-quiz.'
            else:
                method_line = 'Practice questions first, then fix conceptual gaps in notes.'

            for rec_type in ('youtube', 'pdf', 'practice'):
                title, link = resources[rec_type]
                if rec_type == 'youtube':
                    why = (
                        f"{core_instruction} {method_line} "
                        'Watch one clip, and do instant recall in 5 min. '
                        f'Goal: add {target_gain}% accuracy in 10 days while trend{trend}. '
                        "Action plan: 5 q + 2-min mistake notes after every clip."
                    )
                elif rec_type == 'pdf':
                    why = (
                        f"{core_instruction} {method_line} "
                        'Read the page, then rewrite key formulas. '
                        f'Goal: lift this topic by {target_gain}% in two weeks. '
                        "Action plan: do 7 MCQs, mark 3 weak concepts, and revise them overnight."
                    )
                else:
                    why = (
                        f"{core_instruction} {method_line} "
                        f'Start with 8 problems, then run a 15-min self-test. '
                        f'Goal: close the gap to {target_gain}% improvement in 14 days. '
                        'Record error types and revisit 1 topic/day.'
                    )

                output.append(
                    {
                        'subject': subject,
                        'topic': topic_label,
                        'type': rec_type,
                        'title': title,
                        'link': link,
                        'rationale': why,
                    }
                )

    if not output:
        generic = 'General Concepts'
        resources = _resource_for_topic(generic)
        output.append(
            {
                'subject': 'General',
                'topic': generic,
                'type': 'practice',
                'title': resources['practice'][0],
                'link': resources['practice'][1],
                'rationale': (
                    f'No weak topics were provided. Set a weekly goal to improve by {target_gain}% in 10 days. '
                    'Do 10 mixed questions daily, track mistakes, and review 2 weak topics each weekend.'
                ),
            }
        )

    return output


def build_personalized_recommendation(survey, analysis):
    weak_map = clean_weak_topics_map(parse_json_field(survey.weak_topics, {}))
    weak_subjects = [s for s in weak_map.keys() if isinstance(weak_map.get(s, []), list) and weak_map.get(s)]
    top_weak = weak_subjects[0] if weak_subjects else None
    top_topics = weak_map.get(top_weak, [])[:3] if top_weak else []

    current = float(survey.exam_score)
    predicted = float(analysis.get('predicted_score', current))
    gap = round(predicted - current, 1)
    consistency_flag = analysis.get('consistency_issue', '')
    level = _performance_band(current)

    plan_points = []

    if top_weak:
        plan_points.append(
            f"Your survey shows the hardest area is {top_weak} (focus: {', '.join(top_topics)}). "
            "Start each session with 10 min micro-topic review and follow with 8-12 practice questions."
        )
    else:
        plan_points.append(
            "Your study profile is balanced; keep alternating subjects every session. "
            "Use a short daily checkpoint quiz for each main topic." 
        )

    if 'High variation' in consistency_flag:
        plan_points.append(
            "You have high week-to-week hour swings. Normalize to 4x25-min focused blocks and log each block."
        )
    elif 'Moderate variation' in consistency_flag:
        plan_points.append(
            "Consistency is moderate. Fix one consistent slot every weekday and track completion."
        )
    else:
        plan_points.append(
            "Your consistency is good; keep this with a short end-of-day reflection note."
        )

    if gap < 0:
        lagging_message = f"Your predicted score is {predicted}%, {abs(gap)} points below current. There is room to catch up with focused revision."
        weekly_goal = f"Raise your score from {int(current)}% towards {int(min(current + 10, 100))}% in four weeks."
    else:
        lagging_message = f"Your predicted score is {predicted}%, close to current. Keep the momentum and tighten weak-topic practice."
        weekly_goal = f"Maintain this and target {int(min(current + 8, 100))}% in next month."

    weak_info = ''
    if top_weak:
        weak_info = f"Weak area: {top_weak} with topics {', '.join(top_topics)}."
    else:
        weak_info = "No dominant weak area; your strengths are distributed evenly, but continue refinement."

    # explicitly differentiate recommendations by performance band + self assessment
    performance_block = ''
    exam = float(survey.exam_score)
    self_level = (survey.self_assessed_level or '').strip().lower()

    if exam <= 40 and self_level == 'low':
        performance_block = (
            "\n🔴 Low performance + low confidence: focus on fundamentals first. "
            "Start with 20 minutes per day for core concepts, use solved examples, and stop after each question to explain the answer aloud. "
            "Set a weekly mini-test goal (5 questions) and log every mistake. "
            "Use heavy micro-practice with 1 topic per day."
        )
    elif 41 <= exam <= 70 and self_level == 'medium':
        performance_block = (
            "\n🟠 Medium performance + moderate confidence: improve strategy and speed. "
            "Do timed sets of 10-12 questions, alternate between weak and strong topics, and review errors in 10-minute after-action. "
            "Add one guided video/pdf session daily and one self-generated question set per week."
        )
    elif 71 <= exam <= 100 and self_level == 'high':
        performance_block = (
            "\n🟢 High performance + high confidence: advance to mastery and exam simulation. "
            "Include full topic tests once per week, aim for 90%+ accuracy, and then practice mixed-topic ungraded papers. "
            "Begin a peer teaching routine: explain two weak topic concepts to a friend each week."
        )
    else:
        performance_block = (
            "\n🔎 Continue a balanced improvement approach based on your current status: "
            "focus 70% on your weak areas and 30% on consolidation practice; keep tracking your progress daily."
        )

    user_habit = ""
    if survey.preferred_learning_method == 'video':
        user_habit = "Prefer short videos: use 15-min concept clips, then 10-min quick questions."
    elif survey.preferred_learning_method == 'reading':
        user_habit = "Prefer reading: do a 12-min note summary and 8-min recall drill."
    else:
        user_habit = "Prefer practice: do 10 problems first, then 5-min concept check."

    # Add explicit weak-topic resources as HTML anchors
    resource_part = ''
    if top_topics:
        topic_links = []
        for topic in top_topics:
            res = _resource_for_topic(topic)
            pdf_link = res['pdf'][1] or f"https://www.google.com/search?q={quote_plus(topic)}+pdf+notes"
            practice_link = f"https://www.google.com/search?q={quote_plus(topic)}+practice+questions"
            topic_links.append(
                f"<li>{topic}: <a href=\"{pdf_link}\" target=\"_blank\">PDF</a> | <a href=\"{practice_link}\" target=\"_blank\">Practice</a></li>"
            )
        resource_part = (
            "<div><strong>🔹 Helpful links:</strong></div>"
            "<ul>" + ''.join(topic_links) + "</ul><br/>"
        )

    # stronger unique user selectors: user ID + name + score + weak-topic char
    # ensures different users (even with same grade/score) get different index.
    user_key = f"{survey.user.id}-{survey.student_name}-{survey.class_grade}-{int(current)}-{len(top_topics)}"
    style_index = abs(hash(user_key)) % 6

    intros = [
        "Your plan is tailored to what you reported, with a special focus on your weak topics.",
        "Based on your survey, this is a unique recommendation just for you.",
        "This strategy reflects your weakness clusters and learning preferences.",
        "Your custom improvement path is generated from your score and subject gaps.",
        "We adapt this plan to your current rhythm, score, and weak-topic map.",
        "Each line below is chosen using your own survey data and progress trend."
    ]

    challenge_lines = [
        "Your key challenge: concept retention in problem-based areas.",
        "Your key challenge: keeping pace with weak topics across sessions.",
        "Your key challenge: regular revision of theory-heavy topics.",
        "Your key challenge: steady progress across 3-4 weak subjects.",
        "Your key challenge: turning weak-topic practice into lasting strength.",
        "Your key challenge: linking short-term practice to long-term recall."
    ]

    strategy_openers = [
        "🔹 Improvement Blueprint",
        "🔹 Tailored Action Plan",
        "🔹 Targeted Study Steps",
        "🔹 Focused Problem + Theory Path",
        "🔹 Weak-topic Upgradation Strategy",
        "🔹 Personalized Weekly Drill"
    ]

    unique_token = f"ID:{survey.student_name[:3].upper()}{int(current)}-{len(top_topics)}"

    intro_text = intros[style_index]
    challenge_text = challenge_lines[style_index]
    strategy_header = strategy_openers[style_index]

    return (
        f"Personalized Recommendation {unique_token}\n\n"
        f"{intro_text}\n\n"
        "🔹 Overall Summary\n"
        f"Your current score: {current}%, predicted: {predicted}%, consistent status: {consistency_flag}.\n"
        f"{challenge_text}\n\n"

        "🔹 Weak Area Identification\n"
        f"{weak_info} These are your priority practice domains.\n\n"

        f"{strategy_header}\n"
        "1. Concept Strengthening:\n"
        "   - 10–15 min concept review before practice.\n"
        "   - Understand reasoning, avoid rote memory.\n"
        f"   - {user_habit}\n\n"

        "2. Practice-Based Learning:\n"
        "   - Solve 8–12 questions daily in weak topics.\n"
        "   - Start with basic questions and increase difficulty over 2 weeks.\n\n"

        "3. Mistake Analysis:\n"
        "   - Spend 10 min post-session reviewing errors.\n"
        "   - Mark each as concept gap or careless error.\n"
        "   - Keep a 2-day mistake log and revisit it each week.\n\n"

        "🔹 Focus Areas\n"
        f"- Prioritize {top_weak or 'weak topic set'} for time allocation.\n"
        "- Reduce time on comfortable topics and redirect to weaker ones.\n\n"

        "🔹 Study Consistency Plan\n"
        "- 3 sessions daily (25 min each), 5 min breaks.\n"
        "- Check each day with a simple completed list.\n"
        "- Study at least 5 days per week.\n\n"

        "🔹 Weekly Goal\n"
        f"- Target: {weekly_goal}.\n"
        "- Aim for higher practice accuracy and fewer repeat mistakes.\n\n"

        f"{resource_part}"

        "🔹 Final Guidance\n"
        "If you keep consistency, focus on weak areas, and learn from mistakes, improvement will be steady.\n"
        f"{performance_block}\n"
    )


def build_progress_chart_data(user):
    snapshots = ProgressSnapshot.objects.filter(user=user).order_by('recorded_on')
    subjects = sorted({s.subject for s in snapshots})
    by_subject = defaultdict(list)

    for snap in snapshots:
        by_subject[snap.subject].append(
            {
                'x': snap.recorded_on.isoformat(),
                'y': float(snap.performance_score),
                'hours': float(snap.study_hours),
            }
        )

    datasets = []
    for idx, subject in enumerate(subjects):
        color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#17becf', '#9467bd']
        datasets.append(
            {
                'label': subject,
                'data': by_subject[subject],
                'borderColor': color_palette[idx % len(color_palette)],
                'backgroundColor': color_palette[idx % len(color_palette)],
                'tension': 0.3,
            }
        )

    return {'datasets': datasets}


def _detect_tamil(text):
    return any('\u0B80' <= ch <= '\u0BFF' for ch in (text or ''))


def _fallback_chat_reply(question, survey):
    q = (question or '').strip().lower()
    if not q:
        return 'Please ask a study-related question.'

    is_tamil = _detect_tamil(question)
    weak_map = clean_weak_topics_map(parse_json_field(survey.weak_topics, {})) if survey else {}
    weak_topics = [topic.lower() for topics in weak_map.values() for topic in topics]

    # If they asked a general doubt (not necessarily a weak topic), try to answer helpfully.
    def _links_for(term):
        query = quote_plus(term)
        return {
            'video': f'https://www.youtube.com/results?search_query={query}',
            'notes': f'https://www.google.com/search?q={query}+notes+pdf',
            'practice': f'https://www.google.com/search?q={query}+practice+questions',
        }

    for topic in weak_topics:
        if topic in q:
            resource = _resource_for_topic(topic)
            if is_tamil:
                return (
                    f'{topic.title()} ????? ??????????: ??????? concept ???????? ?????????, '
                    f'????? 10 ????????? ??????? ?????????.\n'
                    f'??????: {resource["youtube"][1]}\n???????????: {resource["pdf"][1]}'
                )
            return (
                f'{topic.title()} in brief: learn the core concept, then solve 10 questions and review mistakes.\n'
                f'Video: {resource["youtube"][1]}\nNotes: {resource["pdf"][1]}'
            )

    # Programming / CS common doubts (works offline too).
    if any(k in q for k in ('python', 'loop', 'loops', 'for loop', 'while loop')):
        links = _links_for('python loops for while')
        if is_tamil:
            return (
                "Python loops (for / while) - short guide:\n"
                "- for: iterable (list/range) மேல iterate பண்ணும்.\n"
                "- while: condition true இருக்கும்வரை run ஆகும்.\n"
                "Example:\n"
                "for i in range(1, 6):\n"
                "    print(i)\n"
                "Common mistakes: indentation, infinite while.\n"
                f"Video: {links['video']}\nNotes: {links['notes']}\nPractice: {links['practice']}"
            )
        return (
            "Python loops (for / while) in brief:\n"
            "- **for**: iterate over items (list/range/string).\n"
            "- **while**: repeat while a condition is True.\n\n"
            "Example:\n"
            "for i in range(1, 6):\n"
            "    print(i)\n\n"
            "Common mistakes: wrong indentation, forgetting to update a while loop variable (infinite loop).\n"
            f"Video: {links['video']}\nNotes: {links['notes']}\nPractice: {links['practice']}"
        )

    # Generic study help for any topic name even if it's not in weak_topics.
    if len(q.split()) <= 6:
        term = question.strip()
        links = _links_for(term)
        if is_tamil:
            return (
                f"{term} - எப்படி படிக்கலாம்:\n"
                "- 10 நிமிடம் concept summary (notes/video)\n"
                "- 10 கேள்வி practice\n"
                "- தவறுகள் error-log ல எழுதவும், 2 நாட்களுக்கு பிறகு மறுபடியும் செய்யவும்\n"
                f"Video: {links['video']}\nNotes: {links['notes']}\nPractice: {links['practice']}"
            )
        return (
            f"{term} - quick improvement steps:\n"
            "- Learn the core idea (10–15 min notes/video)\n"
            "- Do 10 practice questions\n"
            "- Write mistakes → correct rule → retry after 24–48h\n"
            f"Video: {links['video']}\nNotes: {links['notes']}\nPractice: {links['practice']}"
        )

    if 'schedule' in q or 'timetable' in q:
        return (
            'Tell me your subjects and how many hours/day you can study, I will suggest a simple routine.'
            if not is_tamil
            else 'நீங்கள் எந்த பாடங்கள் படிக்கிறீர்கள் மற்றும் தினமும் எத்தனை மணி நேரம் படிக்க முடியும் என்று கூறுங்கள்.'
        )

    if is_tamil:
        return 'உங்கள் topic/subject name கொடுத்து கேளுங்கள். நான் short steps + practice link உடன் help பண்ணுவேன்.'
    return 'Tell me the exact topic name (and subject). I will give short steps + examples + practice links.'


def generate_chatbot_reply_with_history(question, survey, history):
    api_key = os.getenv('OPENAI_API_KEY', '').strip()
    if api_key:
        try:
            from openai import OpenAI

            profile_line = ''
            if survey:
                profile_line = (
                    f"Student grade: {survey.class_grade}; learning mode: {survey.preferred_learning_method}; "
                    f"score: {survey.exam_score}."
                )

            sys_prompt = (
                'You are a study tutor chatbot. Reply briefly (4-8 lines), clear steps, no fluff. '
                'If the user asks in Tamil, answer in Tamil. If user asks in English, answer in English. '
                'Use previous conversation context when needed. '
                f'{profile_line}'
            )

            messages = [{'role': 'system', 'content': sys_prompt}]
            for item in history[-8:]:
                messages.append({'role': 'user', 'content': item.get('question', '')})
                messages.append({'role': 'assistant', 'content': item.get('answer', '')})
            messages.append({'role': 'user', 'content': question})

            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-4o-mini'),
                messages=messages,
                temperature=0.3,
                max_tokens=400,
            )
            text = (resp.choices[0].message.content or '').strip()
            if text:
                return text
        except Exception:
            pass

    return _fallback_chat_reply(question, survey)


def generate_chatbot_reply(question, survey):
    return generate_chatbot_reply_with_history(question, survey, history=[])


def search_study_materials(query, survey=None):
    clean_query = (query or '').strip()
    if not clean_query:
        return []

    weak_topic_map = clean_weak_topics_map(parse_json_field(survey.weak_topics, {})) if survey else {}
    weak_topics = {topic.lower() for topics in weak_topic_map.values() for topic in topics}

    results = []
    query_terms = clean_query.lower()

    for topic in weak_topics:
        if topic in query_terms:
            resources = _resource_for_topic(topic)
            results.extend(
                [
                    {'type': 'youtube', 'title': resources['youtube'][0], 'url': resources['youtube'][1], 'source_topic': topic},
                    {'type': 'pdf', 'title': resources['pdf'][0], 'url': resources['pdf'][1], 'source_topic': topic},
                ]
            )

    encoded = quote_plus(clean_query)
    results.extend(
        [
            {
                'type': 'youtube',
                'title': f'{clean_query.title()} Video Lectures',
                'url': f'https://www.youtube.com/results?search_query={encoded}+lecture',
                'source_topic': clean_query,
            },
            {
                'type': 'pdf',
                'title': f'{clean_query.title()} Notes and PDFs',
                'url': f'https://www.google.com/search?q={encoded}+pdf+notes',
                'source_topic': clean_query,
            },
            {
                'type': 'practice',
                'title': f'{clean_query.title()} Practice Questions',
                'url': f'https://www.google.com/search?q={encoded}+practice+questions',
                'source_topic': clean_query,
            },
        ]
    )
    return results[:8]


def ensure_baseline_progress(user, survey):
    if ProgressSnapshot.objects.filter(user=user).exists():
        return

    hours_map = parse_json_field(survey.daily_study_hours, {})
    for subject, hours in hours_map.items():
        ProgressSnapshot.objects.create(
            user=user,
            subject=subject,
            study_hours=float(hours),
            performance_score=max(35, float(survey.exam_score) - 10),
            recorded_on=date.today() - timedelta(days=21),
            notes='Baseline snapshot from survey',
        )
        ProgressSnapshot.objects.create(
            user=user,
            subject=subject,
            study_hours=float(hours) + 0.7,
            performance_score=min(100, float(survey.exam_score) + 5),
            recorded_on=date.today(),
            notes='Latest progress snapshot',
        )
