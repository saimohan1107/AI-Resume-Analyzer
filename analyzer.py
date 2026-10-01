import re
import time
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# SKILLS DATABASE
# =========================================================

SKILLS = [
    "python",
    "java",
    "c",
    "c++",
    "javascript",
    "typescript",
    "html",
    "css",
    "sql",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "data science",
    "data structures",
    "algorithms",
    "numpy",
    "pandas",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "git",
    "github",
    "streamlit",
    "nlp",
    "natural language processing",
    "flask",
    "django",
    "docker",
    "aws",
    "mongodb",
    "mysql",
    "power bi",
    "tableau",
    "excel",
    "matlab",
    "verilog",
    "vlsi",
    "computer vision",
    "opencv",
    "statistics",
    "linux",
    "rest api",
    "react",
    "node.js",
]


# =========================================================
# RESUME SECTIONS
# =========================================================

SECTION_PATTERNS = {
    "Summary / Objective": [
        r"\bsummary\b",
        r"\bobjective\b",
        r"\bprofile\b",
    ],
    "Skills": [
        r"\bskills\b",
        r"\btechnical skills\b",
        r"\bcore competencies\b",
    ],
    "Education": [
        r"\beducation\b",
        r"\bacademic background\b",
    ],
    "Experience": [
        r"\bexperience\b",
        r"\bwork experience\b",
        r"\bemployment\b",
        r"\bprofessional experience\b",
    ],
    "Projects": [
        r"\bprojects\b",
        r"\bpersonal projects\b",
        r"\bacademic projects\b",
    ],
    "Certifications": [
        r"\bcertifications?\b",
        r"\bcertificates?\b",
    ],
}


# =========================================================
# TEXT HELPERS
# =========================================================

def clean_text(text):
    """Clean text without destroying useful resume information."""

    if not text:
        return ""

    text = text.replace("\u2013", "-")
    text = text.replace("\u2014", "-")
    text = text.replace("\u2019", "'")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def contains_skill(text, skill):
    """Check whether a skill occurs in text."""

    if not text:
        return False

    text = text.lower()
    skill = skill.lower()

    # C needs special handling because C++ contains the letter C.
    if skill == "c":
        return bool(
            re.search(
                r"(?<![a-z0-9+#])c(?![a-z0-9+#])",
                text
            )
        )

    # C++ needs special handling because + is not a word character.
    if skill == "c++":
        return bool(
            re.search(
                r"(?<!\w)c\+\+(?!\w)",
                text
            )
        )

    # Allow spaces/hyphens between words.
    escaped = re.escape(skill)
    escaped = escaped.replace(r"\ ", r"\s+")

    pattern = r"(?<!\w)" + escaped + r"(?!\w)"

    return bool(re.search(pattern, text))


# =========================================================
# TF-IDF SIMILARITY
# =========================================================

def calculate_match_score(resume_text, job_description):
    """
    Calculate textual similarity between resume and job description.
    This is NOT a real ATS score.
    """

    resume_text = clean_text(resume_text)
    job_description = clean_text(job_description)

    if not resume_text or not job_description:
        return 0.0

    try:

        documents = [
            resume_text,
            job_description
        ]

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        matrix = vectorizer.fit_transform(documents)

        similarity = cosine_similarity(
            matrix[0:1],
            matrix[1:2]
        )[0][0]

        return round(similarity * 100, 2)

    except ValueError:
        return 0.0


# =========================================================
# SKILL ANALYSIS
# =========================================================

def analyze_skill_gap(resume_text, job_description):

    required_skills = []
    matching_skills = []
    missing_skills = []

    for skill in SKILLS:

        if contains_skill(job_description, skill):

            required_skills.append(skill)

            if contains_skill(resume_text, skill):
                matching_skills.append(skill)

            else:
                missing_skills.append(skill)

    return required_skills, matching_skills, missing_skills


def find_matching_skills(resume_text, job_description):

    _, matching_skills, missing_skills = analyze_skill_gap(
        resume_text,
        job_description
    )

    return matching_skills, missing_skills


def calculate_skill_match_percentage(
    required_skills,
    matching_skills
):

    if not required_skills:
        return 100.0

    percentage = (
        len(matching_skills)
        / len(required_skills)
    ) * 100

    return round(percentage, 2)


# =========================================================
# KEYWORD ANALYSIS
# =========================================================

def extract_keywords(job_description, top_n=20):

    if not job_description:
        return []

    try:

        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1
        )

        matrix = vectorizer.fit_transform(
            [job_description]
        )

        words = vectorizer.get_feature_names_out()

        counts = matrix.toarray()[0]

        pairs = list(
            zip(words, counts)
        )

        pairs.sort(
            key=lambda x: x[1],
            reverse=True
        )

        keywords = []

        for word, count in pairs:

            if len(word) < 3:
                continue

            if word not in keywords:
                keywords.append(word)

            if len(keywords) >= top_n:
                break

        return keywords

    except ValueError:

        return []


def find_keyword_gaps(
    resume_text,
    job_description,
    top_n=20
):

    keywords = extract_keywords(
        job_description,
        top_n
    )

    matching = []
    missing = []

    for keyword in keywords:

        if contains_skill(resume_text, keyword):
            matching.append(keyword)

        else:
            missing.append(keyword)

    return keywords, matching, missing


def calculate_keyword_match_percentage(
    resume_text,
    job_description
):

    keywords = extract_keywords(
        job_description,
        top_n=20
    )

    if not keywords:
        return 0.0

    matched = 0

    for keyword in keywords:

        if contains_skill(resume_text, keyword):
            matched += 1

    percentage = (
        matched / len(keywords)
    ) * 100

    return round(percentage, 2)


# =========================================================
# RESUME SECTION ANALYSIS
# =========================================================

def analyze_resume_sections(resume_text):

    text = resume_text.lower()

    results = {}

    for section, patterns in SECTION_PATTERNS.items():

        found = False

        for pattern in patterns:

            if re.search(pattern, text):

                found = True
                break

        results[section] = found

    # Contact information
    email_found = bool(
        re.search(
            r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
            text
        )
    )

    phone_found = bool(
        re.search(
            r"(\+91[\s-]?)?[6-9]\d{9}",
            text
        )
    )

    linkedin_found = "linkedin.com" in text
    github_found = "github.com" in text

    results["Email"] = email_found
    results["Phone"] = phone_found
    results["LinkedIn"] = linkedin_found
    results["GitHub"] = github_found

    return results


def calculate_resume_completeness(resume_text):

    sections = analyze_resume_sections(
        resume_text
    )

    important_sections = [
        "Summary / Objective",
        "Skills",
        "Education",
        "Experience",
        "Projects",
        "Certifications",
        "Email",
        "Phone",
        "LinkedIn",
        "GitHub",
    ]

    found = sum(
        1
        for section in important_sections
        if sections.get(section, False)
    )

    return round(
        (found / len(important_sections)) * 100,
        2
    )


# =========================================================
# EXPERIENCE / PROJECT CHECKS
# =========================================================

def calculate_experience_projects_score(resume_text):

    text = resume_text.lower()

    experience = bool(
        re.search(
            r"\b(experience|internship|employment|work experience)\b",
            text
        )
    )

    projects = bool(
        re.search(
            r"\b(projects|personal project|academic project)\b",
            text
        )
    )

    if experience and projects:
        return 100.0

    if experience or projects:
        return 70.0

    return 20.0


def calculate_education_score(resume_text):

    text = resume_text.lower()

    education_words = [
        "education",
        "b.tech",
        "btech",
        "bachelor",
        "degree",
        "university",
        "college",
        "school"
    ]

    found = any(
        word in text
        for word in education_words
    )

    return 100.0 if found else 0.0


# =========================================================
# ATS-STYLE HEURISTIC SCORE
# =========================================================

def calculate_ats_score(
    skill_match,
    keyword_match,
    completeness,
    experience_projects,
    education,
    tfidf_similarity
):

    score = (
        skill_match * 0.35
        + keyword_match * 0.20
        + completeness * 0.15
        + experience_projects * 0.10
        + education * 0.05
        + tfidf_similarity * 0.15
    )

    return round(
        min(score, 100),
        2
    )


# =========================================================
# SUGGESTIONS
# =========================================================

def generate_suggestions(
    resume_text,
    job_description,
    matching_skills,
    missing_skills,
    keyword_gaps=None
):

    suggestions = []

    resume_lower = resume_text.lower()

    if missing_skills:

        suggestions.append(
            "Consider adding relevant experience, coursework, "
            "or projects demonstrating: "
            + ", ".join(
                skill.title()
                for skill in missing_skills
            )
            + "."
        )

    if matching_skills:

        suggestions.append(
            "Make your strongest matching skills visible "
            "inside your Skills and Projects sections: "
            + ", ".join(
                skill.title()
                for skill in matching_skills
            )
            + "."
        )

    if keyword_gaps:

        suggestions.append(
            "Review the job-description keywords and naturally "
            "include relevant ones where they accurately describe "
            "your experience."
        )

    if "projects" not in resume_lower:

        suggestions.append(
            "Add a Projects section with 2–4 relevant technical projects."
        )

    if "github" not in resume_lower:

        suggestions.append(
            "Consider adding your GitHub profile and relevant repositories."
        )

    if "linkedin" not in resume_lower:

        suggestions.append(
            "Consider adding your LinkedIn profile."
        )

    if not re.search(
        r"\d+%|\d+\+|\b\d+\s*(users|projects|datasets|records)",
        resume_lower
    ):

        suggestions.append(
            "Where truthful, use measurable results such as "
            "accuracy, users, dataset size, performance improvement, "
            "or time saved."
        )

    if not re.search(
        r"\b(education|b\.?tech|btech|bachelor|degree)\b",
        resume_lower
    ):

        suggestions.append(
            "Make sure your Education section is clearly visible."
        )

    return suggestions


# =========================================================
# GEMINI MODEL CONFIGURATION
# =========================================================

# We try the newest model first.
# If it is temporarily unavailable, we automatically
# try the other Flash models.

GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
]


# =========================================================
# GEMINI HELPER
# =========================================================

def generate_gemini_response(prompt, api_key=None):
    """
    Send a prompt to Gemini with retry and model fallback.

    Handles temporary:
    - 503 UNAVAILABLE
    - 429 RESOURCE_EXHAUSTED

    Returns:
        Generated text

    Raises:
        RuntimeError if all models/retries fail.
    """

    from google import genai

    client = (
        genai.Client(api_key=api_key)
        if api_key
        else genai.Client()
    )

    last_error = None

    # Try each model
    for model in GEMINI_MODELS:

        # Try each model up to 3 times
        for attempt in range(3):

            try:

                response = client.models.generate_content(
                    model=model,
                    contents=prompt
                )

                text = getattr(response, "text", None)

                if text:
                    return text.strip()

                last_error = RuntimeError(
                    f"Gemini returned an empty response using {model}."
                )

            except Exception as e:

                last_error = e

                error_text = str(e).upper()

                # Temporary errors
                temporary_error = (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                )

                if temporary_error:

                    # Exponential backoff:
                    # 2 seconds
                    # 4 seconds
                    # 8 seconds
                    wait_time = 2 ** attempt

                    time.sleep(wait_time)

                    continue

                # Permanent errors such as invalid API key
                # should not be retried with the same model.
                break

    raise RuntimeError(
        "All Gemini models were temporarily unavailable. "
        f"Last error: {last_error}"
    )


# =========================================================
# AI REVIEW — GEMINI
# =========================================================

def generate_ai_review(
    resume_text,
    job_description,
    score,
    matching_skills,
    missing_skills,
    api_key=None
):

    try:

        prompt = f"""
You are an expert resume reviewer.

Analyze the resume against the job description.

IMPORTANT:
- Use ONLY information provided in the resume.
- Do NOT invent skills, companies, achievements, metrics,
  education, responsibilities, or experience.
- If something is missing, say that it is missing.
- Do not encourage the candidate to falsely add skills.
- Keep the review practical and concise.

RESUME:
{resume_text}

JOB DESCRIPTION:
{job_description}

CURRENT HEURISTIC MATCH SCORE:
{score}%

MATCHING SKILLS:
{", ".join(matching_skills) if matching_skills else "None identified"}

MISSING SKILLS:
{", ".join(missing_skills) if missing_skills else "None identified"}

Return the review using these headings:

## Executive Summary

## Strengths

## Skill Gaps

## ATS / Keyword Improvements

## Resume Improvements

## 3 Priority Actions
"""

        return generate_gemini_response(
            prompt,
            api_key
        )

    except Exception as e:

        return (
            "Gemini review could not be generated.\n\n"
            f"Error: {str(e)}"
        )


# =========================================================
# AI BULLET IMPROVER
# =========================================================

def improve_bullet_with_ai(
    bullet,
    job_description,
    api_key=None
):

    try:

        prompt = f"""
Rewrite this resume bullet to make it concise,
professional, achievement-oriented and ATS-friendly.

Do NOT invent:
- numbers
- percentages
- tools
- technologies
- responsibilities
- achievements
- results

Only improve the wording of information already present.

JOB DESCRIPTION:
{job_description}

ORIGINAL BULLET:
{bullet}

Return only the improved bullet.
"""

        response = generate_gemini_response(
            prompt,
            api_key
        )

        return response.strip()

    except Exception as e:

        return (
            "Could not improve bullet.\n\n"
            f"Error: {str(e)}"
        )


# =========================================================
# REPORT GENERATOR
# =========================================================

def build_report(analysis):

    lines = []

    lines.append("AI RESUME ANALYZER REPORT")
    lines.append("=" * 40)
    lines.append("")

    lines.append(
        f"ATS-style heuristic score: "
        f"{analysis['ats_score']}/100"
    )

    lines.append(
        f"TF-IDF similarity: "
        f"{analysis['tfidf_score']}%"
    )

    lines.append(
        f"Skills match: "
        f"{analysis['skill_match']}%"
    )

    lines.append(
        f"Keyword match: "
        f"{analysis['keyword_match']}%"
    )

    lines.append(
        f"Resume completeness: "
        f"{analysis['completeness']}%"
    )

    lines.append("")

    lines.append("REQUIRED SKILLS")
    lines.append("-" * 20)

    for skill in analysis["required_skills"]:
        lines.append(f"- {skill}")

    lines.append("")

    lines.append("MATCHING SKILLS")
    lines.append("-" * 20)

    for skill in analysis["matching_skills"]:
        lines.append(f"- {skill}")

    lines.append("")

    lines.append("MISSING SKILLS")
    lines.append("-" * 20)

    for skill in analysis["missing_skills"]:
        lines.append(f"- {skill}")

    lines.append("")

    lines.append("SUGGESTIONS")
    lines.append("-" * 20)

    for suggestion in analysis["suggestions"]:
        lines.append(f"- {suggestion}")

    return "\n".join(lines)