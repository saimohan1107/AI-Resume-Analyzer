import streamlit as st

from resume_parser import extract_text_from_pdf

from analyzer import (
    calculate_match_score,
    analyze_skill_gap,
    calculate_skill_match_percentage,
    calculate_keyword_match_percentage,
    find_keyword_gaps,
    analyze_resume_sections,
    calculate_resume_completeness,
    calculate_experience_projects_score,
    calculate_education_score,
    calculate_ats_score,
    generate_suggestions,
    generate_ai_review,
    improve_bullet_with_ai,
    build_report,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("📄 AI Resume Analyzer")

st.write(
    "Upload your resume and compare it with a job description "
    "using skill matching, keyword analysis, similarity analysis "
    "and optional Gemini AI review."
)

st.info(
    "The ATS score shown here is a heuristic estimate created by "
    "this project. It is not the score from a real company's ATS."
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("⚙️ Settings")

    st.write(
        "Upload a PDF resume and paste the target job description."
    )

    st.divider()

    st.subheader("Features")

    st.write("✅ Resume PDF extraction")
    st.write("✅ Skill gap analysis")
    st.write("✅ Keyword analysis")
    st.write("✅ TF-IDF similarity")
    st.write("✅ ATS-style heuristic score")
    st.write("✅ Resume section analysis")
    st.write("✅ Improvement suggestions")
    st.write("✅ Gemini AI review")
    st.write("✅ AI bullet rewriting")


# =========================================================
# INPUTS
# =========================================================

st.header("1️⃣ Upload Resume")

resume_file = st.file_uploader(
    "Choose your resume PDF",
    type=["pdf"]
)


st.header("2️⃣ Job Description")

job_description = st.text_area(
    "Paste the complete job description here",
    height=250,
    placeholder="Paste the job description..."
)


# =========================================================
# RESUME EXTRACTION
# =========================================================

resume_text = ""

if resume_file:

    try:

        resume_text = extract_text_from_pdf(
            resume_file
        )

        if resume_text.strip():

            st.success(
                "✅ Resume processed successfully!"
            )

            with st.expander(
                "👀 View extracted resume text"
            ):

                st.text_area(
                    "Extracted text",
                    resume_text,
                    height=300
                )

        else:

            st.error(
                "Could not extract text from this PDF."
            )

    except Exception as e:

        st.error(
            f"Error reading resume: {e}"
        )


# =========================================================
# ANALYZE BUTTON
# =========================================================

ready = bool(
    resume_text.strip()
    and job_description.strip()
)


st.header("3️⃣ Analyze")

analyze_button = st.button(
    "🔍 Analyze Resume",
    type="primary",
    disabled=not ready,
    use_container_width=True
)


if not ready:

    st.caption(
        "Upload a PDF resume and paste a job description "
        "to enable analysis."
    )


# =========================================================
# RUN ANALYSIS
# =========================================================

if analyze_button:

    with st.spinner(
        "Analyzing your resume..."
    ):

        # ---------------------------------------------
        # Similarity
        # ---------------------------------------------

        tfidf_score = calculate_match_score(
            resume_text,
            job_description
        )

        # ---------------------------------------------
        # Skills
        # ---------------------------------------------

        (
            required_skills,
            matching_skills,
            missing_skills
        ) = analyze_skill_gap(
            resume_text,
            job_description
        )

        skill_match = calculate_skill_match_percentage(
            required_skills,
            matching_skills
        )

        # ---------------------------------------------
        # Keywords
        # ---------------------------------------------

        (
            keywords,
            matching_keywords,
            missing_keywords
        ) = find_keyword_gaps(
            resume_text,
            job_description
        )

        keyword_match = calculate_keyword_match_percentage(
            resume_text,
            job_description
        )

        # ---------------------------------------------
        # Resume completeness
        # ---------------------------------------------

        completeness = calculate_resume_completeness(
            resume_text
        )

        # ---------------------------------------------
        # Experience / projects
        # ---------------------------------------------

        experience_projects = (
            calculate_experience_projects_score(
                resume_text
            )
        )

        # ---------------------------------------------
        # Education
        # ---------------------------------------------

        education = calculate_education_score(
            resume_text
        )

        # ---------------------------------------------
        # Overall ATS-style score
        # ---------------------------------------------

        ats_score = calculate_ats_score(
            skill_match,
            keyword_match,
            completeness,
            experience_projects,
            education,
            tfidf_score
        )

        # ---------------------------------------------
        # Sections
        # ---------------------------------------------

        sections = analyze_resume_sections(
            resume_text
        )

        # ---------------------------------------------
        # Suggestions
        # ---------------------------------------------

        suggestions = generate_suggestions(
            resume_text,
            job_description,
            matching_skills,
            missing_skills,
            missing_keywords
        )

        # ---------------------------------------------
        # Save results
        # ---------------------------------------------

        st.session_state.analysis = {

            "ats_score": ats_score,

            "tfidf_score": tfidf_score,

            "skill_match": skill_match,

            "keyword_match": keyword_match,

            "completeness": completeness,

            "experience_projects": experience_projects,

            "education": education,

            "required_skills": required_skills,

            "matching_skills": matching_skills,

            "missing_skills": missing_skills,

            "keywords": keywords,

            "matching_keywords": matching_keywords,

            "missing_keywords": missing_keywords,

            "sections": sections,

            "suggestions": suggestions,
        }

        st.session_state.ai_review = None

    st.success(
        "Analysis completed successfully!"
    )


# =========================================================
# DISPLAY RESULTS
# =========================================================

analysis = st.session_state.get(
    "analysis"
)


if analysis:

    st.divider()

    st.header("📊 Resume Analysis")


    # =====================================================
    # MAIN SCORE
    # =====================================================

    st.subheader("Overall Score")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "ATS-style Score",
            f"{analysis['ats_score']}/100"
        )

    with col2:

        st.metric(
            "TF-IDF Similarity",
            f"{analysis['tfidf_score']}%"
        )

    with col3:

        st.metric(
            "Skill Match",
            f"{analysis['skill_match']}%"
        )


    # =====================================================
    # SCORE BREAKDOWN
    # =====================================================

    st.subheader("📈 Score Breakdown")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Skills Match",
            f"{analysis['skill_match']}%"
        )

        st.metric(
            "Keyword Match",
            f"{analysis['keyword_match']}%"
        )

    with col2:

        st.metric(
            "Resume Completeness",
            f"{analysis['completeness']}%"
        )

        st.metric(
            "Experience / Projects",
            f"{analysis['experience_projects']}%"
        )

    with col3:

        st.metric(
            "Education",
            f"{analysis['education']}%"
        )

        st.metric(
            "TF-IDF Similarity",
            f"{analysis['tfidf_score']}%"
        )


    # =====================================================
    # SKILLS
    # =====================================================

    st.divider()

    st.header("🛠️ Skill Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.subheader("Required Skills")

        if analysis["required_skills"]:

            for skill in analysis["required_skills"]:

                st.write(
                    f"🔹 {skill.title()}"
                )

        else:

            st.write(
                "No tracked skills identified."
            )


    with col2:

        st.subheader("Matching Skills")

        if analysis["matching_skills"]:

            for skill in analysis["matching_skills"]:

                st.write(
                    f"✅ {skill.title()}"
                )

        else:

            st.write(
                "No matching tracked skills found."
            )


    with col3:

        st.subheader("Missing Skills")

        if analysis["missing_skills"]:

            for skill in analysis["missing_skills"]:

                st.write(
                    f"❌ {skill.title()}"
                )

        else:

            st.write(
                "No missing tracked skills."
            )


    # =====================================================
    # KEYWORDS
    # =====================================================

    st.divider()

    st.header("🔑 Keyword Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Keywords Found")

        if analysis["matching_keywords"]:

            for keyword in analysis["matching_keywords"]:

                st.write(
                    f"✅ {keyword}"
                )

        else:

            st.write(
                "No matching keywords identified."
            )


    with col2:

        st.subheader("Keywords Missing")

        if analysis["missing_keywords"]:

            for keyword in analysis["missing_keywords"]:

                st.write(
                    f"❌ {keyword}"
                )

        else:

            st.write(
                "No major keyword gaps identified."
            )


    # =====================================================
    # RESUME SECTIONS
    # =====================================================

    st.divider()

    st.header("📑 Resume Sections")

    section_cols = st.columns(2)

    section_items = list(
        analysis["sections"].items()
    )

    for index, (
        section,
        present
    ) in enumerate(section_items):

        with section_cols[index % 2]:

            if present:

                st.write(
                    f"✅ {section}"
                )

            else:

                st.write(
                    f"❌ {section}"
                )


    # =====================================================
    # SUGGESTIONS
    # =====================================================

    st.divider()

    st.header("💡 Resume Improvement Suggestions")

    for suggestion in analysis["suggestions"]:

        st.info(
            suggestion
        )


    # =====================================================
    # GEMINI AI REVIEW
    # =====================================================

    st.divider()

    st.header("🤖 Gemini AI Review")

    try:

        api_key = st.secrets.get(
            "GEMINI_API_KEY",
            ""
        )

    except Exception:

        api_key = ""


    if not api_key:

        st.warning(
            "Gemini API key was not found. "
            "Check .streamlit/secrets.toml."
        )

    else:

        if st.button(
            "🤖 Generate AI Review",
            use_container_width=True
        ):

            with st.spinner(
                "Gemini is reviewing your resume..."
            ):

                review = generate_ai_review(
                    resume_text,
                    job_description,
                    analysis["ats_score"],
                    analysis["matching_skills"],
                    analysis["missing_skills"],
                    api_key
                )

                st.session_state.ai_review = review


        if st.session_state.get(
            "ai_review"
        ):

            st.markdown(
                st.session_state.ai_review
            )


    # =====================================================
    # AI BULLET IMPROVER
    # =====================================================

    st.divider()

    st.header("✨ AI Resume Bullet Improver")

    bullet = st.text_area(
        "Paste one resume bullet",
        placeholder=(
            "Example: Developed a Python project "
            "for analyzing student data."
        ),
        height=120
    )


    if st.button(
        "✨ Improve This Bullet",
        use_container_width=True
    ):

        if not bullet.strip():

            st.warning(
                "Please enter a resume bullet first."
            )

        elif not api_key:

            st.warning(
                "Gemini API key was not found."
            )

        else:

            with st.spinner(
                "Improving your bullet..."
            ):

                improved = improve_bullet_with_ai(
                    bullet,
                    job_description,
                    api_key
                )

            st.subheader(
                "Improved Bullet"
            )

            st.success(
                improved
            )


    # =====================================================
    # DOWNLOAD REPORT
    # =====================================================

    st.divider()

    st.header("📥 Download Report")

    report = build_report(
        analysis
    )

    if st.session_state.get(
        "ai_review"
    ):

        report += (
            "\n\n"
            + "=" * 40
            + "\nGEMINI AI REVIEW\n"
            + "=" * 40
            + "\n\n"
            + st.session_state.ai_review
        )

    st.download_button(
        "📥 Download Analysis Report",
        data=report,
        file_name="resume_analysis_report.txt",
        mime="text/plain",
        use_container_width=True
    )