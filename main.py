import os
import json
import tempfile
from typing import Dict, List, Optional, Tuple
import requests
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib import colors
from reportlab.lib.units import inch
from dotenv import load_dotenv
from datetime import datetime
import base64
import plotly.graph_objects as go
from fpdf import FPDF
import pdfkit

# Load environment variables
load_dotenv()

# Constants
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DEFAULT_MODEL = "qwen/qwen3-30b-a3b:free"
MAX_TOKENS = 4000
TEMPERATURE = 0.7
THEMES = {
    "Professional Blue": {"primary": "#2B579A", "secondary": "#6B8CD9", "text": "#333333"},
    "Modern Green": {"primary": "#2E7D32", "secondary": "#66BB6A", "text": "#37474F"},
    "Creative Orange": {"primary": "#F57C00", "secondary": "#FFA726", "text": "#5D4037"},
    "Elegant Purple": {"primary": "#7B1FA2", "secondary": "#BA68C8", "text": "#4A148C"},
    "Minimal Black": {"primary": "#212121", "secondary": "#616161", "text": "#000000"}
}

# Utility function to call OpenRouter API
def call_openrouter(prompt: str, system_prompt: Optional[str] = None, model: str = DEFAULT_MODEL) -> str:
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://cv-builder.streamlit.app",
        "X-Title": "Professional CV Builder"
    }
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.error(f"OpenRouter API Error: {str(e)}")
        return ""

# Replace all LangChain-based functions with direct OpenRouter calls
def validate_required_fields(fields: dict, section: str) -> bool:
    """Check if all required fields are filled. Show error if not."""
    missing = [k for k, v in fields.items() if not v]
    if missing:
        st.error(f"Please fill in the following required fields in {section}: {', '.join(missing)}")
        return False
    return True

def call_langchain_chain(prompt_template: str, input_variables: Dict, memory_key: Optional[str] = None) -> str:
    # Compose the prompt using the template and variables
    prompt = prompt_template
    for k, v in input_variables.items():
        prompt = prompt.replace("{" + k + "}", str(v))
    return call_openrouter(prompt)

def generate_professional_summary(role: str, experience: str, skills: List[str]) -> str:
    prompt_template = """
    You are an expert CV writer with 20 years of experience helping professionals stand out.
    Create a compelling professional summary for a {role} with {experience} of experience.

    Skills: {skills}

    Guidelines:
    1. Start with a powerful opening line that grabs attention
    2. Highlight 2-3 key achievements or strengths
    3. Showcase relevant skills and expertise
    4. Keep it concise (3-4 sentences max)
    5. Use action verbs and quantifiable results where possible
    6. Tailor it to {role} position

    Return only the summary text.
    """
    return call_langchain_chain(
        prompt_template,
        {
            "role": role,
            "experience": experience,
            "skills": ", ".join(skills) if skills else "relevant skills"
        }
    )

def improve_content(content: str, content_type: str) -> str:
    prompt_template = """
    You are a professional editor specializing in CV optimization. Improve the following {content_type}:

    Original Content:
    {content}

    Guidelines:
    1. Make it more professional and impactful
    2. Use action verbs (e.g., "led", "developed", "optimized")
    3. Quantify achievements where possible
    4. Remove unnecessary words
    5. Ensure proper grammar and syntax
    6. Keep the original meaning

    Return only the improved version.
    """
    return call_langchain_chain(
        prompt_template,
        {
            "content_type": content_type,
            "content": content
        }
    )

def suggest_skills(job_description: str) -> List[str]:
    prompt_template = """
    Analyze this job description and extract the key skills required:
    {job_description}

    Return a list of 10-15 relevant skills in this exact format:
    - Skill 1
    - Skill 2
    - Skill 3
    """
    result = call_langchain_chain(
        prompt_template,
        {"job_description": job_description}
    )
    return [skill.strip('-• ').strip() for skill in result.split('\n') if skill.strip()]

def analyze_job_fit(job_description: str, cv_data: Dict) -> str:
    prompt_template = """
    Analyze how well this CV matches the job description:

    Job Description:
    {job_description}

    CV Data:
    {cv_data}

    Provide a detailed analysis with:
    1. Strengths (3-5 points)
    2. Weaknesses or gaps (3-5 points)
    3. Specific recommendations to improve fit
    4. Match percentage estimate

    Format your response with clear headings.
    """
    return call_langchain_chain(
        prompt_template,
        {
            "job_description": job_description,
            "cv_data": json.dumps(cv_data)
        }
    )

def generate_cover_letter(job_description: str, company: str) -> str:
    prompt_template = """
    Write a compelling cover letter for this job:

    Job Description:
    {job_description}

    Company: {company}

    CV Data:
    {cv_data}

    Guidelines:
    1. Address it to "Hiring Manager" (unless name is known)
    2. First paragraph: Express interest and highlight 1-2 most relevant qualifications
    3. Second paragraph: Showcase key achievements matching job requirements
    4. Third paragraph: Explain why you're a great culture fit
    5. Closing: Call to action and thank you
    6. Keep it to 3-4 paragraphs max
    7. Professional but enthusiastic tone
    """
    return call_langchain_chain(
        prompt_template,
        {
            "job_description": job_description,
            "company": company,
            "cv_data": json.dumps(st.session_state.cv_data)
        }
    )

def generate_achievement_bullets(role: str, responsibilities: str) -> str:
    prompt_template = """
    Transform these job responsibilities into impressive achievement statements:

    Role: {role}
    Responsibilities: {responsibilities}

    Guidelines:
    1. Start with action verbs (e.g., "Led", "Developed", "Increased")
    2. Quantify results where possible (e.g., "by 30%", "for 50K users")
    3. Focus on impact and outcomes
    4. Keep each bullet point to 1-2 lines
    5. Return 3-5 bullet points

    Format as:
    - Achievement 1
    - Achievement 2
    """
    return call_langchain_chain(
        prompt_template,
        {
            "role": role,
            "responsibilities": responsibilities
        }
    )

# Form sections with enhanced UI
def personal_information_section():
    """Render personal information form section with enhanced UI."""
    st.subheader("👤 Personal Information")
    
    with st.expander("Basic Information", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.cv_data['personal_info']['full_name'] = st.text_input(
                "Full Name*", 
                value=st.session_state.cv_data['personal_info'].get('full_name', ''),
                placeholder="John Doe"
            )
            st.session_state.cv_data['personal_info']['email'] = st.text_input(
                "Email*", 
                value=st.session_state.cv_data['personal_info'].get('email', ''),
                placeholder="john.doe@example.com"
            )
            st.session_state.cv_data['personal_info']['phone'] = st.text_input(
                "Phone Number", 
                value=st.session_state.cv_data['personal_info'].get('phone', ''),
                placeholder="+1 (123) 456-7890"
            )
        
        with col2:
            st.session_state.cv_data['personal_info']['linkedin'] = st.text_input(
                "LinkedIn Profile", 
                value=st.session_state.cv_data['personal_info'].get('linkedin', ''),
                placeholder="https://linkedin.com/in/yourprofile"
            )
            st.session_state.cv_data['personal_info']['portfolio'] = st.text_input(
                "Portfolio Website", 
                value=st.session_state.cv_data['personal_info'].get('portfolio', ''),
                placeholder="https://yourportfolio.com"
            )
            st.session_state.cv_data['personal_info']['location'] = st.text_input(
                "Location", 
                value=st.session_state.cv_data['personal_info'].get('location', ''),
                placeholder="City, Country"
            )
    
    with st.expander("Career Objectives"):
        st.session_state.cv_data['personal_info']['target_role'] = st.text_input(
            "Target Role/Title*", 
            value=st.session_state.cv_data['personal_info'].get('target_role', ''),
            placeholder="e.g., Senior Software Engineer"
        )
        st.session_state.cv_data['personal_info']['career_objective'] = st.text_area(
            "Career Objective (Optional)",
            value=st.session_state.cv_data['personal_info'].get('career_objective', ''),
            placeholder="Brief statement about your career goals",
            height=100
        )
    
    with st.expander("Photo (Optional)"):
        st.session_state.cv_data['settings']['show_photo'] = st.checkbox(
            "Include photo in CV",
            value=st.session_state.cv_data['settings']['show_photo']
        )
        uploaded_file = st.file_uploader(
            "Upload professional photo", 
            type=["jpg", "jpeg", "png"],
            help="Use a high-quality, professional headshot"
        )
        
        if uploaded_file is not None:
            st.session_state.cv_data['settings']['photo'] = {
                "name": uploaded_file.name,
                "data": base64.b64encode(uploaded_file.read()).decode("utf-8")
            }
            st.image(uploaded_file, width=150)
    
    # Validation
    required_fields = {
        "Full Name": st.session_state.cv_data['personal_info'].get('full_name', ''),
        "Email": st.session_state.cv_data['personal_info'].get('email', ''),
        "Target Role": st.session_state.cv_data['personal_info'].get('target_role', '')
    }
    
    if st.button("💾 Save Personal Information", use_container_width=True):
        if validate_required_fields(required_fields, "Personal Information"):
            st.success("Personal information saved successfully!")
            st.balloons()

def professional_summary_section():
    """Render professional summary form section with enhanced UI."""
    st.subheader("📝 Professional Summary")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.session_state.cv_data['professional_summary'] = st.text_area(
            "Write your professional summary (3-5 sentences)", 
            value=st.session_state.cv_data['professional_summary'],
            height=200,
            placeholder="Results-driven professional with X years of experience in...",
            help="Highlight your key qualifications, achievements, and what makes you unique"
        )
    
    with col2:
        st.markdown("### AI Assistant")
        with st.expander("Generate Options"):
            if st.button("✨ Generate Summary", help="AI will create a professional summary based on your profile"):
                if not st.session_state.cv_data['personal_info'].get('target_role'):
                    st.error("Please fill in your target role in Personal Information first.")
                else:
                    with st.spinner("Generating professional summary..."):
                        skills = st.session_state.cv_data['skills']
                        summary = generate_professional_summary(
                            st.session_state.cv_data['personal_info']['target_role'],
                            "experience",  # Could enhance with actual experience calculation
                            skills if skills else ["relevant skills"]
                        )
                        st.session_state.cv_data['professional_summary'] = summary
                        st.rerun()
            
            if st.button("🔧 Improve Summary", help="AI will enhance your existing summary"):
                if not st.session_state.cv_data['professional_summary']:
                    st.error("Please write a summary first.")
                else:
                    with st.spinner("Improving summary..."):
                        improved = improve_content(
                            st.session_state.cv_data['professional_summary'],
                            "professional summary"
                        )
                        st.session_state.cv_data['professional_summary'] = improved
                        st.rerun()
            
            if st.button("📊 Quantify Achievements", help="Add metrics and numbers to strengthen your summary"):
                if not st.session_state.cv_data['professional_summary']:
                    st.error("Please write a summary first.")
                else:
                    with st.spinner("Adding quantifiable achievements..."):
                        prompt = f"Add quantifiable achievements to this professional summary: {st.session_state.cv_data['professional_summary']}. Return only the improved version with added metrics."
                        quantified = call_langchain_chain(
                            "Add quantifiable achievements to this text: {text}. Return only the improved version.",
                            {"text": st.session_state.cv_data['professional_summary']}
                        )
                        st.session_state.cv_data['professional_summary'] = quantified
                        st.rerun()
    
    if st.button("💾 Save Professional Summary", use_container_width=True):
        if st.session_state.cv_data['professional_summary']:
            st.success("Professional summary saved successfully!")
        else:
            st.warning("Professional summary is empty.")

def work_experience_section():
    """Render work experience form section with enhanced UI."""
    st.subheader("💼 Work Experience")
    
    # Add new experience
    with st.expander("➕ Add New Work Experience", expanded=True):
        with st.form("work_experience_form"):
            col1, col2 = st.columns(2)
            with col1:
                job_title = st.text_input("Job Title*", placeholder="e.g., Software Engineer")
                company = st.text_input("Company Name*", placeholder="e.g., Tech Corp Inc.")
                location = st.text_input("Location", placeholder="e.g., San Francisco, CA")
            
            with col2:
                start_date = st.date_input("Start Date*")
                end_date = st.date_input("End Date*", key="work_end_date")
                current_job = st.checkbox("I currently work here")
            
            description = st.text_area(
                "Responsibilities & Achievements*", 
                height=150,
                placeholder="Describe your key responsibilities and achievements...",
                help="Focus on achievements rather than just duties. Use action verbs and quantify results."
            )
            
            if st.form_submit_button("Add Experience"):
                required_fields = {
                    "Job Title": job_title,
                    "Company Name": company,
                    "Start Date": start_date,
                    "Description": description
                }
                
                if validate_required_fields(required_fields, "Work Experience"):
                    experience = {
                        "job_title": job_title,
                        "company": company,
                        "location": location,
                        "start_date": str(start_date),
                        "end_date": "Present" if current_job else str(end_date),
                        "description": description
                    }
                    st.session_state.cv_data['work_experience'].append(experience)
                    st.success("Work experience added successfully!")
    
    # List existing experiences with enhanced edit/delete options
    st.write("### Your Work Experiences")
    
    if not st.session_state.cv_data['work_experience']:
        st.info("No work experiences added yet. Add your first experience above.")
    else:
        for i, exp in enumerate(st.session_state.cv_data['work_experience']):
            with st.expander(f"🔹 {exp['job_title']} at {exp['company']}", expanded=False):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    **Company:** {exp['company']}  
                    **Position:** {exp['job_title']}  
                    **Duration:** {exp['start_date']} to {exp['end_date']}  
                    **Location:** {exp.get('location', 'N/A')}
                    """)
                    st.markdown(f"**Description:**  \n{exp['description']}")
                
                with col2:
                    if st.button(f"🗑️ Delete", key=f"del_exp_{i}"):
                        st.session_state.cv_data['work_experience'].pop(i)
                        st.rerun()
                    
                    if st.button(f"✨ Enhance", key=f"enhance_exp_{i}"):
                        with st.spinner("Enhancing description..."):
                            enhanced = generate_achievement_bullets(
                                exp['job_title'],
                                exp['description']
                            )
                            st.session_state.cv_data['work_experience'][i]['description'] = enhanced
                            st.rerun()
                    
                    if st.button(f"🔧 Improve", key=f"improve_exp_{i}"):
                        with st.spinner("Improving description..."):
                            improved = improve_content(exp['description'], "job description")
                            st.session_state.cv_data['work_experience'][i]['description'] = improved
                            st.rerun()

def education_section():
    """Render education form section with enhanced UI."""
    st.subheader("🎓 Education")
    
    # Add new education
    with st.expander("➕ Add New Education", expanded=True):
        with st.form("education_form"):
            col1, col2 = st.columns(2)
            with col1:
                institution = st.text_input("Institution*", placeholder="e.g., Stanford University")
                degree = st.text_input("Degree/Certificate*", placeholder="e.g., B.S. in Computer Science")
            
            with col2:
                start_date = st.date_input("Start Date")
                end_date = st.date_input("End Date", key="edu_end_date")
                current_student = st.checkbox("Currently enrolled")
            
            description = st.text_area(
                "Details (e.g., honors, relevant coursework)", 
                height=100,
                placeholder="GPA: 3.8, Dean's List, Relevant Coursework: Data Structures, Algorithms..."
            )
            
            if st.form_submit_button("Add Education"):
                required_fields = {
                    "Institution": institution,
                    "Degree/Certificate": degree
                }
                
                if validate_required_fields(required_fields, "Education"):
                    education = {
                        "institution": institution,
                        "degree": degree,
                        "start_date": str(start_date) if start_date else "",
                        "end_date": "Present" if current_student else (str(end_date) if end_date else ""),
                        "description": description
                    }
                    st.session_state.cv_data['education'].append(education)
                    st.success("Education added successfully!")
    
    # List existing education with enhanced options
    st.write("### Your Education")
    
    if not st.session_state.cv_data['education']:
        st.info("No education entries added yet. Add your first education above.")
    else:
        for i, edu in enumerate(st.session_state.cv_data['education']):
            with st.expander(f"🔹 {edu['degree']} at {edu['institution']}", expanded=False):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    **Institution:** {edu['institution']}  
                    **Degree:** {edu['degree']}  
                    **Duration:** {edu['start_date']} to {edu['end_date']}
                    """)
                    if edu['description']:
                        st.markdown(f"**Details:**  \n{edu['description']}")
                
                with col2:
                    if st.button(f"🗑️ Delete", key=f"del_edu_{i}"):
                        st.session_state.cv_data['education'].pop(i)
                        st.rerun()
                    
                    if st.button(f"🔧 Improve", key=f"improve_edu_{i}") and edu['description']:
                        with st.spinner("Improving description..."):
                            improved = improve_content(edu['description'], "education description")
                            st.session_state.cv_data['education'][i]['description'] = improved
                            st.rerun()

def skills_section():
    """Render skills form section with enhanced UI."""
    st.subheader("🛠️ Skills")
    
    # Main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Skill categories
        with st.expander("➕ Add Skills by Category", expanded=True):
            skill_categories = {
                "Technical": ["Python", "JavaScript", "SQL", "Machine Learning"],
                "Soft": ["Leadership", "Communication", "Teamwork"],
                "Tools": ["Git", "Docker", "AWS"],
                "Languages": ["English", "Spanish", "French"]
            }
            
            selected_category = st.selectbox(
                "Skill Category",
                list(skill_categories.keys())
            )
            
            selected_skills = st.multiselect(
                f"Select {selected_category} Skills",
                skill_categories[selected_category],
                help="Select from popular skills or add your own below"
            )
            
            custom_skill = st.text_input(
                f"Add Custom {selected_category} Skill",
                placeholder="e.g., TensorFlow"
            )
            
            if st.button(f"Add {selected_category} Skills"):
                added = False
                for skill in selected_skills:
                    if skill not in st.session_state.cv_data['skills']:
                        st.session_state.cv_data['skills'].append(skill)
                        added = True
                if custom_skill and custom_skill not in st.session_state.cv_data['skills']:
                    st.session_state.cv_data['skills'].append(custom_skill)
                    added = True
                if added:
                    st.success(f"Added {len(selected_skills) + (1 if custom_skill else 0)} skills!")
                else:
                    st.warning("No new skills added (they may already be in your list)")
    
    with col2:
        # AI skill suggestion
        with st.expander("🤖 AI Skill Suggestions"):
            job_desc = st.text_area(
                "Paste a job description", 
                height=100,
                placeholder="Paste job description to get skill suggestions...",
                key="job_desc_skills"
            )
            if st.button("Analyze & Suggest Skills"):
                if job_desc:
                    with st.spinner("Analyzing job description..."):
                        suggested_skills = suggest_skills(job_desc)
                        st.session_state.ai_suggestions['skills'] = suggested_skills
                        st.success(f"Found {len(suggested_skills)} relevant skills!")
                else:
                    st.error("Please paste a job description first")
    
    # Display suggested skills if available
    if 'skills' in st.session_state.ai_suggestions:
        st.subheader("✨ Suggested Skills")
        st.write("These skills were identified from the job description:")
        
        cols = st.columns(3)
        for i, skill in enumerate(st.session_state.ai_suggestions['skills']):
            if skill not in st.session_state.cv_data['skills']:
                with cols[i % 3]:
                    if st.button(
                        f"➕ {skill}",
                        key=f"add_suggested_{i}",
                        help=f"Add {skill} to your skills list"
                    ):
                        st.session_state.cv_data['skills'].append(skill)
                        st.rerun()
    
    # Current skills visualization
    st.subheader("Your Skills Dashboard")
    
    if not st.session_state.cv_data['skills']:
        st.info("No skills added yet. Add skills using the options above.")
    else:
        # Skill proficiency levels
        st.write("### Skill Proficiency Levels")
        skill_proficiencies = {}
        
        cols = st.columns(3)
        for i, skill in enumerate(st.session_state.cv_data['skills']):
            with cols[i % 3]:
                proficiency = st.select_slider(
                    f"Proficiency for {skill}",
                    options=["Beginner", "Intermediate", "Advanced", "Expert"],
                    key=f"proficiency_{skill}"
                )
                skill_proficiencies[skill] = proficiency
        
        # Visualize skill levels
        st.write("### Skill Visualization")
        tab1, tab2 = st.tabs(["Radar Chart", "Bar Chart"])
        
        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=[4 if level == "Expert" else 3 if level == "Advanced" else 2 if level == "Intermediate" else 1 
                   for level in skill_proficiencies.values()],
                theta=list(skill_proficiencies.keys()),
                fill='toself',
                name="Skill Proficiency"
            ))
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 4]
                    )),
                showlegend=False,
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=list(skill_proficiencies.keys()),
                y=[4 if level == "Expert" else 3 if level == "Advanced" else 2 if level == "Intermediate" else 1 
                   for level in skill_proficiencies.values()],
                marker_color=THEMES[st.session_state.cv_data['settings']['theme']]['primary']
            ))
            fig.update_layout(
                yaxis=dict(range=[0, 4]),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Skill cloud
        st.write("### Skill Word Cloud")
        st.image(f"https://quickchart.io/wordcloud?text={','.join(st.session_state.cv_data['skills'])}&width=800&height=400", use_column_width=True)

def projects_section():
    """Render projects form section with enhanced UI."""
    st.subheader("📂 Projects")
    
    # Add new project
    with st.expander("➕ Add New Project", expanded=True):
        with st.form("project_form"):
            name = st.text_input("Project Name*", placeholder="e.g., E-commerce Website")
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input("Start Date")
                end_date = st.date_input("End Date", key="project_end_date")
                ongoing = st.checkbox("Ongoing Project")
            
            with col2:
                url = st.text_input("Project URL", placeholder="https://github.com/yourproject")
                tech_stack = st.text_input("Technologies Used", placeholder="Python, React, MongoDB")
            
            description = st.text_area(
                "Description*", 
                height=150,
                placeholder="Describe the project, your role, technologies used, and outcomes..."
            )
            
            if st.form_submit_button("Add Project"):
                required_fields = {
                    "Project Name": name,
                    "Description": description
                }
                
                if validate_required_fields(required_fields, "Projects"):
                    project = {
                        "name": name,
                        "description": description,
                        "url": url,
                        "tech_stack": tech_stack,
                        "start_date": str(start_date) if start_date else "",
                        "end_date": "Present" if ongoing else (str(end_date) if end_date else "")
                    }
                    st.session_state.cv_data['projects'].append(project)
                    st.success("Project added successfully!")
    
    # List existing projects with enhanced options
    st.write("### Your Projects")
    
    if not st.session_state.cv_data['projects']:
        st.info("No projects added yet. Add your first project above.")
    else:
        for i, project in enumerate(st.session_state.cv_data['projects']):
            with st.expander(f"🔹 {project['name']}", expanded=False):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    **Technologies:** {project.get('tech_stack', 'N/A')}  
                    **Duration:** {project.get('start_date', '')} to {project.get('end_date', '')}
                    """)
                    if project['url']:
                        st.markdown(f"**URL:** [{project['url']}]({project['url']})")
                    st.markdown(f"**Description:**  \n{project['description']}")
                
                with col2:
                    if st.button(f"🗑️ Delete", key=f"del_proj_{i}"):
                        st.session_state.cv_data['projects'].pop(i)
                        st.rerun()
                    
                    if st.button(f"✨ Enhance", key=f"enhance_proj_{i}"):
                        with st.spinner("Enhancing project description..."):
                            enhanced = generate_achievement_bullets(
                                project['name'],
                                project['description']
                            )
                            st.session_state.cv_data['projects'][i]['description'] = enhanced
                            st.rerun()
                    
                    if st.button(f"🔧 Improve", key=f"improve_proj_{i}"):
                        with st.spinner("Improving description..."):
                            improved = improve_content(project['description'], "project description")
                            st.session_state.cv_data['projects'][i]['description'] = improved
                            st.rerun()

def certifications_section():
    """Render certifications form section with enhanced UI."""
    st.subheader("🏆 Certifications")
    
    # Add new certification
    with st.expander("➕ Add New Certification", expanded=True):
        with st.form("certification_form"):
            name = st.text_input("Certification Name*", placeholder="e.g., AWS Certified Solutions Architect")
            issuer = st.text_input("Issuing Organization*", placeholder="e.g., Amazon Web Services")
            
            col1, col2 = st.columns(2)
            with col1:
                date = st.date_input("Date Obtained")
                expiry_date = st.date_input("Expiry Date (if applicable)", key="cert_expiry")
            
            with col2:
                url = st.text_input("Credential URL", placeholder="https://credential.net/yourid")
                credential_id = st.text_input("Credential ID (if applicable)")
            
            if st.form_submit_button("Add Certification"):
                required_fields = {
                    "Certification Name": name,
                    "Issuing Organization": issuer
                }
                
                if validate_required_fields(required_fields, "Certifications"):
                    certification = {
                        "name": name,
                        "issuer": issuer,
                        "date": str(date) if date else "",
                        "expiry_date": str(expiry_date) if expiry_date else "",
                        "url": url,
                        "credential_id": credential_id
                    }
                    st.session_state.cv_data['certifications'].append(certification)
                    st.success("Certification added successfully!")
    
    # List existing certifications with enhanced options
    st.write("### Your Certifications")
    
    if not st.session_state.cv_data['certifications']:
        st.info("No certifications added yet. Add your first certification above.")
    else:
        for i, cert in enumerate(st.session_state.cv_data['certifications']):
            with st.expander(f"🔹 {cert['name']}", expanded=False):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.markdown(f"""
                    **Issuer:** {cert['issuer']}  
                    **Date Obtained:** {cert.get('date', 'N/A')}  
                    **Expiry Date:** {cert.get('expiry_date', 'N/A')}
                    """)
                    if cert.get('credential_id'):
                        st.markdown(f"**Credential ID:** {cert['credential_id']}")
                    if cert['url']:
                        st.markdown(f"**URL:** [{cert['url']}]({cert['url']})")
                
                with col2:
                    if st.button(f"🗑️ Delete", key=f"del_cert_{i}"):
                        st.session_state.cv_data['certifications'].pop(i)
                        st.rerun()

def custom_sections():
    """Render custom sections form with enhanced UI."""
    st.subheader("📌 Custom Sections")
    
    # Add new custom section
    with st.expander("➕ Add Custom Section", expanded=True):
        with st.form("custom_section_form"):
            section_title = st.text_input("Section Title*", placeholder="e.g., Publications, Volunteer Work")
            section_content = st.text_area(
                "Content*", 
                height=150,
                placeholder="Add details for this section...",
                help="You can add bullet points, paragraphs, or any other relevant information"
            )
            
            if st.form_submit_button("Add Section"):
                required_fields = {
                    "Section Title": section_title,
                    "Content": section_content
                }
                
                if validate_required_fields(required_fields, "Custom Sections"):
                    section = {
                        "title": section_title,
                        "content": section_content
                    }
                    st.session_state.cv_data['custom_sections'].append(section)
                    st.success("Custom section added successfully!")
    
    # List existing custom sections with enhanced options
    st.write("### Your Custom Sections")
    
    if not st.session_state.cv_data['custom_sections']:
        st.info("No custom sections added yet. Add your first section above.")
    else:
        for i, section in enumerate(st.session_state.cv_data['custom_sections']):
            with st.expander(f"🔹 {section['title']}", expanded=False):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(section['content'])
                
                with col2:
                    if st.button(f"🗑️ Delete", key=f"del_section_{i}"):
                        st.session_state.cv_data['custom_sections'].pop(i)
                        st.rerun()
                    
                    if st.button(f"🔧 Improve", key=f"improve_section_{i}"):
                        with st.spinner("Improving content..."):
                            improved = improve_content(section['content'], section['title'])
                            st.session_state.cv_data['custom_sections'][i]['content'] = improved
                            st.rerun()

def settings_section():
    """Render CV customization settings."""
    st.subheader("⚙️ CV Customization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Theme selection
        st.write("### Design Theme")
        theme = st.selectbox(
            "Select a color theme",
            list(THEMES.keys()),
            index=list(THEMES.keys()).index(st.session_state.cv_data['settings']['theme'])
        )
        st.session_state.cv_data['settings']['theme'] = theme
        
        # Show color preview
        theme_colors = THEMES[theme]
        st.markdown(f"""
        <div style="background-color: {theme_colors['primary']}; height: 30px; border-radius: 5px; margin: 10px 0;"></div>
        <div style="background-color: {theme_colors['secondary']}; height: 30px; border-radius: 5px; margin: 10px 0;"></div>
        """, unsafe_allow_html=True)
        
        # Font selection
        st.write("### Font Selection")
        font = st.selectbox(
            "Select font style",
            ["Arial", "Times New Roman", "Calibri", "Helvetica", "Georgia"],
            index=["Arial", "Times New Roman", "Calibri", "Helvetica", "Georgia"].index(
                st.session_state.cv_data['settings'].get('font', 'Arial'))
        )
        st.session_state.cv_data['settings']['font'] = font
    
    with col2:
        # Layout selection
        st.write("### CV Layout")
        layout = st.selectbox(
            "Select CV layout style",
            ["Traditional", "Modern", "Creative", "Minimalist", "ATS-Friendly"],
            index=["Traditional", "Modern", "Creative", "Minimalist", "ATS-Friendly"].index(
                st.session_state.cv_data['settings'].get('layout', 'Traditional'))
        )
        st.session_state.cv_data['settings']['layout'] = layout
        
        # Additional options
        st.write("### Additional Options")
        st.session_state.cv_data['settings']['show_photo'] = st.checkbox(
            "Show photo in CV",
            value=st.session_state.cv_data['settings']['show_photo']
        )
        
        st.session_state.cv_data['settings']['compact_mode'] = st.checkbox(
            "Compact layout (1-page CV)",
            value=st.session_state.cv_data['settings'].get('compact_mode', False)
        )
    
    st.success("Settings saved automatically!")

# Enhanced PDF Generation with multiple formats
def generate_pdf():
    """Generate a PDF version of the CV with enhanced styling."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            filename = tmpfile.name
        
        # Get selected theme
        theme = THEMES[st.session_state.cv_data['settings']['theme']]
        font = st.session_state.cv_data['settings']['font']
        
        # Font mapping for PDF generation
        FONT_MAP = {
            "Arial": "Helvetica",
            "Arial-Bold": "Helvetica-Bold",
            "Arial-Italic": "Helvetica-Oblique",
            "Times New Roman": "Times-Roman",
            "Times New Roman-Bold": "Times-Bold",
            "Times New Roman-Italic": "Times-Italic",
            "Calibri": "Helvetica",  # Substitute, as Calibri is not built-in
            "Calibri-Bold": "Helvetica-Bold",
            "Calibri-Italic": "Helvetica-Oblique",
            "Helvetica": "Helvetica",
            "Helvetica-Bold": "Helvetica-Bold",
            "Helvetica-Italic": "Helvetica-Oblique",
            "Georgia": "Times-Roman",  # Substitute, as Georgia is not built-in
            "Georgia-Bold": "Times-Bold",
            "Georgia-Italic": "Times-Italic",
        }
        
        # Base font settings
        font_base = FONT_MAP.get(font, "Helvetica")
        font_bold = FONT_MAP.get(f"{font}-Bold", "Helvetica-Bold")
        font_italic = FONT_MAP.get(f"{font}-Italic", "Helvetica-Oblique")
        
        # Create PDF document
        doc = SimpleDocTemplate(
            filename,
            pagesize=letter,
            leftMargin=0.5*inch,
            rightMargin=0.5*inch,
            topMargin=0.5*inch,
            bottomMargin=0.5*inch
        )
        
        styles = getSampleStyleSheet()
        
        # Custom styles
        header_style = ParagraphStyle(
            'Header1',
            parent=styles['Heading1'],
            fontName=font_base,  # FIXED
            fontSize=18,
            leading=22,
            spaceAfter=12,
            textColor=colors.HexColor(theme['primary']),
            alignment=TA_LEFT
        )

        name_style = ParagraphStyle(
            'NameStyle',
            parent=styles['Heading1'],
            fontName=font_bold,  # FIXED
            fontSize=24,
            leading=28,
            spaceAfter=6,
            textColor=colors.HexColor(theme['primary']),
            alignment=TA_CENTER
        )

        section_style = ParagraphStyle(
            'Header2',
            parent=styles['Heading2'],
            fontName=font_bold,  # FIXED
            fontSize=12,
            leading=16,
            spaceAfter=6,
            textColor=colors.HexColor(theme['primary']),
            alignment=TA_LEFT
        )

        subtitle_style = ParagraphStyle(
            'Subtitle',
            parent=styles['Normal'],
            fontName=font_italic,  # FIXED
            fontSize=10,
            leading=12,
            spaceAfter=12,
            textColor=colors.HexColor(theme['text']),
            alignment=TA_CENTER
        )

        normal_style = ParagraphStyle(
            'Normal',
            parent=styles['Normal'],
            fontName=font_base,  # FIXED
            fontSize=10,
            leading=12,
            textColor=colors.HexColor(theme['text']),
            alignment=TA_JUSTIFY
        )

        # Build PDF content
        content = []
        
        # Header with photo if enabled
        personal_info = st.session_state.cv_data['personal_info']
        
        if st.session_state.cv_data['settings']['show_photo'] and st.session_state.cv_data['settings']['photo']:
            # Create a table for header with photo
            photo_data = st.session_state.cv_data['settings']['photo']['data']
            photo_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name
            with open(photo_path, "wb") as f:
                f.write(base64.b64decode(photo_data))
            
            header_table = Table([
                [
                    Image(photo_path, width=1*inch, height=1*inch),
                    Paragraph(personal_info.get('full_name', ''), name_style),
                    ""  # Empty cell for layout
                ]
            ], colWidths=[1.2*inch, 4*inch, 1.8*inch])
            
            header_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('ALIGN', (0, 0), (0, 0), 'CENTER'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12)
            ]))
            
            content.append(header_table)
        else:
            # Center-aligned name without photo
            content.append(Paragraph(personal_info.get('full_name', ''), name_style))
        
        # Contact information
        contact_info = []
        if personal_info.get('email'):
            contact_info.append(personal_info['email'])
        if personal_info.get('phone'):
            contact_info.append(personal_info['phone'])
        if personal_info.get('location'):
            contact_info.append(personal_info['location'])
        
        if contact_info:
            content.append(Paragraph(" | ".join(contact_info), subtitle_style))
        
        # Links
        links = []
        if personal_info.get('linkedin'):
            links.append(f"LinkedIn: {personal_info['linkedin']}")
        if personal_info.get('portfolio'):
            links.append(f"Portfolio: {personal_info['portfolio']}")
        
        if links:
            content.append(Paragraph(" | ".join(links), subtitle_style))
        
        content.append(Spacer(1, 12))
        
        # Add horizontal line
        content.append(Table(
            [[ "" ]],
            colWidths=[7*inch],
            style=[
                ('LINEABOVE', (0, 0), (-1, -1), 1, colors.HexColor(theme['secondary']))
            ]
        ))
        content.append(Spacer(1, 12))
        
        # Professional Summary
        if st.session_state.cv_data['professional_summary']:
            content.append(Paragraph("PROFESSIONAL SUMMARY", section_style))
            content.append(Paragraph(st.session_state.cv_data['professional_summary'], normal_style))
            content.append(Spacer(1, 12))
        
        # Work Experience
        if st.session_state.cv_data['work_experience']:
            content.append(Paragraph("WORK EXPERIENCE", section_style))
            for exp in st.session_state.cv_data['work_experience']:
                job_title = exp.get('job_title', '')
                company = exp.get('company', '')
                duration = f"{exp.get('start_date', '')} - {exp.get('end_date', '')}"
                location = exp.get('location', '')
                
                # Create experience header table
                exp_table = Table([
                    [
                        Paragraph(f"<b>{job_title}</b>", normal_style),
                        Paragraph(duration, normal_style)
                    ],
                    [
                        Paragraph(f"<i>{company}</i>", normal_style),
                        Paragraph(location if location else "", normal_style)
                    ]
                ], colWidths=[4*inch, 3*inch])
                
                exp_table.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 2)
                ]))
                
                content.append(exp_table)
                content.append(Paragraph(exp.get('description', ''), normal_style))
                content.append(Spacer(1, 8))
        
        # Education
        if st.session_state.cv_data['education']:
            content.append(Paragraph("EDUCATION", section_style))
            for edu in st.session_state.cv_data['education']:
                degree = edu.get('degree', '')
                institution = edu.get('institution', '')
                duration = ""
                
                if edu.get('start_date') or edu.get('end_date'):
                    duration = f"{edu.get('start_date', '')} - {edu.get('end_date', '')}"
                
                # Create education header table
                edu_table = Table([
                    [
                        Paragraph(f"<b>{degree}</b>", normal_style),
                        Paragraph(duration, normal_style)
                    ],
                    [
                        Paragraph(f"<i>{institution}</i>", normal_style),
                        ""
                    ]
                ], colWidths=[4*inch, 3*inch])
                
                edu_table.setStyle(TableStyle([
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 2)
                ]))
                
                content.append(edu_table)
                if edu.get('description'):
                    content.append(Paragraph(edu.get('description'), normal_style))
                content.append(Spacer(1, 8))
        
        # Skills
        if st.session_state.cv_data['skills']:
            content.append(Paragraph("SKILLS", section_style))
            
            # Create a 3-column table for skills
            skills = st.session_state.cv_data['skills']
            skill_rows = []
            
            # Split skills into groups of 3 for each row
            for i in range(0, len(skills), 3):
                row = skills[i:i+3]
                # Pad with empty strings if needed
                row += [''] * (3 - len(row))
                skill_rows.append(row)
            
            skills_table = Table(skill_rows, colWidths=[2.33*inch]*3)
            skills_table.setStyle(TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('FONTNAME', (0, 0), (-1, -1), font_base),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 4)
            ]))
            
            content.append(skills_table)
            content.append(Spacer(1, 12))
        
        # Projects
        if st.session_state.cv_data['projects']:
            content.append(Paragraph("PROJECTS", section_style))
            for project in st.session_state.cv_data['projects']:
                name = project.get('name', '')
                url = project.get('url', '')
                
                proj_header = f"<b>{name}</b>"
                if url:
                    proj_header += f" | {url}"
                
                content.append(Paragraph(proj_header, normal_style))
                content.append(Paragraph(project.get('description', ''), normal_style))
                content.append(Spacer(1, 8))
        
        # Certifications
        if st.session_state.cv_data['certifications']:
            content.append(Paragraph("CERTIFICATIONS", section_style))
            for cert in st.session_state.cv_data['certifications']:
                name = cert.get('name', '')
                issuer = cert.get('issuer', '')
                date = cert.get('date', '')
                url = cert.get('url', '')
                
                cert_header = f"<b>{name}</b>, {issuer}"
                if date:
                    cert_header += f" | {date}"
                if url:
                    cert_header += f" | {url}"
                
                content.append(Paragraph(cert_header, normal_style))
                content.append(Spacer(1, 8))
        
        # Custom Sections
        for section in st.session_state.cv_data['custom_sections']:
            content.append(Paragraph(section['title'].upper(), section_style))
            content.append(Paragraph(section['content'], normal_style))
            content.append(Spacer(1, 12))
        
        # Build PDF
        doc.build(content)
        
        return filename
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

def generate_html_cv():
    """Generate an HTML version of the CV."""
    try:
        theme = THEMES[st.session_state.cv_data['settings']['theme']]
        personal_info = st.session_state.cv_data['personal_info']
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{personal_info.get('full_name', 'CV')}'s CV</title>
            <style>
                body {{
                    font-family: {st.session_state.cv_data['settings']['font']}, sans-serif;
                    line-height: 1.6;
                    color: {theme['text']};
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    text-align: center;
                    margin-bottom: 20px;
                }}
                .name {{
                    font-size: 28px;
                    font-weight: bold;
                    color: {theme['primary']};
                    margin-bottom: 5px;
                }}
                .contact-info {{
                    font-style: italic;
                    margin-bottom: 5px;
                }}
                .section {{
                    margin-bottom: 20px;
                }}
                .section-title {{
                    font-size: 18px;
                    font-weight: bold;
                    color: {theme['primary']};
                    border-bottom: 2px solid {theme['secondary']};
                    padding-bottom: 5px;
                    margin-bottom: 10px;
                }}
                .job-title, .degree {{
                    font-weight: bold;
                }}
                .company, .institution {{
                    font-style: italic;
                }}
                .date {{
                    float: right;
                }}
                .skills-list {{
                    column-count: 3;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <div class="name">{personal_info.get('full_name', '')}</div>
                <div class="contact-info">
                    {personal_info.get('email', '')} | {personal_info.get('phone', '')} | {personal_info.get('location', '')}
                </div>
                <div>
                    {f'<a href="{personal_info["linkedin"]}">LinkedIn</a>' if personal_info.get('linkedin') else ''}
                    {f' | <a href="{personal_info["portfolio"]}">Portfolio</a>' if personal_info.get('portfolio') else ''}
                </div>
            </div>
        """
        
        # Professional Summary
        if st.session_state.cv_data['professional_summary']:
            html += f"""
            <div class="section">
                <div class="section-title">PROFESSIONAL SUMMARY</div>
                <div>{st.session_state.cv_data['professional_summary']}</div>
            </div>
            """
        
        # Work Experience
        if st.session_state.cv_data['work_experience']:
            html += """
            <div class="section">
                <div class="section-title">WORK EXPERIENCE</div>
            """
            
            for exp in st.session_state.cv_data['work_experience']:
                html += f"""
                <div>
                    <span class="job-title">{exp['job_title']}</span>, 
                    <span class="company">{exp['company']}</span>
                    <span class="date">{exp['start_date']} - {exp['end_date']}</span>
                    <div>{exp.get('location', '')}</div>
                    <p>{exp['description']}</p>
                </div>
                """
            
            html += "</div>"
        
        # Education
        if st.session_state.cv_data['education']:
            html += """
            <div class="section">
                <div class="section-title">EDUCATION</div>
            """
            
            for edu in st.session_state.cv_data['education']:
                html += f"""
                <div>
                    <span class="degree">{edu['degree']}</span>, 
                    <span class="institution">{edu['institution']}</span>
                    <span class="date">{edu['start_date']} - {edu['end_date']}</span>
                    <p>{edu.get('description', '')}</p>
                </div>
                """
            
            html += "</div>"
        
        # Skills
        if st.session_state.cv_data['skills']:
            html += """
            <div class="section">
                <div class="section-title">SKILLS</div>
                <div class="skills-list">
                    <ul>
            """
            
            for skill in st.session_state.cv_data['skills']:
                html += f"<li>{skill}</li>"
            
            html += """
                    </ul>
                </div>
            </div>
            """
        
        # Projects
        if st.session_state.cv_data['projects']:
            html += """
            <div class="section">
                <div class="section-title">PROJECTS</div>
            """
            
            for project in st.session_state.cv_data['projects']:
                html += f"""
                <div>
                    <div class="job-title">{project['name']}</div>
                    {f'<div><a href="{project["url"]}">{project["url"]}</a></div>' if project.get('url') else ''}
                    <p>{project['description']}</p>
                </div>
                """
            
            html += "</div>"
        
        # Certifications
        if st.session_state.cv_data['certifications']:
            html += """
            <div class="section">
                <div class="section-title">CERTIFICATIONS</div>
            """
            
            for cert in st.session_state.cv_data['certifications']:
                html += f"""
                <div>
                    <span class="job-title">{cert['name']}</span>, 
                    <span class="company">{cert['issuer']}</span>
                    <span class="date">{cert.get('date', '')}</span>
                    {f'<div><a href="{cert["url"]}">{cert["url"]}</a></div>' if cert.get('url') else ''}
                </div>
                """
            
            html += "</div>"
        
        # Custom Sections
        for section in st.session_state.cv_data['custom_sections']:
            html += f"""
            <div class="section">
                <div class="section-title">{section['title'].upper()}</div>
                <div>{section['content']}</div>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmpfile:
            tmpfile.write(html.encode('utf-8'))
            return tmpfile.name
    except Exception as e:
        st.error(f"Error generating HTML: {str(e)}")
        return None

# Enhanced Preview functionality
def preview_cv():
    """Render an enhanced preview of the CV."""
    st.subheader("👁️ CV Preview")
    
    # Theme styling
    theme = THEMES[st.session_state.cv_data['settings']['theme']]
    font = st.session_state.cv_data['settings']['font']
    
    # Apply theme to preview
    st.markdown(f"""
    <style>
        .cv-preview {{
            font-family: {font}, sans-serif;
            color: {theme['text']};
            padding: 20px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: white;
        }}
        .cv-header {{
            color: {theme['primary']};
            border-bottom: 2px solid {theme['secondary']};
            padding-bottom: 5px;
            margin-bottom: 15px;
        }}
        .cv-section {{
            margin-bottom: 20px;
        }}
        .cv-section-title {{
            color: {theme['primary']};
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 10px;
        }}
        .cv-job-title, .cv-degree {{
            font-weight: bold;
        }}
        .cv-company, .cv-institution {{
            font-style: italic;
        }}
        .cv-date {{
            float: right;
            color: #666;
        }}
    </style>
    """, unsafe_allow_html=True)
    
    personal_info = st.session_state.cv_data['personal_info']
    
    # Preview container
    with st.container():
        st.markdown("""
        <div class="cv-preview">
            <div style="text-align: center;">
                <div style="font-size: 24px; font-weight: bold; color: %s; margin-bottom: 5px;">%s</div>
                <div style="font-style: italic; margin-bottom: 5px;">
                    %s | %s | %s
                </div>
                <div>
                    %s %s
                </div>
            </div>
            <hr style="border-top: 2px solid %s; margin: 15px 0;">
        """ % (
            theme['primary'],
            personal_info.get('full_name', 'Your Name'),
            personal_info.get('email', 'email@example.com'),
            personal_info.get('phone', '+1 (123) 456-7890'),
            personal_info.get('location', 'City, Country'),
            f'<a href="{personal_info["linkedin"]}">LinkedIn</a>' if personal_info.get('linkedin') else '',
            f' | <a href="{personal_info["portfolio"]}">Portfolio</a>' if personal_info.get('portfolio') else '',
            theme['secondary']
        ), unsafe_allow_html=True)
        
        # Professional Summary
        if st.session_state.cv_data['professional_summary']:
            st.markdown("""
            <div class="cv-section">
                <div class="cv-section-title">PROFESSIONAL SUMMARY</div>
                <div>%s</div>
            </div>
            """ % st.session_state.cv_data['professional_summary'], unsafe_allow_html=True)
        
        # Work Experience
        if st.session_state.cv_data['work_experience']:
            st.markdown("""
            <div class="cv-section">
                <div class="cv-section-title">WORK EXPERIENCE</div>
            """, unsafe_allow_html=True)
            
            for exp in st.session_state.cv_data['work_experience']:
                st.markdown("""
                <div style="margin-bottom: 10px;">
                    <span class="cv-job-title">%s</span>, 
                    <span class="cv-company">%s</span>
                    <span class="cv-date">%s - %s</span>
                    <div>%s</div>
                    <div style="margin-top: 5px;">%s</div>
                </div>
                """ % (
                    exp['job_title'],
                    exp['company'],
                    exp['start_date'],
                    exp['end_date'],
                    exp.get('location', ''),
                    exp['description']
                ), unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Education
        if st.session_state.cv_data['education']:
            st.markdown("""
            <div class="cv-section">
                <div class="cv-section-title">EDUCATION</div>
            """, unsafe_allow_html=True)
            
            for edu in st.session_state.cv_data['education']:
                st.markdown("""
                <div style="margin-bottom: 10px;">
                    <span class="cv-degree">%s</span>, 
                    <span class="cv-institution">%s</span>
                    <span class="cv-date">%s - %s</span>
                    <div style="margin-top: 5px;">%s</div>
                </div>
                """ % (
                    edu['degree'],
                    edu['institution'],
                    edu['start_date'],
                    edu['end_date'],
                    edu.get('description', '')
                ), unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Skills
        if st.session_state.cv_data['skills']:
            st.markdown("""
            <div class="cv-section">
                <div class="cv-section-title">SKILLS</div>
                <div style="column-count: 3;">
                    <ul style="margin-top: 0;">
            """, unsafe_allow_html=True)
            
            for skill in st.session_state.cv_data['skills']:
                st.markdown(f"<li>{skill}</li>", unsafe_allow_html=True)
            
            st.markdown("""
                    </ul>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Projects
        if st.session_state.cv_data['projects']:
            st.markdown("""
            <div class="cv-section">
                <div class="cv-section-title">PROJECTS</div>
            """, unsafe_allow_html=True)
            
            for project in st.session_state.cv_data['projects']:
                st.markdown("""
                <div style="margin-bottom: 10px;">
                    <div class="cv-job-title">%s</div>
                    %s
                    <div style="margin-top: 5px;">%s</div>
                </div>
                """ % (
                    project['name'],
                    f'<div><a href="{project["url"]}">{project["url"]}</a></div>' if project.get('url') else '',
                    project['description']
                ), unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Certifications
        if st.session_state.cv_data['certifications']:
            st.markdown("""
            <div class="cv-section">
                <div class="cv-section-title">CERTIFICATIONS</div>
            """, unsafe_allow_html=True)
            
            for cert in st.session_state.cv_data['certifications']:
                st.markdown("""
                <div style="margin-bottom: 10px;">
                    <span class="cv-job-title">%s</span>, 
                    <span class="cv-company">%s</span>
                    <span class="cv-date">%s</span>
                    %s
                </div>
                """ % (
                    cert['name'],
                    cert['issuer'],
                    cert.get('date', ''),
                    f'<div><a href="{cert["url"]}">{cert["url"]}</a></div>' if cert.get('url') else ''
                ), unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Custom Sections
        for section in st.session_state.cv_data['custom_sections']:
            st.markdown("""
            <div class="cv-section">
                <div class="cv-section-title">%s</div>
                <div>%s</div>
            </div>
            """ % (
                section['title'].upper(),
                section['content']
            ), unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

# Job matching and analysis
def job_matching_section():
    """Section for job matching and analysis."""
    st.subheader("🔍 Job Matching Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Job Description Analysis")
        job_description = st.text_area(
            "Paste the job description you're applying for",
            height=200,
            placeholder="Paste the full job description here..."
        )
        
        if st.button("Analyze Job Fit"):
            if job_description and st.session_state.cv_data['personal_info'].get('full_name'):
                with st.spinner("Analyzing how well your CV matches this job..."):
                    analysis = analyze_job_fit(job_description, st.session_state.cv_data)
                    st.session_state.ai_suggestions['job_fit_analysis'] = analysis
                    st.rerun()
            else:
                st.error("Please paste a job description and fill in your personal information first")
    
    with col2:
        st.write("### Cover Letter Generator")
        company = st.text_input("Company Name", placeholder="e.g., Google")
        
        if st.button("Generate Cover Letter"):
            if job_description and company:
                with st.spinner("Generating tailored cover letter..."):
                    cover_letter = generate_cover_letter(job_description, company)
                    st.session_state.ai_suggestions['cover_letter'] = cover_letter
                    st.rerun()
            else:
                st.error("Please provide both job description and company name")
    
    # Display analysis if available
    if 'job_fit_analysis' in st.session_state.ai_suggestions:
        st.markdown("### Job Fit Analysis")
        st.markdown(st.session_state.ai_suggestions['job_fit_analysis'])
    
    # Display cover letter if available
    if 'cover_letter' in st.session_state.ai_suggestions:
        st.markdown("### Generated Cover Letter")
        st.markdown(st.session_state.ai_suggestions['cover_letter'])
        
        if st.button("Download Cover Letter"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmpfile:
                tmpfile.write(st.session_state.ai_suggestions['cover_letter'].encode('utf-8'))
                st.download_button(
                    label="⬇️ Download Cover Letter",
                    data=open(tmpfile.name, "rb").read(),
                    file_name=f"Cover_Letter_{company}.txt",
                    mime="text/plain"
                )

# Main app with enhanced navigation
def main():
    """Main application function with enhanced UI."""

    # --- SESSION STATE INITIALIZATION ---
    if "active_tab" not in st.session_state:
        st.session_state.active_tab = "Personal Information"
    if "cv_data" not in st.session_state:
        st.session_state.cv_data = {
            "personal_info": {
                "full_name": "",
                "email": "",
                "phone": "",
                "linkedin": "",
                "portfolio": "",
                "location": "",
                "target_role": "",
                "career_objective": ""
            },
            "professional_summary": "",
            "work_experience": [],
            "education": [],
            "skills": [],
            "projects": [],
            "certifications": [],
            "custom_sections": [],
            "settings": {
                "theme": "Professional Blue",
                "font": "Arial",
                "layout": "Traditional",
                "show_photo": False,
                "photo": None,
                "compact_mode": False
            }
        }
    if "ai_suggestions" not in st.session_state:
        st.session_state.ai_suggestions = {}

    # Custom CSS for the entire app
    st.markdown("""
    <style>
        .stApp {
            background-color: #181a1b !important;
            color: #f5f6fa !important;
        }
        .sidebar .sidebar-content {
            background-color: #23272f !important;
            color: #f5f6fa !important;
        }
        .stButton>button {
            border-radius: 5px;
            font-weight: bold;
            transition: all 0.3s ease;
            background-color: #23272f !important;
            color: #f5f6fa !important;
            border: 1px solid #444 !important;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            background-color: #2c313c !important;
        }
        .stTextInput>div>div>input, .stTextArea>div>div>textarea {
            border-radius: 5px;
            background-color: #23272f !important;
            color: #f5f6fa !important;
            border: 1px solid #444 !important;
        }
        .stExpander {
            border-radius: 5px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3);
            background-color: #23272f !important;
            color: #f5f6fa !important;
        }
        .stAlert {
            border-radius: 5px;
            background-color: #23272f !important;
            color: #f5f6fa !important;
        }
        .stMarkdown, .stSubheader, .stTitle, .stHeader {
            color: #f5f6fa !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image("https://via.placeholder.com/150x50?text=CV+Builder", width=150)
    with col2:
        st.title("Professional CV Builder with AI Assistance")
        st.markdown("Create a stunning, ATS-friendly CV with AI-powered enhancements")
    
    # Sidebar navigation with icons
    with st.sidebar:
        tab_options = [
            "Personal Information",
            "Professional Summary",
            "Work Experience",
            "Education",
            "Skills",
            "Projects",
            "Certifications",
            "Custom Sections",
            "Job Matching",
            "Settings",
            "Preview & Export"
        ]
        tab_icons = [
            "👤", "📝", "💼", "🎓", "🛠️", "📂", "🏆", "📌", "🔍", "⚙️", "👁️"
        ]
        tab_labels = [f"{icon} {label}" for icon, label in zip(tab_icons, tab_options)]
        selected = st.radio(
            "Navigation",
            tab_labels,
            index=tab_options.index(st.session_state.active_tab),
            label_visibility="collapsed"
        )
        st.session_state.active_tab = tab_options[tab_labels.index(selected)]
    
    # Main content area
    if st.session_state.active_tab == "Personal Information":
        personal_information_section()
    elif st.session_state.active_tab == "Professional Summary":
        professional_summary_section()
    elif st.session_state.active_tab == "Work Experience":
        work_experience_section()
    elif st.session_state.active_tab == "Education":
        education_section()
    elif st.session_state.active_tab == "Skills":
        skills_section()
    elif st.session_state.active_tab == "Projects":
        projects_section()
    elif st.session_state.active_tab == "Certifications":
        certifications_section()
    elif st.session_state.active_tab == "Custom Sections":
        custom_sections()
    elif st.session_state.active_tab == "Job Matching":
        job_matching_section()
    elif st.session_state.active_tab == "Settings":
        settings_section()
    elif st.session_state.active_tab == "Preview & Export":
        preview_cv()
        
        st.markdown("---")
        st.subheader("📤 Export Options")
        
        # Format optimization suggestions
        if st.button("🛠️ Get Optimization Suggestions", use_container_width=True):
            with st.spinner("Analyzing your CV for optimization..."):
                suggestions = format_optimization_suggestions(st.session_state.cv_data)
                st.markdown("### Format Optimization Suggestions")
                st.write(suggestions)
        
        # Export buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # PDF Export
            if st.button("📄 Generate PDF", use_container_width=True):
                with st.spinner("Generating PDF..."):
                    pdf_file = generate_pdf()
                    if pdf_file:
                        with open(pdf_file, "rb") as f:
                            st.download_button(
                                label="⬇️ Download PDF",
                                data=f,
                                file_name="professional_cv.pdf",
                                mime="application/pdf",
                                use_container_width=True
                            )
                        os.unlink(pdf_file)
        
        with col2:
            # HTML Export
            if st.button("🌐 Generate HTML", use_container_width=True):
                with st.spinner("Generating HTML..."):
                    html_file = generate_html_cv()
                    if html_file:
                        with open(html_file, "rb") as f:
                            st.download_button(
                                label="⬇️ Download HTML",
                                data=f,
                                file_name="professional_cv.html",
                                mime="text/html",
                                use_container_width=True
                            )
                        os.unlink(html_file)
        
        with col3:
            # JSON Export (for data backup)
            if st.button("💾 Export JSON", use_container_width=True):
                json_data = json.dumps(st.session_state.cv_data, indent=2)
                st.download_button(
                    label="⬇️ Download JSON",
                    data=json_data,
                    file_name="cv_data.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        # Import JSON (for data loading)
        st.markdown("---")
        st.subheader("📥 Import CV Data")
        uploaded_file = st.file_uploader("Upload JSON file to load CV data", type=["json"])
        if uploaded_file is not None:
            try:
                data = json.load(uploaded_file)
                st.session_state.cv_data = data
                st.success("CV data imported successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error importing JSON: {str(e)}")

if __name__ == "__main__":
    main()