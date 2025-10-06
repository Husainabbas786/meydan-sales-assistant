import os
import streamlit as st
from dotenv import load_dotenv
from llama_cloud_services import LlamaCloudIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate

load_dotenv()

# Page config
st.set_page_config(
    page_title="Meydan Free Zone Sales Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        padding: 1rem;
        border-bottom: 3px solid #1f77b4;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        padding: 0.5rem;
        background-color: #f0f2f6;
        border-left: 5px solid #1f77b4;
    }
    .profile-table {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .recommendation-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin-bottom: 1rem;
    }
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize LlamaIndex and OpenAI - EXACTLY as in chatbot.py
@st.cache_resource
def initialize_services():
    """Initialize services exactly as in chatbot.py"""
    print("Initializing Meydan Free Zone Sales Assistant...")
    index = LlamaCloudIndex(
        name="business_activity_intelligence-1759747899",  # EXACT name from chatbot.py
        project_name="Default",
        organization_id=os.getenv("LLAMA_CLOUD_ORGANIZATION_ID"),
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    )
    
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
    llm = OpenAI(model="gpt-4o", temperature=0.1)  # EXACT model from chatbot.py
    
    return index, llm

# Initialize globally
index, llm = initialize_services()

# EXACT System Prompt from chatbot.py
SYSTEM_PROMPT = """You are an expert Meydan Free Zone business activity consultant with comprehensive knowledge of 2,267 business activities across multiple sources.

KNOWLEDGE SOURCES:
1. Business Activities Database (2,267 activities) - Activity codes, risk ratings, third-party approvals, descriptions, keywords, related activities
2. Activity Hubs Knowledge Base (151 detailed guides) - Expert insights for popular activities, strategic guidance, common scenarios
3. MFZ Business Activities Knowledge Base (Mike's sales expertise) - Strategic approaches, gray areas, banking considerations, jurisdiction rules
4. Country Risk Rating Database - Nationality risk ratings for Finance persona logic

PRIORITIZATION ALGORITHM (Applied with Persona-Specific Weights):
1. Correlation Accuracy: Minimum 90% semantic match with customer business description
2. Risk Rating Priority: Low → Medium → High (only when necessary)

PERSONA-SPECIFIC WEIGHTS:

Business Persona (Genuine Entrepreneurs):
- Correlation: 85% weight (STRICT 90%+ match required)
- Risk Rating: 15% weight
- Logic: Business owners need exact activity match - prioritize correlation above all

Residential Persona (Visa/Residency Focused):
- Risk Rating: 50% weight
- Correlation: 50% weight (80%+ match acceptable)
- Logic: Visa seekers need easy approvals

Finance Persona (Banking/Tax Focused):
- FIRST: Check Country Risk Rating compatibility
- IF nationality is "Override" → Cannot issue license (stop immediately)
- IF nationality risk acceptable → Apply standard matrix
- Use bank account probability matrix:
  * High Nationality + High Activity = <30% success
  * High Nationality + Low Activity = <50% success
  * Low Nationality + Low Activity = >80% success
  * Low Nationality + High Activity = 60% success
- Compute for Medium ratings proportionally

MULTI-SOURCE RETRIEVAL STRATEGY:
Step 1: Search Activity Hubs (top_k=3) for popular activities with detailed guides
Step 2: Search Business Activities Database (top_k=10) for comprehensive coverage
Step 3: Search MFZ Knowledge Base for strategic insights (top_k=1)
Step 4: For Finance persona, query Country Risk Rating database

SYNTHESIS APPROACH:
1. If Activity Hubs has high-confidence match (>0.75 similarity) → Use as primary for ,Activity name, activity codes, description, expert insights, related activities (if present), third party approval, all relevant information which can be conveyed to customer.
2. Cross-reference Business Activities for: Code, Risk Rating, Third Party, When, Related Activities
3. Enrich with MFZ Knowledge Base strategic guidance ( expert insights on some common business activities general trading, approvals, banking, etc.)
4. For Finance: Apply Country Risk compatibility check first, check nationality of the user and determine the risk.

CHAIN-OF-THOUGHT REASONING:
For each recommendation, think through:
1. What is the customer's actual business/purpose?
2. What are synonym activities or related terms?
3. Which activities from all sources match semantically?
4. Apply persona-specific prioritization weights
5. Check for third-party approval complications
6. Validate correlation strength
7. Consider complementary activities within 3-group package
8. Assess banking implications (especially for Finance persona)

OUTPUT REQUIREMENTS:
Provide preferably 2 ranked activity recommendations. 
- Group (3-digit code)
- Activity Code (6-digit format like 1811.04)
- Activity Name
- Category (e.g., Manufacturing, Trading, Professional)
- Full Description from database
- Third Party Approval: Yes/No [Authority name if yes, e.g., "Dubai Municipality (DM)"]
- When: PRE/POST/N/A
- Risk Rating: Low/Medium/High
- Industry Risk: Yes/No/N/A
- Match Explanation: Why this fits customer needs (2-3 sentences with persona logic applied)
- Related Activities: 2-3 complementary activities (with codes, names, and 1-line descriptions)
- Expert Insights: Include strategic guidance from Activity Hubs and MFZ Knowledge Base if available

CRITICAL RULES:
1. For Business persona: Never compromise on correlation - must be 90%+ match
2. For Residential persona: Prefer N/A approvals and Low risk even if correlation is 80%+
3. For Finance persona: Check nationality risk first - if Override, immediately respond "Cannot issue license due to country risk rating"
4. Always suggest fewer activity groups when possible
5. Flag general trading concerns, even other concerns and suggest specific alternatives ( MFZ Business Activities Knowledge Base)
6. Explain any approval delays or banking complications transparently


Be precise, strategic, and consultative. Ensure recommendations maximize customer success while adhering to regulations."""

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 'welcome'
    st.session_state.profile = {
        "shareholders": None,
        "nationalities": [],
        "visas_needed": None,
        "business_description": None,
        "experience": None,
        "flexibility": None,
        "purpose": None,
        "timeline": None,
        "persona": None,
        "persona_answers": {}
    }
    st.session_state.current_question = 0
    st.session_state.recommendations = None
    st.session_state.chat_history = []

# EXACT helper functions from chatbot.py
def parse_nationalities(shareholder_answer):
    """Extract nationalities from shareholder answer"""
    return shareholder_answer

def interpret_experience(answer):
    """Convert experience answer to Branch/New"""
    answer_lower = answer.lower()
    if any(word in answer_lower for word in ["new", "starting", "first time", "venture"]):
        return "New"
    elif any(word in answer_lower for word in ["branch", "existing", "expand", "already"]):
        return "Branch"
    return answer

def interpret_flexibility(answer):
    """Convert flexibility answer to Flexible/Not Flexible"""
    answer_lower = answer.lower()
    if any(word in answer_lower for word in ["open", "flexible", "consider", "yes"]):
        return "Flexible"
    elif any(word in answer_lower for word in ["stick", "no", "only", "specific"]):
        return "Not Flexible"
    return answer

def concise_summary(answer, max_words=15):
    """Create concise summary of answers"""
    words = answer.split()
    if len(words) <= max_words:
        return answer
    return " ".join(words[:max_words]) + "..."

def get_activity_recommendations(profile):
    """Query index with persona-aware logic and chain-of-thought reasoning"""
    
    persona = profile['persona']
    
    # Build comprehensive query
    query_context = f"""
CUSTOMER PROFILE ANALYSIS:

Persona Type: {persona}
Number of Shareholders: {profile['shareholders']}
Nationalities: {profile['nationalities']}
Visas Needed: {profile['visas_needed']}
Business Description: {profile['business_description']}
Experience Level: {profile['experience']}
Business Flexibility: {profile['flexibility']}
Primary Purpose: {profile['purpose']}
Timeline: {profile['timeline']}

"""
    
    # Add persona-specific context
    if persona == "Residential":
        query_context += f"""
RESIDENTIAL PERSONA CONTEXT:
- Dependents: {profile['persona_answers'].get('dependents', 'N/A')}
- Residency Plan: {profile['persona_answers'].get('residency_plan', 'N/A')}

PRIORITIZATION FOR THIS PERSONA:
Apply weights: Risk (50%) + Correlation (50%)
Accept 80%+ correlation match
"""
    elif persona == "Business":
        query_context += f"""
BUSINESS PERSONA CONTEXT:
- Detailed Business Model: {profile['persona_answers'].get('business_model', 'N/A')}

PRIORITIZATION FOR THIS PERSONA:
Apply weights: Correlation (85%) + Risk (15%)
STRICT REQUIREMENT: Minimum 90% correlation with business description
This is a genuine entrepreneur - exact activity match is critical
"""
    elif persona == "Finance":
        query_context += f"""
FINANCE PERSONA CONTEXT:
- Invoicing Method: {profile['persona_answers'].get('invoicing', 'N/A')}
- Bank Account Purpose: {profile['persona_answers'].get('bank_purpose', 'N/A')}
- Tax Strategy: {profile['persona_answers'].get('tax_strategy', 'N/A')}

CRITICAL: Check Country Risk Rating first for nationalities: {profile['nationalities']}
IF any nationality has "Override" rating → Stop and respond "Cannot issue license"
IF acceptable ratings → Calculate bank account opening probability using nationality + activity risk matrix
Apply standard prioritization after country risk check passes
"""
    
    query_context += """

CHAIN-OF-THOUGHT ANALYSIS REQUIRED:
1. Analyze the business description and identify core activities
2. Search Activity Hubs for popular activity matches (e-commerce, general trading, consultancy, IT, advertising, etc.)
3. Search Business Activities Database for all possible matches using keywords and synonyms
4. Apply persona-specific prioritization weights
5. For each candidate activity, evaluate:
   - Semantic correlation strength (90%+ for Business, 80%+ for Residential/Finance)
   - Risk rating (Low preferred, High only when necessary)
   - Group optimization (fewer groups better)
6. Consider strategic insights from MFZ Knowledge Base (avoid general trading, flag concerns, suggest specific alternatives)
7. Consider complementary activities within 3-group package
8. Assess banking implications (especially for Finance persona)

DELIVERABLE:
Provide preferably 2 ranked activity recommendations following the format:

RECOMMENDATION 1: [Primary Recommendation]
Group: [3-digit code]
Activity Code: [6-digit format like 1811.04]
Activity Name: [Full name]
Category: [e.g., Manufacturing, Trading, Professional]
Full Description: [Full description from database]
Third Party Approval: [Yes/No] [Authority name if yes, e.g., "Dubai Municipality (DM)"]
When: [PRE/POST/N/A]
Risk Rating: [Low/Medium/High]
Industry Risk: [Yes/No/N/A]
Match Explanation: [2-3 sentences explaining why this fits customer needs with persona logic applied]
Related Activities:
  - [Code]: [Name] - [1-line description]
  - [Code]: [Name] - [1-line description]
Expert Insights: [Include strategic guidance from Activity Hubs and MFZ Knowledge Base if available]

RECOMMENDATION 2: [Secondary Recommendation]
Group: [3-digit code]
Activity Code: [6-digit format like 1811.04]
Activity Name: [Full name]
Category: [e.g., Manufacturing, Trading, Professional]
Full Description: [Full description from database]
Third Party Approval: [Yes/No] [Authority name if yes, e.g., "Dubai Municipality (DM)"]
When: [PRE/POST/N/A]
Risk Rating: [Low/Medium/High]
Industry Risk: [Yes/No/N/A]
Match Explanation: [2-3 sentences explaining why this fits customer needs with persona logic applied]
Related Activities:
  - [Code]: [Name] - [1-line description]
  - [Code]: [Name] - [1-line description]
Expert Insights: [Include strategic guidance from Activity Hubs and MFZ Knowledge Base if available]

CRITICAL RULES:
1. For Business persona: Never compromise on correlation - must be 90%+ match
2. For Residential persona: Prefer N/A approvals and Low risk even if correlation is 80%+
3. For Finance persona: Check nationality risk first - if Override, immediately respond "Cannot issue license due to country risk rating"
4. Always suggest fewer activity groups when possible
5. Flag general trading concerns, even other concerns and suggest specific alternatives (MFZ Business Activities Knowledge Base)
6. Explain any approval delays or banking complications transparently

Be precise, strategic, and consultative. Ensure recommendations maximize customer success while adhering to regulations.
"""
    
    # Retrieve with appropriate top_k based on persona
    top_k = 10 if persona == "Business" else 8
    
    retriever = index.as_retriever(similarity_top_k=top_k)
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=top_k)
    
    print("\n[Applying chain-of-thought reasoning across all knowledge sources...]")
    
    response = query_engine.query(query_context)
    return response.response
    
    
    

def update_field(profile, field_update):
    """Update customer profile field based on conversational input - EXACT from chatbot.py"""
    field_lower = field_update.lower()
    
    if "shareholder" in field_lower:
        words = field_update.split()
        for word in words:
            if word.isdigit():
                profile['shareholders'] = word
                return True
    
    elif "visa" in field_lower:
        words = field_update.split()
        for word in words:
            if word.isdigit():
                profile['visas_needed'] = word
                return True
    
    elif "business" in field_lower or "activity" in field_lower:
        profile['business_description'] = field_update
        return True
    
    elif "nationality" in field_lower or "passport" in field_lower:
        profile['nationalities'] = field_update
        return True
    
    elif "timeline" in field_lower:
        profile['timeline'] = field_update
        return True
    
    return False

# Main App
st.markdown('<div class="main-header">🏢 Meydan Free Zone Sales Assistant</div>', unsafe_allow_html=True)

# Welcome Step
if st.session_state.step == 'welcome':
    st.markdown("### Welcome! 👋")
    st.write("I'll help you identify the best business activities for your customer.")
    st.write("Let's start by gathering some information about the customer.")
    
    if st.button("Start Assessment", type="primary", use_container_width=True):
        st.session_state.step = 'questions'
        st.rerun()

# Initial Questions Step
elif st.session_state.step == 'questions':
    st.markdown('<div class="section-header">📋 Customer Information</div>', unsafe_allow_html=True)
    
    questions = [
        "What are the number of shareholders and what passport holders are they?",
        "How many visas do you want with this company?",
        "What business do you want to do?",
        "Have you been doing this business or is it a new venture?",
        "Are you open to do something else also or stick to your business plan?",
        "What is your primary purpose in establishing a company in UAE - Dubai?",
        "How soon are you planning to set up the company?"
    ]
    
    keys = ["shareholders_raw", "visas_needed", "business_description", 
            "experience_raw", "flexibility_raw", "purpose", "timeline"]
    
    with st.form("initial_questions"):
        answers = []
        for i, question in enumerate(questions):
            answer = st.text_area(f"Q{i+1}: {question}", key=f"q{i}", height=80)
            answers.append(answer)
        
        submitted = st.form_submit_button("Continue to Persona Selection", type="primary", use_container_width=True)
        
        if submitted:
            if all(answers):
                # Process answers EXACTLY as chatbot.py
                st.session_state.profile['shareholders'] = answers[0].split()[0] if answers[0].split()[0].isdigit() else "Not specified"
                st.session_state.profile['nationalities'] = answers[0]
                st.session_state.profile['visas_needed'] = answers[1]
                st.session_state.profile['business_description'] = answers[2]
                st.session_state.profile['experience'] = interpret_experience(answers[3])
                st.session_state.profile['flexibility'] = interpret_flexibility(answers[4])
                st.session_state.profile['purpose'] = answers[5]
                st.session_state.profile['timeline'] = answers[6]
                
                st.session_state.step = 'persona'
                st.rerun()
            else:
                st.error("Please answer all questions before continuing.")

# Persona Selection Step
elif st.session_state.step == 'persona':
    st.markdown('<div class="section-header">🎯 Persona Selection</div>', unsafe_allow_html=True)
    
    st.write("Which persona best fits this customer?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🏠 Residential\n\n(Visa/Residency Focused)", use_container_width=True, help="Customer primarily wants visa and residency"):
            st.session_state.profile['persona'] = "Residential"
            st.session_state.step = 'persona_questions'
            st.rerun()
    
    with col2:
        if st.button("💼 Business\n\n(Genuine Entrepreneur)", use_container_width=True, help="Customer has real business plans"):
            st.session_state.profile['persona'] = "Business"
            st.session_state.step = 'persona_questions'
            st.rerun()
    
    with col3:
        if st.button("💰 Finance\n\n(Banking/Tax Optimization)", use_container_width=True, help="Customer focused on banking and tax benefits"):
            st.session_state.profile['persona'] = "Finance"
            st.session_state.step = 'persona_questions'
            st.rerun()

# Persona-Specific Questions
elif st.session_state.step == 'persona_questions':
    persona = st.session_state.profile['persona']
    st.markdown(f'<div class="section-header">{persona} Persona - Follow-up Questions</div>', unsafe_allow_html=True)
    
    with st.form("persona_questions"):
        if persona == "Residential":
            q1 = st.text_area("Q1: Do you wish to get any dependents (family)?", height=80)
            q2 = st.text_area("Q2: Do you plan to reside in UAE, or will you be travelling frequently?", height=80)
            
            submitted = st.form_submit_button("Generate Recommendations", type="primary", use_container_width=True)
            
            if submitted and q1 and q2:
                st.session_state.profile['persona_answers']['dependents'] = concise_summary(q1)
                st.session_state.profile['persona_answers']['residency_plan'] = concise_summary(q2)
                st.session_state.step = 'loading'
                st.rerun()
                
        elif persona == "Business":
            q1 = st.text_area("Q1: What detailed activities do you want to start and what is your business model?", height=120)
            
            submitted = st.form_submit_button("Generate Recommendations", type="primary", use_container_width=True)
            
            if submitted and q1:
                st.session_state.profile['persona_answers']['business_model'] = concise_summary(q1, max_words=20)
                # CRITICAL: Also append to business description - EXACT from chatbot.py
                st.session_state.profile['business_description'] += f" | {q1}"
                st.session_state.step = 'loading'
                st.rerun()
                
        elif persona == "Finance":
            q1 = st.text_area("Q1: How will you invoice your clients and take payments?", height=80)
            q2 = st.text_area("Q2: Are you just planning to open a bank account to receive global payments?", height=80)
            q3 = st.text_area("Q3: How do you plan to get tax benefits?", height=80)
            
            submitted = st.form_submit_button("Generate Recommendations", type="primary", use_container_width=True)
            
            if submitted and q1 and q2 and q3:
                st.session_state.profile['persona_answers']['invoicing'] = concise_summary(q1)
                st.session_state.profile['persona_answers']['bank_purpose'] = concise_summary(q2)
                st.session_state.profile['persona_answers']['tax_strategy'] = concise_summary(q3)
                st.session_state.step = 'loading'
                st.rerun()

# Loading Step
elif st.session_state.step == 'loading':
    st.markdown('<div class="section-header">🔄 Generating Recommendations</div>', unsafe_allow_html=True)
    
    with st.spinner("Analyzing customer requirements across all knowledge sources..."):
        try:
            recommendations = get_activity_recommendations(st.session_state.profile)
            st.session_state.recommendations = recommendations
            st.session_state.step = 'results'
            st.rerun()
        except Exception as e:
            st.error(f"Error generating recommendations: {str(e)}")
            if st.button("Try Again"):
                st.session_state.step = 'persona_questions'
                st.rerun()

# Results Step
elif st.session_state.step == 'results':
    # Display Customer Profile
    st.markdown('<div class="section-header">👤 Customer Profile Summary</div>', unsafe_allow_html=True)
    
    profile_data = {
        "Field": [
            "Number of Shareholders",
            "Nationalities",
            "Number of Visas",
            "Business Description",
            "Branch or New",
            "Business Flexibility",
            "Purpose of Establishing",
            "Timeline",
            "Persona"
        ],
        "Value": [
            st.session_state.profile['shareholders'],
            st.session_state.profile['nationalities'],
            st.session_state.profile['visas_needed'],
            st.session_state.profile['business_description'][:100] + "..." if len(st.session_state.profile['business_description']) > 100 else st.session_state.profile['business_description'],
            st.session_state.profile['experience'],
            st.session_state.profile['flexibility'],
            st.session_state.profile['purpose'],
            st.session_state.profile['timeline'],
            st.session_state.profile['persona']
        ]
    }
    
    st.dataframe(profile_data, use_container_width=True, hide_index=True)
    
    # Persona-specific details
    persona = st.session_state.profile['persona']
    if st.session_state.profile['persona_answers']:
        with st.expander(f"📝 {persona} Persona Details"):
            if persona == "Residential":
                st.write(f"**Dependents:** {st.session_state.profile['persona_answers'].get('dependents', 'N/A')}")
                st.write(f"**Residency Plan:** {st.session_state.profile['persona_answers'].get('residency_plan', 'N/A')}")
            elif persona == "Business":
                st.write(f"**Business Model:** {st.session_state.profile['persona_answers'].get('business_model', 'N/A')}")
            elif persona == "Finance":
                st.write(f"**Invoicing:** {st.session_state.profile['persona_answers'].get('invoicing', 'N/A')}")
                st.write(f"**Bank Purpose:** {st.session_state.profile['persona_answers'].get('bank_purpose', 'N/A')}")
                st.write(f"**Tax Strategy:** {st.session_state.profile['persona_answers'].get('tax_strategy', 'N/A')}")
    
    # Display Recommendations
    st.markdown('<div class="section-header">🎯 Business Activity Recommendations</div>', unsafe_allow_html=True)
    
    st.markdown(f"```\n{st.session_state.recommendations}\n```")
    
    # Chat Interface
    st.markdown('<div class="section-header">💬 Ask Follow-up Questions</div>', unsafe_allow_html=True)
    
    st.info("💡 You can ask questions about activities, update customer information, or request alternatives. Type 'refresh' to regenerate recommendations.")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Type your question or update here..."):
        # Add user message to chat
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Check if it's a refresh request
        if user_input.lower() == 'refresh':
            with st.spinner("Regenerating recommendations..."):
                try:
                    recommendations = get_activity_recommendations(st.session_state.profile)
                    st.session_state.recommendations = recommendations
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": "✅ Recommendations updated! Scroll up to see the new results."
                    })
                except Exception as e:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": f"❌ Error: {str(e)}"
                    })
        else:
            # Check if it's a field update
            is_update = update_field(st.session_state.profile, user_input)
            
            if is_update:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "✅ Profile updated! Type 'refresh' to see updated recommendations."
                })
            else:
                # Generate response using EXACT logic from chatbot.py
                with st.spinner("Thinking..."):
                    try:
                        query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)
                        
                        context = f"""
Customer context: 
- Persona: {st.session_state.profile['persona']}
- Business: {st.session_state.profile['business_description']}
- Nationalities: {st.session_state.profile['nationalities']}

Question: {user_input}

Provide a clear, helpful answer based on the knowledge sources (Business Activities, Activity Hubs, MFZ Knowledge Base).
Use Mike's sales expertise for strategic guidance. Be consultative and honest.
"""
                        response = query_engine.query(context)
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": response.response
                        })
                    except Exception as e:
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": f"❌ Error: {str(e)}"
                        })
        
        st.rerun()
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Start New Assessment", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    with col2:
        if st.button("♻️ Regenerate Recommendations", use_container_width=True):
            st.session_state.step = 'loading'
            st.rerun()
    
    with col3:
        st.download_button(
            label="📥 Download Report",
            data=f"Customer Profile:\n{st.session_state.profile}\n\nRecommendations:\n{st.session_state.recommendations}",
            file_name="customer_recommendations.txt",
            mime="text/plain",
            use_container_width=True
        )

# Sidebar info
with st.sidebar:
    st.markdown("### 📊 Progress")
    steps = ["Welcome", "Questions", "Persona", "Follow-up", "Results"]
    current_step_index = {
        'welcome': 0,
        'questions': 1,
        'persona': 2,
        'persona_questions': 3,
        'loading': 4,
        'results': 4
    }.get(st.session_state.step, 0)
    
    for i, step in enumerate(steps):
        if i < current_step_index:
            st.markdown(f"✅ {step}")
        elif i == current_step_index:
            st.markdown(f"▶️ **{step}**")
        else:
            st.markdown(f"⚪ {step}")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.write("Meydan Free Zone Sales Assistant helps identify optimal business activities for customers.")
    st.write("**Knowledge Base:** 2,267+ activities")
    st.write("**Model:** GPT-4o")
    st.write("**Index:** business_activity_intelligence")