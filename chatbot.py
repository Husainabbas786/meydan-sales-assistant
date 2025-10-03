import os
from dotenv import load_dotenv
from llama_cloud_services import LlamaCloudIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.prompts import PromptTemplate

load_dotenv()

# Initialize
print("Initializing Meydan Free Zone Sales Assistant...")
index = LlamaCloudIndex(
    name="business_activity_intelligence",
    project_name="Default",
    organization_id=os.getenv("LLAMA_CLOUD_ORGANIZATION_ID"),
    api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
)

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = OpenAI(model="gpt-4o", temperature=0.1)

# System Prompt
SYSTEM_PROMPT = """You are an expert Meydan Free Zone business activity consultant with comprehensive knowledge of 2,267 business activities across multiple sources.

KNOWLEDGE SOURCES:
1. Business Activities Database (2,267 activities) - Activity codes, risk ratings, third-party approvals, descriptions, keywords, related activities
2. Activity Hubs Knowledge Base (151 detailed guides) - Expert insights for popular activities, strategic guidance, common scenarios
3. MFZ Business Activities Knowledge Base (Mike's sales expertise) - Strategic approaches, gray areas, banking considerations, jurisdiction rules
4. Country Risk Rating Database - Nationality risk ratings for Finance persona logic

PRIORITIZATION ALGORITHM (Applied with Persona-Specific Weights):
1. Correlation Accuracy: Minimum 90% semantic match with customer business description
2. Risk Rating Priority: Low → Medium → High (only when necessary)
3. Third-Party Approval: Prioritize N/A approval activities (avoid PRE/POST when alternatives exist)
4. Group Minimization: Recommend fewer activity groups when suitable

PERSONA-SPECIFIC WEIGHTS:

Business Persona (Genuine Entrepreneurs):
- Correlation: 60% weight (STRICT 90%+ match required)
- Risk Rating: 25% weight
- Third-Party Approval: 15% weight
- Logic: Business owners need exact activity match - prioritize correlation above all

Residential Persona (Visa/Residency Focused):
- Risk Rating: 40% weight
- Third-Party Approval: 40% weight
- Correlation: 20% weight (70-80% match acceptable)
- Logic: Visa seekers need easy approvals - prioritize low risk and no third-party

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
Step 3: Search MFZ Knowledge Base for strategic insights (top_k=2)
Step 4: For Finance persona, query Country Risk Rating database

SYNTHESIS APPROACH:
1. If Activity Hubs has high-confidence match (>0.75 similarity) → Use as primary for expert insights
2. Cross-reference Business Activities for: Code, Risk Rating, Third Party, When, Related Activities
3. Enrich with MFZ Knowledge Base strategic guidance (Mike's expertise on general trading, approvals, banking, etc.)
4. For Finance: Apply Country Risk compatibility check first

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
Provide exactly 3 ranked activity recommendations. For each activity:
- Activity Code (6-digit format like 1811.04)
- Activity Name
- Category (e.g., Manufacturing, Trading, Professional)
- Group (3-digit code)
- Full Description from database
- Third Party Approval: Yes/No [Authority name if yes, e.g., "Dubai Municipality (DM)"]
- When: PRE/POST/N/A
- Risk Rating: Low/Medium/High
- Industry Risk: Yes/No/N/A
- Match Explanation: Why this fits customer needs (2-3 sentences with persona logic applied)
- Related Activities: 2-3 complementary activities (with codes, names, and 1-line descriptions)
- Expert Insights: Include strategic guidance from Activity Hubs or MFZ Knowledge Base if available

CRITICAL RULES:
1. For Business persona: Never compromise on correlation - must be 90%+ match
2. For Residential persona: Prefer N/A approvals and Low risk even if correlation is 70-80%
3. For Finance persona: Check nationality risk first - if Override, immediately respond "Cannot issue license due to country risk rating"
4. Always suggest fewer activity groups when possible (maximize 3-group package value)
5. Flag general trading concerns and suggest specific alternatives (Mike's strategy)
6. Explain any approval delays or banking complications transparently
7. Use Activity Hubs content for popular activities (e-commerce, general trading, management consultancy, IT, advertising, holding companies, etc.)

Be precise, strategic, and consultative. Ensure recommendations maximize customer success while adhering to regulations."""

# Customer profile storage
customer_profile = {
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

# Questions
initial_questions = [
    "What are the number of shareholders and what passport holders are they?",
    "How many visas do you want with this company?",
    "What business do you want to do?",
    "Have you been doing this business or is it a new venture?",
    "Are you open to do something else also or stick to your business plan?",
    "What is your primary purpose in establishing a company in UAE - Dubai?",
    "How soon are you planning to set up the company?"
]

persona_questions = {
    "Residential": [
        "Do you wish to get any dependents (family)?",
        "Do you plan to reside in UAE, or will you be travelling frequently?"
    ],
    "Business": [
        "What detailed activities do you want to start and what is your business model?"
    ],
    "Finance": [
        "How will you invoice your clients and take payments?",
        "Are you just planning to open a bank account to receive global payments?",
        "How do you plan to get tax benefits?"
    ]
}

def parse_nationalities(shareholder_answer):
    """Extract nationalities from shareholder answer"""
    # Simple extraction - GPT will help parse this better in production
    return shareholder_answer

def interpret_experience(answer):
    """Convert experience answer to Branch/New"""
    answer_lower = answer.lower()
    if any(word in answer_lower for word in ["new", "starting", "first time", "venture"]):
        return "New"
    elif any(word in answer_lower for word in ["branch", "existing", "expand", "already"]):
        return "Branch"
    return answer  # Return as-is if unclear

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
Apply weights: Risk (40%) + Third-Party Approval (40%) + Correlation (20%)
Accept 70-80% correlation if it means Low risk and N/A approval
"""
    elif persona == "Business":
        query_context += f"""
BUSINESS PERSONA CONTEXT:
- Detailed Business Model: {profile['persona_answers'].get('business_model', 'N/A')}

PRIORITIZATION FOR THIS PERSONA:
Apply weights: Correlation (60%) + Risk (25%) + Third-Party Approval (15%)
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
   - Semantic correlation strength (90%+ for Business, 70%+ for Residential/Finance)
   - Risk rating (Low preferred, High only when necessary)
   - Third-party approval status (N/A preferred)
   - Group optimization (fewer groups better)
6. Consider Mike's strategic insights (avoid general trading, suggest specific alternatives, flag banking challenges)
7. Validate final recommendations against decision matrix

DELIVERABLE:
Provide exactly 3 ranked activity recommendations following the format:

RECOMMENDATION 1: [Primary Recommendation]
Activity Code: [6-digit code]
Activity Name: [Full name]
Category: [Category]
Group: [3-digit group]
Description: [Full description from database]
Third Party Approval: [Yes/No] [Authority if yes]
When: [PRE/POST/N/A]
Risk Rating: [Low/Medium/High]
Industry Risk: [Yes/No/N/A]
Match Explanation: [2-3 sentences explaining why this fits with persona logic applied]
Related Activities:
  - [Code]: [Name] - [1-line description]
  - [Code]: [Name] - [1-line description]
Expert Insights: [Strategic guidance from Activity Hubs or MFZ Knowledge Base if available]

[Repeat for RECOMMENDATION 2 and 3]
"""
    
    # Retrieve with appropriate top_k based on persona
    top_k = 15 if persona == "Business" else 10
    
    retriever = index.as_retriever(similarity_top_k=top_k)
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=top_k)
    
    print("\n[Applying chain-of-thought reasoning across all knowledge sources...]")
    response = query_engine.query(query_context)
    
    return response.response

def print_summary_tables(profile, recommendations=None):
    """Print both summary tables"""
    
    print("\n" + "="*100)
    print(" "*35 + "CUSTOMER SUMMARY & RECOMMENDATIONS")
    print("="*100)
    
    # TABLE 1: Customer Profile
    print("\n" + "─"*100)
    print("TABLE 1: CUSTOMER PROFILE")
    print("─"*100)
    
    print(f"\n{'Field':<30} {'Value':<70}")
    print("─"*100)
    print(f"{'Number of Shareholders':<30} {profile['shareholders']:<70}")
    print(f"{'Nationalities':<30} {profile['nationalities']:<70}")
    print(f"{'Number of Visas':<30} {profile['visas_needed']:<70}")
    print(f"{'Business Description':<30} {profile['business_description']:<70}")
    print(f"{'Branch or New':<30} {profile['experience']:<70}")
    print(f"{'Business Flexibility':<30} {profile['flexibility']:<70}")
    print(f"{'Purpose of Establishing':<30} {profile['purpose']:<70}")
    print(f"{'Timeline':<30} {profile['timeline']:<70}")
    print(f"{'Persona':<30} {profile['persona']:<70}")
    
    # Add persona-specific answers
    if profile['persona'] == "Residential" and profile['persona_answers']:
        print("\n" + "─"*100)
        print("RESIDENTIAL PERSONA DETAILS:")
        print("─"*100)
        print(f"{'Dependents':<30} {profile['persona_answers'].get('dependents', 'N/A'):<70}")
        print(f"{'Residency Plan':<30} {profile['persona_answers'].get('residency_plan', 'N/A'):<70}")
    
    elif profile['persona'] == "Business" and profile['persona_answers']:
        print("\n" + "─"*100)
        print("BUSINESS PERSONA DETAILS:")
        print("─"*100)
        print(f"{'Business Model Details':<30} {profile['persona_answers'].get('business_model', 'N/A'):<70}")
    
    elif profile['persona'] == "Finance" and profile['persona_answers']:
        print("\n" + "─"*100)
        print("FINANCE PERSONA DETAILS:")
        print("─"*100)
        print(f"{'Invoicing Method':<30} {profile['persona_answers'].get('invoicing', 'N/A'):<70}")
        print(f"{'Bank Account Purpose':<30} {profile['persona_answers'].get('bank_purpose', 'N/A'):<70}")
        print(f"{'Tax Strategy':<30} {profile['persona_answers'].get('tax_strategy', 'N/A'):<70}")
    
    # TABLE 2: Activity Recommendations
    if recommendations:
        print("\n" + "="*100)
        print("TABLE 2: BUSINESS ACTIVITY RECOMMENDATIONS")
        print("="*100 + "\n")
        print(recommendations)
    
    print("\n" + "="*100 + "\n")

def update_field(profile, field_update):
    """Update customer profile field based on conversational input"""
    field_lower = field_update.lower()
    
    # Parse updates
    if "shareholder" in field_lower:
        # Extract number and update
        words = field_update.split()
        for word in words:
            if word.isdigit():
                profile['shareholders'] = word
                print(f"[Updated: Number of Shareholders = {word}]")
                return True
    
    elif "visa" in field_lower:
        words = field_update.split()
        for word in words:
            if word.isdigit():
                profile['visas_needed'] = word
                print(f"[Updated: Number of Visas = {word}]")
                return True
    
    elif "business" in field_lower or "activity" in field_lower:
        # Update business description
        profile['business_description'] = field_update
        print(f"[Updated: Business Description]")
        return True
    
    elif "nationality" in field_lower or "passport" in field_lower:
        profile['nationalities'] = field_update
        print(f"[Updated: Nationalities]")
        return True
    
    elif "timeline" in field_lower:
        profile['timeline'] = field_update
        print(f"[Updated: Timeline]")
        return True
    
    return False

def run_chatbot():
    """Main chatbot flow"""
    
    print("\n" + "="*100)
    print(" "*25 + "MEYDAN FREE ZONE SALES ASSISTANT")
    print("="*100)
    print("\nI'll help you identify the best business activities for your customer.")
    print("Let's start by gathering some information.\n")
    
    # Ask initial questions
    keys = ["shareholders", "visas_needed", "business_description", "experience", 
            "flexibility", "purpose", "timeline"]
    
    for i, question in enumerate(initial_questions):
        print(f"\nQ{i+1}: {question}")
        answer = input("Answer: ").strip()
        
        if i == 0:  # Shareholders question
            customer_profile['shareholders'] = answer.split()[0] if answer.split()[0].isdigit() else "Not specified"
            customer_profile['nationalities'] = answer
        elif i == 3:  # Experience
            customer_profile['experience'] = interpret_experience(answer)
        elif i == 4:  # Flexibility
            customer_profile['flexibility'] = interpret_flexibility(answer)
        else:
            customer_profile[keys[i]] = answer
    
    # Persona selection
    print("\nQ8: Which persona best fits this customer?")
    print("  a. Residential (visa/residency focused)")
    print("  b. Business (genuine entrepreneur)")
    print("  c. Finance (banking/tax optimization)")
    
    persona_choice = input("Select (a/b/c): ").strip().lower()
    
    if persona_choice == 'a':
        customer_profile['persona'] = "Residential"
    elif persona_choice == 'b':
        customer_profile['persona'] = "Business"
    elif persona_choice == 'c':
        customer_profile['persona'] = "Finance"
    else:
        print("Invalid choice. Defaulting to Business persona.")
        customer_profile['persona'] = "Business"
    
    print(f"\n[Persona Identified: {customer_profile['persona']}]")
    
    # Ask persona-specific questions
    print(f"\n--- {customer_profile['persona'].upper()} PERSONA FOLLOW-UP QUESTIONS ---")
    
    persona = customer_profile['persona']
    questions = persona_questions[persona]
    
    if persona == "Residential":
        print(f"\nQ1: {questions[0]}")
        ans1 = input("Answer: ").strip()
        customer_profile['persona_answers']['dependents'] = concise_summary(ans1)
        
        print(f"\nQ2: {questions[1]}")
        ans2 = input("Answer: ").strip()
        customer_profile['persona_answers']['residency_plan'] = concise_summary(ans2)
    
    elif persona == "Business":
        print(f"\nQ1: {questions[0]}")
        ans1 = input("Answer: ").strip()
        customer_profile['persona_answers']['business_model'] = concise_summary(ans1, max_words=20)
        # Also append to business description for better correlation
        customer_profile['business_description'] += f" | {ans1}"
    
    elif persona == "Finance":
        print(f"\nQ1: {questions[0]}")
        ans1 = input("Answer: ").strip()
        customer_profile['persona_answers']['invoicing'] = concise_summary(ans1)
        
        print(f"\nQ2: {questions[1]}")
        ans2 = input("Answer: ").strip()
        customer_profile['persona_answers']['bank_purpose'] = concise_summary(ans2)
        
        print(f"\nQ3: {questions[2]}")
        ans3 = input("Answer: ").strip()
        customer_profile['persona_answers']['tax_strategy'] = concise_summary(ans3)
    
    # Generate initial recommendations
    print("\n[Analyzing customer requirements across all knowledge sources...]")
    print("[Applying persona-specific prioritization logic...]")
    print("[Searching 2,267 activities + expert insights...]")
    
    recommendations = get_activity_recommendations(customer_profile)
    
    # Display summary tables
    print_summary_tables(customer_profile, recommendations)
    
    # Interactive conversation
    print("\n" + "─"*100)
    print("You can now:")
    print("1. Ask questions about activities, services, pricing, or regulations")
    print("2. Update any customer information (e.g., 'customer now wants 5 visas')")
    print("3. Request alternative activities")
    print("Type 'refresh' to regenerate recommendations with updated info")
    print("Type 'done' to end conversation")
    print("─"*100 + "\n")
    
    query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)
    
    while True:
        user_input = input("\nYour input: ").strip()
        
        if user_input.lower() == 'done':
            print("\nThank you for using Meydan Free Zone Sales Assistant!")
            break
        
        elif user_input.lower() == 'refresh':
            print("\n[Regenerating recommendations with updated information...]")
            recommendations = get_activity_recommendations(customer_profile)
            print_summary_tables(customer_profile, recommendations)
        
        else:
            # Check if this is a field update
            is_update = update_field(customer_profile, user_input)
            
            if is_update:
                print("Type 'refresh' to see updated recommendations, or continue asking questions.")
            else:
                # General Q&A
                context = f"""
Customer context: 
- Persona: {customer_profile['persona']}
- Business: {customer_profile['business_description']}
- Nationalities: {customer_profile['nationalities']}

Question: {user_input}

Provide a clear, helpful answer based on the knowledge sources (Business Activities, Activity Hubs, MFZ Knowledge Base).
Use Mike's sales expertise for strategic guidance. Be consultative and honest.
"""
                response = query_engine.query(context)
                print(f"\nAnswer: {response.response}")

if __name__ == "__main__":
    run_chatbot()