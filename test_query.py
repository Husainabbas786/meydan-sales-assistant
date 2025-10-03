

import os
from dotenv import load_dotenv
from llama_cloud_services import LlamaCloudIndex
from llama_index.llms.openai import OpenAI

# Load environment variables
load_dotenv()

print("🔄 Initializing connection to LlamaCloud...")

# Initialize LlamaCloud Index
try:
    index = LlamaCloudIndex(
        name="business_activities_index",
        project_name="Default",
        organization_id=os.getenv("LLAMA_CLOUD_ORGANIZATION_ID"),
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
    )
    print("✅ Connected to LlamaCloud index successfully!\n")
except Exception as e:
    print(f"❌ Error connecting to LlamaCloud: {e}")
    exit()

# Initialize OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
llm = OpenAI(model="gpt-4o-mini", temperature=0.1)

# Test query
test_query = "I want to start a printing business in Dubai"
print(f"📝 Query: {test_query}\n")
print("🔍 Searching business activities database...\n")

try:
    # Retrieve relevant activities
    retriever = index.as_retriever(similarity_top_k=3)
    nodes = retriever.retrieve(test_query)
    
    print("=" * 80)
    print("TOP 3 MATCHING BUSINESS ACTIVITIES")
    print("=" * 80)
    
    for i, node in enumerate(nodes, 1):
        print(f"\n--- Result {i} ---")
        print(f"Score: {node.score:.4f}")
        print(f"Content:\n{node.text[:500]}...")  # First 500 chars
        print("-" * 80)
    
    # Generate intelligent response using GPT
    print("\n\n🤖 GENERATING INTELLIGENT RECOMMENDATION...\n")
    
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(
        f"{test_query}. Please provide the activity code, name, third-party approvals, risk rating, and a brief description."
    )
    
    print("=" * 80)
    print("AI RECOMMENDATION")
    print("=" * 80)
    print(response.response)
    print("=" * 80)
    
except Exception as e:
    print(f"❌ Error during query: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ Test completed!")