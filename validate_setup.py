import vertexai
from vertexai.generative_models import GenerativeModel

# Init
vertexai.init(
    project="rag-agent-lab-1772881421",
    location="us-central1"
)

# Use versioned model string — always more reliable than alias
model = GenerativeModel("gemini-2.0-flash-001")
resp = model.generate_content("Say: VERTEX AI CONNECTED")
print("✅ Vertex AI connected:", resp.text)
