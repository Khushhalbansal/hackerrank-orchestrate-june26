import os
from google import genai

# Set via environment variable -- never hardcode:
#   Windows: set GEMINI_API_KEY=your_key_here
#   macOS/Linux: export GEMINI_API_KEY=your_key_here
API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set.")

client = genai.Client(api_key=API_KEY)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Reply with exactly: Gemini Working"
)

print(response.text)
