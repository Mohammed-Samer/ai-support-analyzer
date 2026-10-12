import os
import json

from dotenv import load_dotenv
from groq import Groq


# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")


# Load prompt configuration
with open("prompt.json", "r", encoding="utf-8") as file:
    prompt_config = json.load(file)


# Create Groq client
client = Groq(
    api_key=GROQ_API_KEY
)


# Get user input
customer_message = input(
    "\nEnter customer message:\n> "
)


# Build instructions
instructions = "\n".join(
    f"{index + 1}. {instruction}"
    for index, instruction in enumerate(
        prompt_config["instructions"]
    )
)


# Convert JSON sections to text
priority_rules = json.dumps(
    prompt_config["priority_rules"],
    indent=2
)

output_format = json.dumps(
    prompt_config["output_format"],
    indent=2
)


# Build user prompt
user_prompt = f"""
TASK:
Analyze the following customer support message.

CUSTOMER MESSAGE:
<message>
{customer_message}
</message>

INSTRUCTIONS:
{instructions}

PRIORITY RULES:
{priority_rules}

CONSTRAINTS:
- Use only the information in the customer message.
- Do not invent missing information.
- Return only valid JSON.
- Do not add markdown.
- Do not add explanations outside the JSON.

OUTPUT FORMAT:
{output_format}
"""


# Call Groq LLM
response = client.chat.completions.create(

    model="openai/gpt-oss-20b",

    messages=[
        {
            "role": "system",
            "content": prompt_config["system_prompt"]
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ],

    temperature=0.2,
    max_tokens=300
)


# Get model output
model_output = response.choices[0].message.content


print("\nRAW LLM RESPONSE:")
print(model_output)


# Parse JSON
try:
    result = json.loads(model_output)

except json.JSONDecodeError:
    print("\nThe model did not return valid JSON.")
    print(model_output)
    raise


# Display result
print("\n" + "=" * 50)
print("CUSTOMER ANALYSIS")
print("=" * 50)

print("Sentiment:", result["sentiment"])
print("Intent:", result["intent"])
print("Priority:", result["priority"])
print("Reason:", result["reason"])


# Business logic
priority = result["priority"].lower()

if priority == "high":
    print("\nACTION: Escalate ticket immediately.")

elif priority == "medium":
    print("\nACTION: Send ticket to support queue.")

else:
    print("\nACTION: Normal support queue.")


# Save result
with open(
    "last_result.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        result,
        file,
        indent=2,
        ensure_ascii=False
    )


print("\nResult saved to last_result.json")