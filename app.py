import os
import json

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# =========================================================
# 1) LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")


# =========================================================
# 2) LOAD PROMPT CONFIG
# =========================================================

with open("prompt.json", "r", encoding="utf-8") as file:
    prompt_config = json.load(file)


# =========================================================
# 3) CREATE HUGGING FACE CLIENT
# =========================================================

client = InferenceClient(
    provider="auto",
    api_key=HF_TOKEN
)


# =========================================================
# 4) GET USER INPUT
# =========================================================

customer_message = input(
    "\nEnter customer message:\n> "
)


# =========================================================
# 5) BUILD INSTRUCTIONS
# =========================================================

instructions = "\n".join(
    f"{index + 1}. {instruction}"
    for index, instruction in enumerate(
        prompt_config["instructions"]
    )
)


# =========================================================
# 6) CONVERT RULES TO JSON TEXT
# =========================================================

priority_rules = json.dumps(
    prompt_config["priority_rules"],
    indent=2
)


output_format = json.dumps(
    prompt_config["output_format"],
    indent=2
)


# =========================================================
# 7) BUILD USER PROMPT
# =========================================================

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


# =========================================================
# 8) CALL THE LLM
# =========================================================

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


# =========================================================
# 9) GET MODEL OUTPUT
# =========================================================

model_output = response.choices[0].message.content


print("\nRAW LLM RESPONSE:")
print(model_output)


# =========================================================
# 10) PARSE JSON OUTPUT
# =========================================================

try:

    result = json.loads(model_output)

except json.JSONDecodeError:

    print("\nERROR:")
    print("The model did not return valid JSON.")

    print("\nRAW OUTPUT:")
    print(model_output)

    raise


# =========================================================
# 11) DISPLAY RESULT
# =========================================================

print("\n" + "=" * 50)

print("CUSTOMER ANALYSIS")

print("=" * 50)

print("Sentiment:", result["sentiment"])
print("Intent:", result["intent"])
print("Priority:", result["priority"])
print("Reason:", result["reason"])


# =========================================================
# 12) BUSINESS LOGIC
# =========================================================

priority = result["priority"].lower()


if priority == "high":

    print("\nACTION: Escalate ticket immediately.")


elif priority == "medium":

    print("\nACTION: Send ticket to support queue.")


else:

    print("\nACTION: Normal support queue.")


# =========================================================
# 13) SAVE RESULT TO JSON FILE
# =========================================================

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