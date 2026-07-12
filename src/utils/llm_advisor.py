import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from utils.knowledge_base import DISEASE_INFO, NO_DISEASE_INFO

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.3  # lower temperature = more consistent, less "creative" for medical-adjacent content
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful skin health assistant. You will be given structured information 
about a skin condition that an AI model has predicted from an image, with a confidence score.

IMPORTANT RULES:
- This is an AI PREDICTION, not a medical diagnosis. Never use the words "diagnosed" or "diagnosis" — 
  instead say "the AI model predicted..." or "this may indicate...".
- Only use the information provided below. Do NOT add any medical facts, statistics, or claims not present in the input.
- Always include a clear disclaimer that this is not a substitute for professional medical diagnosis, 
  and that a dermatologist should confirm any finding.
- Be supportive but not alarmist, even for serious conditions.
- Structure your response with clear sections: What it is, Recommendations, Diet suggestions."""),
    ("user", """Detected condition: {condition_name}
Confidence: {confidence}

Description: {description}

Recommendations:
{recommendations}

Diet suggestions:
{diet}

Please present this clearly for the patient.""")
])

def get_advice(predicted_class: str, confidence: float, no_disease: bool = False) -> str:
    if no_disease:
        chain = prompt_template | llm
        response = chain.invoke({
            "condition_name": "No disease detected",
            "confidence": confidence,
            "description": NO_DISEASE_INFO["message"],
            "recommendations": "\n".join(f"- {r}" for r in NO_DISEASE_INFO["recommendations"]),
            "diet": "No specific diet recommendations needed."
        })
        return response.content

    info = DISEASE_INFO.get(predicted_class)
    if not info:
        return "Unable to generate advice: unknown condition class."

    chain = prompt_template | llm
    response = chain.invoke({
        "condition_name": info["full_name"],
        "confidence": confidence,
        "description": info["description"],
        "recommendations": "\n".join(f"- {r}" for r in info["recommendations"]),
        "diet": "\n".join(f"- {d}" for d in info["diet"])
    })
    return response.content