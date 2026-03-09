from predict_specialist import predict_specialist
from datetime import datetime

# ===============================
# Greeting
# ===============================

def greet_user():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    elif hour < 18:
        return "Good afternoon"
    else:
        return "Good evening"


# ===============================
# Emergency Keywords
# ===============================

EMERGENCY_KEYWORDS = [
    "severe chest pain",
    "cannot breathe",
    "unconscious",
    "stroke",
    "bleeding heavily"
]

# ===============================
# Follow-up Rules
# ===============================

FOLLOW_UP_RULES = {
    "chest pain": {
        "question": "Was your last meal heavy or something gastric for you?",
        "yes_specialist": "General Medicine",
        "no_specialist": "Cardiology"
    },
    "headache": {
        "question": "Was it caused by stress, lack of sleep, or dehydration?",
        "yes_specialist": "General Medicine",
        "no_specialist": "Neurology"
    }
}

# ===============================
# In-Memory Session Storage
# ===============================

conversation_state = {}

# ===============================
# Main Handler
# ===============================

def handle_symptoms(user_input: str, user_id: str):

    text = user_input.lower().strip()

    # 1️⃣ Emergency detection
    for keyword in EMERGENCY_KEYWORDS:
        if keyword in text:
            return {
                "type": "Emergency",
                "message": "This may be serious. Please go to the nearest hospital immediately."
            }

    # 2️⃣ Follow-up answer handling
    if user_id in conversation_state:
        rule = conversation_state[user_id]
        del conversation_state[user_id]

        if "yes" in text:
            specialist = rule["yes_specialist"]
        else:
            specialist = rule["no_specialist"]

        return {
            "type": "Specialist Recommendation",
            "specialist": specialist,
            "message": f"You should consult {specialist}."
        }

    # 3️⃣ Check if follow-up needed
    for symptom, rule in FOLLOW_UP_RULES.items():
        if symptom in text:
            conversation_state[user_id] = rule
            return {
                "type": "Follow-Up Question",
                "question": rule["question"]
            }

    # 4️⃣ Normal prediction
    specialist = predict_specialist(text)

    return {
        "type": "Specialist Recommendation",
        "specialist": specialist,
        "message": f"You should consult {specialist}."
    }