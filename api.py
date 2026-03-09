from fastapi import FastAPI, Request
from pydantic import BaseModel

from bot_flow import greet_user, handle_symptoms
from recommend_doctors import recommend_doctors
from geocode_utils import geocode_location

app = FastAPI(
    title="AI Medical Assistant Bot",
    description="Symptom-based doctor recommendation system",
    version="2.0"
)


# =========================
# Request Models
# =========================

class SymptomRequest(BaseModel):
    symptoms: str


class FilterRequest(BaseModel):
    symptoms: str
    location_text: str
    max_distance_km: float
    max_fees: int
    min_rating: float


# =========================
# Greeting Endpoint
# =========================

@app.get("/greet")
def greet():

    return {
        "message": f"{greet_user()} 👋 I’m your AI medical assistant. Please tell me what symptoms you are experiencing."
    }


# =========================
# Symptom Endpoint
# =========================

@app.post("/symptoms")
def process_symptoms(data: SymptomRequest, request: Request):

    user_id = request.client.host

    result = handle_symptoms(data.symptoms, user_id=user_id)

    return result


# =========================
# Recommendation Endpoint
# =========================

@app.post("/recommend")
def recommend(data: FilterRequest, request: Request):

    user_id = request.client.host

    result = handle_symptoms(data.symptoms, user_id=user_id)

    # If follow-up question or emergency
    if result["type"] != "Specialist Recommendation":
        return result

    specialist = result["specialist"]

    lat, lng = geocode_location(data.location_text)

    if lat is None or lng is None:
        return {
            "message": "I couldn’t understand the location you entered.",
            "doctors": []
        }

    results = recommend_doctors(
        specialist=specialist,
        patient_lat=lat,
        patient_lng=lng,
        location_text=data.location_text,
        max_distance_km=data.max_distance_km,
        max_fees=data.max_fees,
        min_rating=data.min_rating
    )

    if results.empty:
        return {
            "message": "No doctors found matching your filters.",
            "doctors": []
        }

    doctors = results[
        [
            "doctor_name",
            "area",
            "distance_km",
            "rating",
            "fees",
            "contact",
            "address",
            "availability_text"
        ]
    ].to_dict(orient="records")

    return {
        "message": f"You should consult {specialist}. Here are the best doctors near you.",
        "doctors": doctors
    }


# =========================
# Reset Endpoint
# =========================

@app.post("/reset")
def reset():

    return {
        "message": "Conversation reset. Please tell me your symptoms."
    }