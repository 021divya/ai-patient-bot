import pandas as pd
from predict_specialist import predict_specialist

# =========================
# Load cleaned doctor dataset
# =========================
doctor_df = pd.read_csv("data/clean_doctor_dataset.csv")

def get_doctors_for_patient(symptoms_text):
    """
    1. Predict specialist using ML model
    2. Filter doctors by that specialist
    """
    # 🔹 ML prediction
    specialist = predict_specialist(symptoms_text).lower()
    print(f"🔮 Predicted Specialist: {specialist}")


    # 🔹 Filter doctors
    filtered_doctors = doctor_df[
        doctor_df['speciality'] == specialist
    ]

    return filtered_doctors


# =========================
# TEST
# =========================
if __name__ == "__main__":
    symptoms = "chest pain and breathlessness"

    doctors = get_doctors_for_patient(symptoms)

    print("\n🩺 Doctors matching your symptoms:\n")
    print(
        doctors[
            [
                "doctor_name",
                "speciality",
                "area",
                "rating",
                "fees",
                "availability_text"
            ]
        ].head()
    )
