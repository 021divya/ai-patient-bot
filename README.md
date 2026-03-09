# AI Patient Triage & Doctor Recommendation Bot

## Overview

This project is an **AI-powered patient triage system** that analyzes patient symptoms and recommends the appropriate medical specialist. It also supports doctor recommendation based on location and specialization.

The goal of the system is to **automate the initial patient screening process** and help route patients to the correct specialist efficiently.

---

## Features

* Symptom → Specialist prediction using **Machine Learning**
* NLP-based symptom understanding using **Sentence Transformers**
* Doctor recommendation system
* Filtering doctors by specialization
* Distance-based doctor search
* Modular backend architecture for easy integration with frontend systems

---

## Project Structure

```
patient-bot
│
├── api.py                       # FastAPI backend
├── predict_specialist.py        # Specialist prediction module
├── train_specialist_model.py    # ML model training script
├── recommend_doctors.py         # Doctor recommendation logic
├── filter_doctors.py            # Filtering doctors by specialization
├── distance_utils.py            # Distance calculation utilities
├── geocode_utils.py             # Location geocoding
├── patient_input.py             # Patient symptom processing
│
├── data
│   ├── clean_doctor_dataset.csv
│   ├── doctor_dataset.xlsx
│   └── final_symptom_speciality_dataset.csv
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/021divya/ai-patient-bot.git
cd ai-patient-bot
```

Create virtual environment (optional):

```
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Train the ML Model

Before running the API, train the specialist prediction model:

```
python train_specialist_model.py
```

This will generate the trained model inside the **model/** directory.

---

## Run the Backend

Start the API server:

```
python api.py
```

or

```
uvicorn api:app --reload
```

---

## Technologies Used

* Python
* FastAPI
* Scikit-Learn
* Sentence Transformers
* Pandas
* Machine Learning
* NLP

