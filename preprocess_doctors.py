import pandas as pd

# =========================
# 1. Load doctor dataset
# =========================
df = pd.read_excel("data/doctor_dataset.xlsx")

print("Original Columns:")
print(df.columns)

# =========================
# 2. Rename columns (standard ML-friendly names)
# =========================
df.rename(columns={
    'Doctor / Clinic Name': 'doctor_name',
    'Speciality': 'speciality',
    'Area': 'area',
    'Latitude': 'latitude',
    'Longitude': 'longitude',
    'Fees (₹)': 'fees',
    'Rating': 'rating',
    'Availability': 'availability',
    'Contact Number': 'contact',
    'Address': 'address'
}, inplace=True)

# =========================
# 3. Remove unwanted Excel columns
# =========================
df = df.loc[:, ~df.columns.str.contains('unnamed', case=False)]

# =========================
# 4. Clean text columns
# =========================
df['speciality'] = df['speciality'].str.lower().str.strip()
df['area'] = df['area'].str.lower().str.strip()

# =========================
# 5. Handle missing numeric values
# =========================
df['fees'] = df['fees'].fillna(df['fees'].median())
df['rating'] = df['rating'].fillna(df['rating'].median())

# =========================
# 6. AVAILABILITY (IMPORTANT PART 🔥)
# =========================
# Keep original availability text for patient display
df['availability_text'] = df['availability']

# Create binary flag for ML (1 = available, 0 = not available)
df['availability_flag'] = df['availability'].apply(
    lambda x: 0 if pd.isna(x) or str(x).strip() == "" else 1
)

# Drop original availability column
df.drop(columns=['availability'], inplace=True)

# =========================
# 7. Drop rows without location
# =========================
df = df.dropna(subset=['latitude', 'longitude'])

# =========================
# 8. Convert datatypes
# =========================
df['fees'] = df['fees'].astype(float)
df['rating'] = df['rating'].astype(float)
df['availability_flag'] = df['availability_flag'].astype(int)

# =========================
# 9. Save cleaned dataset
# =========================
df.to_csv("data/clean_doctor_dataset.csv", index=False)

print("✅ STEP 1 COMPLETE: clean_doctor_dataset.csv created successfully")
