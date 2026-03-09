import requests
import urllib3

# disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def geocode_location(location_text: str):

    if not location_text or not location_text.strip():
        return None, None

    try:
        url = "https://nominatim.openstreetmap.org/search"

        params = {
            "q": location_text + ", India",
            "format": "json",
            "limit": 1
        }

        headers = {
            "User-Agent": "ai_medical_bot"
        }

        response = requests.get(
            url,
            params=params,
            headers=headers,
            verify=False   # IMPORTANT: disables SSL verification
        )

        data = response.json()

        if len(data) > 0:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])

            print("Location found:", lat, lon)
            return lat, lon

        print("Location not found")

    except Exception as e:
        print("Geocoding error:", e)

    return None, None