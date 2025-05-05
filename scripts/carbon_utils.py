import geocoder
import requests

def get_carbon_intensity(api_key):
    """
    Get real-time carbon intensity (in gCO2eq/kWh) using the Electricity Maps API based on your current IP location.
    """
    g = geocoder.ip('me')
    latitude = g.latlng[0]
    longitude = g.latlng[1]

    if not g.latlng:
        raise Exception("Unable to retrieve geolocation from IP")


    url = "https://api.electricitymap.org/v3/carbon-intensity/latest"
    headers = {"auth-token": api_key}
    params = {"lat": latitude, "lon": longitude}

    response = requests.get(url, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        return data.get("carbonIntensity")  # in grams CO2 per kWh
    else:
        print("Error:", response.status_code, response.text)
        raise Exception("Failed to fetch carbon intensity")

def joules_to_carbon(energy_joules, carbon_intensity_g_per_kwh):
    """
    Convert energy usage (in joules) to CO2 emissions (in grams).
    """
    energy_kwh = energy_joules / 3.6e6
    return round(energy_kwh * carbon_intensity_g_per_kwh, 6)