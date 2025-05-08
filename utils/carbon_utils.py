"""
carbon_utils.py

This module provides utilities to estimate carbon emissions associated with
energy usage during model inference. It includes functionality to:

- Retrieve real-time carbon intensity based on geolocation using the
  Electricity Maps API.
- Convert energy consumption in joules to CO2 emissions in grams.

Functions:
- get_carbon_intensity: Query Electricity Maps API for current gCO2eq/kWh.
- joules_to_carbon: Convert energy (J) to carbon emissions (gCO2eq).
"""

import geocoder
import requests


def get_carbon_intensity(api_key):
    """
    Get real-time carbon intensity (in gCO2eq/kWh) based on current IP geolocation.

    Uses the Electricity Maps API to determine the carbon footprint of electricity
    at the user's approximate physical location.

    Args:
        api_key (str): API key for the Electricity Maps service.

    Returns:
        float: Carbon intensity in grams CO2 equivalent per kilowatt-hour.

    Raises:
        Exception: If geolocation or API call fails.
    """
    g = geocoder.ip("me")
    if not g.latlng:
        raise Exception("Unable to retrieve geolocation from IP")

    latitude, longitude = g.latlng

    url = "https://api.electricitymap.org/v3/carbon-intensity/latest"
    headers = {"auth-token": api_key}
    params = {"lat": latitude, "lon": longitude}

    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        data = response.json()
        return data.get("carbonIntensity")  # in grams CO2 per kWh
    else:
        print("Error:", response.status_code, response.text)
        raise Exception("Failed to fetch carbon intensity from API")


def joules_to_carbon(energy_joules, carbon_intensity_g_per_kwh):
    """
    Convert energy usage in joules to carbon emissions in grams CO2 equivalent.

    Args:
        energy_joules (float): Energy consumed in joules.
        carbon_intensity_g_per_kwh (float): Carbon intensity (gCO2eq/kWh).

    Returns:
        float: Estimated CO2 emissions in grams.
    """
    energy_kwh = energy_joules / 3.6e6  # 1 kWh = 3.6 million joules
    return round(energy_kwh * carbon_intensity_g_per_kwh, 6)
