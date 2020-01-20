import requests
import env


def create_api_request(location):
    return f"http://weerlive.nl/api/json-data-10min.php?" \
           f"key={env.WEATHER_API_KEY}&locatie=={location}"


def construct_weather(data):
    return f"""Weather Info:
-------------------

    Temperature: {data["temp"]} (feels like {data["gtemp"]})
    Summary: {data["samenv"]}
    Todays range: {data["d0tmin"]}:{data["d0tmax"]}
    Chance of sun: {data["d0zon"]}%
    Chance of rain: {data["d0neerslag"]}%
    Tomorrow range: {data["d1tmin"]}:{data["d1tmax"]}
    Tommorow sun: {data["d1zon"]}%
    Tommorow rain: {data["d1neerslag"]}%
"""


def get_weather(location):
    url = create_api_request("Utrecht")
    response = requests.get(url)
    data = response.json()
    return construct_weather(data["liveweer"][0])
