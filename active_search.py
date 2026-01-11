import requests

query = "Silk Road trade cultural exchange"
response = requests.get(f"http://127.0.0.1:8080/search?query={query}")
print(response.json())
