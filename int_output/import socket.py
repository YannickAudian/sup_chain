import socket
import requests

# Récupérer l'adresse IP locale
hostname = socket.gethostname()
local_ip = socket.gethostbyname(hostname)
print(f"Adresse IP locale : {local_ip}")

# Récupérer l'adresse IP publique
try:
    response = requests.get('https://api64.ipify.org?format=json')
    public_ip = response.json().get('ip')
    print(f"Adresse IP publique : {public_ip}")
except requests.RequestException as e:
    print(f"Erreur pour récupérer l'IP publique : {e}")
