import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'R&D Spend':153441.51, 'Administration':136897.80 , 'Marketing Spend':471784.10,'State':1})

print(r.json())