import urllib.request
import json
import os
import ssl

def allowSelfSignedHttps(allowed):
    # bypass the server certificate verification on client side
    if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
        ssl._create_default_https_context = ssl._create_unverified_context

allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

data = {
    "data":
    [
        {
            'Identification no': "0",
            'Phenotype1': "0",
            'Phenotype2': "0",
            'Phenotype3': "0",
            'Phenotype4': "0",
            'Phenotype5': "0",
            'Phenotype6': "0",
            'Phenotype7': "0",
            'Phenotype8': "0",
            'Phenotype9': "0",
            'Phenotype10': "0",
            'Phenotype11': "0",
            'Phenotype12': "0",
            'Phenotype13': "0",
            'Phenotype14': "0",
            'Phenotype15': "0",
            'Phenotype16': "0",
            'Phenotype17': "0",
            'Phenotype18': "0",
            'Phenotype19': "0",
            'Phenotype20': "0",
            'Phenotype21': "0",
            'Phenotype22': "0",
            'Phenotype23': "0",
            'Phenotype24': "0",
            'Phenotype25': "0",
            'Phenotype26': "0",
            'Phenotype27': "0",
        },
    ],
}

body = str.encode(json.dumps(data))

url = 'http://5af42fb2-502b-4281-8a72-64f5722f362f.southcentralus.azurecontainer.io/score'
api_key = '' # Replace this with the API key for the web service
headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)

    result = response.read()
    print(result)
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers - they include the requert ID and the timestamp, which are useful for debugging the failure
    print(error.info())
    print(json.loads(error.read().decode("utf8", 'ignore')))