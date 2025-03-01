import requests

# URL of the Flask API endpoint
url = 'http://localhost:5000/detect'

# Path to the image you want to test
image_path = 'image_1.jpg'

# Open the image file in binary mode and send it as part of the POST request
with open(image_path, 'rb') as img_file:
    files = {'image': img_file}
    response = requests.post(url, files=files)

# Check if the request was successful and then print the returned JSON
if response.status_code == 200:
    result = response.json()
    print("Prediction:", result['prediction'])
    print("Confidence:", result['confidence'])
    print("Is AI Generated:", result['is_ai_generated'])
else:
    print("Error:", response.status_code, response.text)