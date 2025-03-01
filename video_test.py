import requests
import os

# URL of the Flask API endpoint for video detection
url = 'http://localhost:5001/detect-video'  # Changed to port 5001

# Path to the video you want to test
video_path = 'test_video.mp4'  # Use the test video we'll create

# Check if file exists
if not os.path.exists(video_path):
    print(f"Error: File not found at {video_path}")
    exit(1)

# Check file size
file_size = os.path.getsize(video_path)
print(f"File size: {file_size / (1024 * 1024):.2f} MB")

# Open the video file in binary mode and send it as part of the POST request
try:
    with open(video_path, 'rb') as video_file:
        files = {'video': video_file}
        response = requests.post(url, files=files, timeout=120)  # Added timeout of 2 minutes

    # Check if the request was successful and then print the returned JSON
    if response.status_code == 200:
        result = response.json()
        print("Prediction:", result.get('prediction', 'N/A'))
        print("Confidence:", result.get('confidence', 'N/A'))
        print("Is AI Generated:", result.get('is_ai_generated', 'N/A'))
        print("Sampled Frames:", result.get('sampled_frames', 'N/A'))
        
        # Additional error checking
        if 'error' in result:
            print("API Error:", result['error'])
    else:
        print("HTTP Error:", response.status_code)
        print("Response Text:", response.text)

except requests.exceptions.RequestException as e:
    print(f"Request Error: {e}")
except Exception as e:
    print(f"Unexpected Error: {e}")
