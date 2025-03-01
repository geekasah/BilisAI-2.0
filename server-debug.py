from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import time

# Create a simplified test server
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def index():
    return jsonify({'status': 'Server is running'})

@app.route('/detect-video', methods=['POST'])
def detect_video_test():
    try:
        # Debug information
        print("\n--- New Video Upload Request ---")
        print(f"Content-Type: {request.content_type}")
        print(f"Request Files: {list(request.files.keys())}")
        
        # Check if the request has the file part
        if 'video' not in request.files:
            print("Error: No video file in request")
            return jsonify({'error': 'No video file provided'}), 400
        
        # Get video file from request
        video_file = request.files['video']
        
        # Print file info
        print(f"Filename: {video_file.filename}")
        print(f"Content Type: {video_file.content_type}")
        
        # Validate file
        if video_file.filename == '':
            print("Error: Empty filename")
            return jsonify({'error': 'Empty file name'}), 400
        
        # Save file to check content
        temp_path = os.path.join(os.getcwd(), 'debug_video.mp4')
        video_file.save(temp_path)
        file_size = os.path.getsize(temp_path)
        print(f"File saved to {temp_path}, size: {file_size / (1024 * 1024):.2f} MB")
        
        # Simulate processing time
        print("Processing video...")
        time.sleep(2)  # Simulate short processing time
        
        # Return mock result
        result = {
            'prediction': 'Test Result',
            'confidence': 0.95,
            'is_ai_generated': True,
            'sampled_frames': 10,
            'status': 'success',
            'debug_info': {
                'file_size': file_size,
                'file_saved': True
            }
        }
        
        print("Returning success response")
        return jsonify(result)
        
    except Exception as e:
        print(f"Server error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = 5001  # Use a different port to avoid conflict with your main server
    print(f"Starting debug server on port {port}...")
    print(f"To test: python video_test.py with URL http://localhost:{port}/detect-video")
    app.run(host='0.0.0.0', port=port, debug=True)
