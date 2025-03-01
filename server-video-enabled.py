from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np

# Create Flask app instance
app = Flask(__name__, static_folder="static")
CORS(app)  # Enable CORS for all routes

# --- Load image detection model and processor ---
print("Loading image detection model...")
image_processor = AutoImageProcessor.from_pretrained("Organika/sdxl-detector")
image_model = AutoModelForImageClassification.from_pretrained("Organika/sdxl-detector")

# Set up device for models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
image_model.to(device)

# Optimize models for inference if on GPU
if device.type == 'cuda':
    # Use mixed precision for faster computation on GPU
    image_model = image_model.half()  # Convert to FP16

# Route to serve the HTML file
@app.route('/')
def index():
    return jsonify({'status': 'Server is running'})

@app.route('/detect-video', methods=['POST'])
def detect_video_route():
    try:
        print("\n--- New Video Upload Request ---")
        # Check if the request has the file part
        if 'video' not in request.files:
            print("Error: No video file in request")
            return jsonify({'error': 'No video file provided'}), 400
        
        # Get video file from request
        video_file = request.files['video']
        print(f"Filename: {video_file.filename}")
        
        # Validate file
        if video_file.filename == '':
            print("Error: Empty filename")
            return jsonify({'error': 'Empty file name'}), 400
        
        # Temporarily save uploaded file
        temp_video_path = 'temp_video.mp4'
        video_file.save(temp_video_path)
        file_size = os.path.getsize(temp_video_path) / (1024 * 1024)
        print(f"File saved to {temp_video_path}, size: {file_size:.2f} MB")
        
        # Open video capture
        video = cv2.VideoCapture(temp_video_path)
        
        # Get video metadata
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        video_duration = total_frames / fps
        
        print(f"Video info: {total_frames} frames, {fps} FPS, {video_duration:.2f} seconds")
        
        # Determine number of frames to sample based on video duration
        if video_duration < 10:
            num_frames = 3
        elif video_duration < 30:
            num_frames = 10
        else:
            num_frames = 15
        
        # Sampling strategy: evenly distribute frames across video
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        print(f"Sampling {num_frames} frames at indices: {frame_indices}")
        
        # Store confidence scores
        confidence_scores = []
        predictions = []
        
        # Iterate through selected frames
        for frame_idx in frame_indices:
            # Set video capture to specific frame
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = video.read()
            
            if not ret:
                print(f"Warning: Could not read frame {frame_idx}")
                continue
            
            # Convert OpenCV image (BGR) to PIL Image (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Process the image
            inputs = image_processor(images=pil_image, return_tensors="pt")
            
            # Move inputs to the same device as the model
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = image_model(**inputs)
                logits = outputs.logits
            
            # Get predicted class and confidence
            predicted_class_idx = logits.argmax(-1).item()
            predicted_label = image_model.config.id2label[predicted_class_idx]
            confidence = torch.softmax(logits, dim=1)[0][predicted_class_idx].item()
            
            # Store confidence score and prediction
            confidence_scores.append(float(confidence))
            predictions.append(predicted_label)
            
            print(f"Frame {frame_idx}: {predicted_label} with {confidence:.4f} confidence")
        
        # Close video capture
        video.release()
        
        # Optional: Remove temporary video file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)
        
        # Calculate results
        if not confidence_scores:
            return jsonify({'error': 'Could not analyze any frames in the video'}), 500
            
        # Count AI vs Not AI predictions
        ai_count = predictions.count('AI')
        not_ai_count = predictions.count('Not AI')
        
        # Determine final prediction
        final_prediction = 'AI' if ai_count > not_ai_count else 'Not AI'
        
        # Calculate average confidence
        avg_confidence = np.mean(confidence_scores)
        
        # Consider it AI-generated if majority of frames are classified as AI
        is_ai_generated = (ai_count > not_ai_count)
        
        result = {
            'prediction': final_prediction,
            'confidence': float(avg_confidence),
            'is_ai_generated': is_ai_generated,
            'sampled_frames': num_frames,
            'ai_frames': ai_count,
            'not_ai_frames': not_ai_count,
            'status': 'success'
        }
        
        print(f"Video analysis result: {result}")
        return jsonify(result)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in video analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/detect', methods=['POST'])
def detect_image():
    try:
        # Check if the request has the file part
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        # Get image file from request
        image_file = request.files['image']
        
        # Validate file
        if image_file.filename == '':
            return jsonify({'error': 'Empty file name'}), 400
            
        # Open and process the image
        image = Image.open(image_file)
        
        # Convert to RGB if the image is in RGBA mode (for PNG transparency)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        # Process the image
        inputs = image_processor(images=image, return_tensors="pt")
        
        # Move inputs to the same device as the model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Make prediction
        with torch.no_grad():
            outputs = image_model(**inputs)
            logits = outputs.logits
        
        # Get predicted class
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = image_model.config.id2label[predicted_class_idx]
        confidence = torch.softmax(logits, dim=1)[0][predicted_class_idx].item()
        
        # Return results as JSON
        return jsonify({
            'prediction': predicted_label,
            'confidence': float(confidence),
            'is_ai_generated': predicted_label.lower() == 'ai',  # Assuming 'ai' is the label for AI images
            'status': 'success'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect-text', methods=['POST'])
def detect_text():
    """
    Temporary replacement for text detection while the model is unavailable.
    Always returns the same placeholder response.
    """
    return jsonify({
        'label': 'Text detection temporarily disabled',
        'probability': 0.5,
        'status': 'success'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    print(f"Starting server on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=debug_mode)