from transformers import AutoImageProcessor, AutoTokenizer, AutoConfig, AutoModelForImageClassification, AutoModel, PreTrainedModel
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np

# Custom AI text detection model class
class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        # Initialize the base transformer model.
        self.model = AutoModel.from_config(config)
        # Define a classifier head.
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Initialize weights (handled by PreTrainedModel)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward pass through the transformer
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=False,
            return_dict=False
        )
        
        # Get the last hidden state more efficiently
        last_hidden_state = outputs[0]
        
        # Optimized mean pooling
        # Create mask once and reuse
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
        if input_mask_expanded.dtype != last_hidden_state.dtype:
            input_mask_expanded = input_mask_expanded.to(last_hidden_state.dtype)
            
        # Compute mean pooling in a single step
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # Classifier
        logits = self.classifier(pooled_output)
        
        # Only compute loss if needed
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())

        # Simplified output
        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output

# Text prediction function
def predict_single_text(text, model, tokenizer, device, max_len=512, threshold=0.5):
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    
    # Move to device in one operation to reduce transfers
    encoded = {k: v.to(device) for k, v in encoded.items()}
    input_ids = encoded['input_ids']
    attention_mask = encoded['attention_mask']

    # Ensure model is in evaluation mode
    model.eval()
    
    # Use torch.inference_mode which is faster than no_grad
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        probability = torch.sigmoid(logits).item()

    label = "AI Generated" if probability >= threshold else "Not AI Generated"
    
    # Return results as a dictionary for JSON serialization
    return {
        "probability": round(probability, 4),
        "label": label
    }

# Create Flask app instance
app = Flask(__name__, static_folder="static")
CORS(app)  # Enable CORS for all routes

# --- Load image detection model and processor ---
image_processor = AutoImageProcessor.from_pretrained("Organika/sdxl-detector")
image_model = AutoModelForImageClassification.from_pretrained("Organika/sdxl-detector")

# --- Load text detection model and tokenizer ---
text_model_directory = "desklib/ai-text-detector-v1.01"
text_tokenizer = AutoTokenizer.from_pretrained(text_model_directory, use_fast=True)
text_model = DesklibAIDetectionModel.from_pretrained(text_model_directory)

# Set up device for models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_model.to(device)
text_model.to(device)

# Optimize models for inference if on GPU
if device.type == 'cuda':
    # Use mixed precision for faster computation on GPU
    text_model = text_model.half()  # Convert to FP16

# Route to serve the HTML file
@app.route('/')
def index():
    return app.send_static_file('index.html')

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
        
        # Convert your inputs to half precision
        inputs = image_processor(images=image, return_tensors="pt")
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
        print(e)
        return jsonify({'error': str(e)}), 500

@app.route('/detect-text', methods=['POST'])
def detect_text():
    try:
        # Check if the request has the text data
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        
        # Check if text is provided
        if 'text' not in data or not data['text']:
            return jsonify({'error': 'No text provided'}), 400
            
        # Get text from request
        text = data['text']
        
        # Process the text and get prediction
        result = predict_single_text(text, text_model, text_tokenizer, device)
        
        # Add status to the response
        result['status'] = 'success'
        
        # Return results as JSON
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)