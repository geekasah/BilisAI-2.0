from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# Create Flask app instance
app = Flask(__name__, static_folder="static")
CORS(app)  # Enable CORS for all routes

# Load model and processor (load once when the API starts)
processor = AutoImageProcessor.from_pretrained("Organika/sdxl-detector")
model = AutoModelForImageClassification.from_pretrained("Organika/sdxl-detector")

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
        
        # Process the image
        inputs = processor(images=image, return_tensors="pt")
        
        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
        
        # Get predicted class
        predicted_class_idx = logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_class_idx]
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

if __name__ == '__main__':
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)