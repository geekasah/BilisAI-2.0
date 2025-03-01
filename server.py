from transformers import AutoImageProcessor, AutoTokenizer, AutoConfig, AutoModelForImageClassification, AutoModel
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from transformers import PreTrainedModel

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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)