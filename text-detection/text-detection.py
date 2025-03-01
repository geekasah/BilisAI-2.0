import torch
import torch.nn as nn
import json
import time
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel

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
        # Pass only what's needed to reduce memory usage
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=False,  # Don't need all hidden states
            return_dict=False  # Simple tuple return is faster
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

def predict_single_text(text, model, tokenizer, device, max_len=512, threshold=0.5):
    # Reduce max_len from 768 to 512 for faster processing
    # Most classification tasks work well with 512 tokens
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

def main():
    # --- Model and Tokenizer Directory ---
    model_directory = "desklib/ai-text-detector-v1.01"
    
    # Load tokenizer with caching enabled
    tokenizer = AutoTokenizer.from_pretrained(model_directory, use_fast=True)
    
    # Load model
    model = DesklibAIDetectionModel.from_pretrained(model_directory)
    
    # --- Set up device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Optimize model for inference if on GPU
    if device.type == 'cuda':
        # Use mixed precision for faster computation on GPU
        model = model.half()  # Convert to FP16

    # --- Example Input text ---
    text_ai = input("Test AI input: ")

    # --- Run prediction and output as JSON ---
    result = predict_single_text(text_ai, model, tokenizer, device)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()