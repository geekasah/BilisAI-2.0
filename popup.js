document.addEventListener('DOMContentLoaded', () => {
  // Get references to DOM elements
  const contentContainer = document.getElementById('content');
  const processingIndicator = document.getElementById('processingIndicator');
  const processingStatus = document.getElementById('processingStatus');
  const processingProgress = document.getElementById('processingProgress');
  const progressFill = document.querySelector('.progress-fill');
  const progressText = document.querySelector('.progress-text');
  const resultsSection = document.getElementById('resultsSection');
  const confidenceFill = document.querySelector('.confidence-fill');
  const confidenceValue = document.querySelector('.confidence-value');
  const errorMessage = document.getElementById('errorMessage');
  const errorText = document.querySelector('.error-text');

<<<<<<< Updated upstream
  // Load current state from storage
  updateFromStorage();
  
  // Listen for storage changes to update the UI
  chrome.storage.onChanged.addListener((changes) => {
    updateFromStorage();
  });
  
  // Function to update the UI based on storage data
  function updateFromStorage() {
    chrome.storage.local.get([
      'mediaType',
      'mediaUrl',
      'isProcessing',
      'aiScore',
      'framesProcessed',
      'totalFrames',
      'error'
    ], (data) => {
      // Reset UI elements
      contentContainer.innerHTML = '';
      processingIndicator.classList.add('hidden');
      processingProgress.classList.add('hidden');
      resultsSection.classList.add('hidden');
      errorMessage.classList.add('hidden');
      
      // Handle error state
      if (data.error) {
        showError(data.error);
        return;
      }
      
      // If no media is selected
      if (!data.mediaUrl) {
        contentContainer.innerHTML = `<p>No media selected.</p>
          <p>Right-click on an image or video and select "Check if this is AI-generated".</p>`;
        return;
      }
      
      // Show the media preview
      showMediaPreview(data.mediaType, data.mediaUrl);
      
      // Processing state
      if (data.isProcessing) {
        showProcessingState(data);
        return;
      }
      
      // Results state
      if (data.aiScore !== null && data.aiScore !== undefined) {
        showResults(data.aiScore);
      }
    });
  }
  
  // Function to show the media preview
  function showMediaPreview(mediaType, mediaUrl) {
    let previewHTML = '';
    
    if (mediaType === 'image') {
      previewHTML = `
        <div id="imageWrapper">
          <img src="${mediaUrl}" alt="Selected Image" id="selectedImage">
        </div>
      `;
    } else if (mediaType === 'video') {
      previewHTML = `
        <div id="videoWrapper">
          <video src="${mediaUrl}" controls muted id="selectedVideo"></video>
        </div>
      `;
    }
    
    contentContainer.innerHTML = previewHTML;
    
    // Size handling for video element
    if (mediaType === 'video') {
      const video = document.getElementById('selectedVideo');
      video.addEventListener('loadedmetadata', () => {
        // Keep video at a reasonable size
        if (video.videoWidth > 280) {
          video.style.width = '280px';
          video.style.height = 'auto';
        }
      });
    }
  }
  
  // Function to show the processing state
  function showProcessingState(data) {
    processingIndicator.classList.remove('hidden');
    
    if (data.mediaType === 'video' && data.totalFrames > 0) {
      // Show progress for video processing
      processingProgress.classList.remove('hidden');
      processingStatus.textContent = 'Analyzing video frames...';
      
      const progress = (data.framesProcessed / data.totalFrames) * 100;
      progressFill.style.width = `${progress}%`;
      progressText.textContent = `${Math.round(progress)}%`;
    } else {
      // Show indefinite spinner for image processing
      processingStatus.textContent = 'Analyzing...';
    }
  }
  
  // Function to show the results
  function showResults(aiScore) {
    resultsSection.classList.remove('hidden');
    
    // Update confidence meter
    const scoreValue = Math.round(aiScore);
    confidenceFill.style.width = `${scoreValue}%`;
    confidenceValue.textContent = `${scoreValue}%`;
    
    // Set color based on confidence level
    if (scoreValue < 30) {
      confidenceFill.style.backgroundColor = '#4CAF50'; // Green for low AI confidence
    } else if (scoreValue < 70) {
      confidenceFill.style.backgroundColor = '#FFC107'; // Yellow for medium AI confidence
    } else {
      confidenceFill.style.backgroundColor = '#F44336'; // Red for high AI confidence
    }
  }
  
  // Function to show an error message
  function showError(errorMessage) {
    errorMessage.classList.remove('hidden');
    errorText.textContent = `Error: ${errorMessage}`;
  }
=======
  // Show loading state immediately
  function showLoading(context) {
    container.innerHTML = `
      <div id="loadingWrapper">
        <div class="loading-spinner"></div>
        <p class="loading-text">Analyzing ${context}...</p>
        <p class="loading-subtext">This may take a few moments</p>
      </div>
    `;
  }

  // Check if we're in a loading state
  chrome.storage.local.get(['isAnalyzing', 'analysisContext'], (loadingData) => {
    if (loadingData.isAnalyzing) {
      showLoading(loadingData.analysisContext || 'content');
      
      // Set a timeout to handle potential stuck loading state
      setTimeout(() => {
        chrome.storage.local.get([
          'dataType', 
          'errorMessage',
          'errorContext'
        ], (data) => {
          // If no data after timeout, show error
          if (data.dataType === "error" || !data.dataType) {
            container.innerHTML = `
              <div id="errorWrapper">
                <h2>Analysis Timed Out</h2>
                <p>The analysis took longer than expected.</p>
                <p class="troubleshoot">
                  Possible reasons:
                  • Large file size
                  • Network slowness
                  • Server processing delay
                </p>
              </div>
            `;
          }
        });
      }, 60000); // 1 minute timeout
      return;
    }

    // Regular data processing
    chrome.storage.local.get([
      'dataType', 
      'selectedImage', 
      'selectedVideo',
      'prediction', 
      'confidence', 
      'selectedText', 
      'label', 
      'probability',
      'is_ai_generated',
      'sampled_frames',
      'errorMessage',
      'errorContext'
    ], (data) => {
      const type = data.dataType;

      // Error handling first
      if (type === "error") {
        container.innerHTML = `
          <div id="errorWrapper">
            <h2>Analysis Error</h2>
            <p><strong>Context:</strong> ${data.errorContext || 'Unknown'}</p>
            <p><strong>Error:</strong> ${data.errorMessage || 'No details available'}</p>
            <p class="troubleshoot">
              Possible reasons:
              • Server not running
              • Network connectivity issues
              • CORS restrictions
              • File type not supported
            </p>
          </div>
        `;
        return;
      }

      if (type === "image" && data.selectedImage) {
        container.innerHTML = `
          <div id="imageWrapper">
            <img src="${data.selectedImage}" alt="Selected Image" id="selectedImage">
          </div>
          <div id="result">
            <div id="predictionLabel">Prediction:</div>
            <div id="predictionValue">${data.prediction}</div>
            <div id="confidenceLabel">Confidence Level:</div>
            <div id="confidenceValue">${(data.confidence * 100).toFixed(2)}%</div>
            <div id="generatedLabel">AI Generated:</div>
            <div id="generatedValue">${data.is_ai_generated ? 'Yes' : 'No'}</div>
          </div>
        `;
      }

      else if (type === "video" && data.selectedVideo) {
        container.innerHTML = `
          <div id="videoWrapper">
            <video controls id="selectedVideo">
              <source src="${data.selectedVideo}" type="video/mp4">
              Your browser does not support the video tag.
            </video>
          </div>
          <div id="result">
            <div id="predictionLabel">Prediction:</div>
            <div id="predictionValue">${data.prediction}</div>
            <div id="confidenceLabel">Confidence Level:</div>
            <div id="confidenceValue">${(data.confidence * 100).toFixed(2)}%</div>
            <div id="generatedLabel">AI Generated:</div>
            <div id="generatedValue">${data.is_ai_generated ? 'Yes' : 'No'}</div>
            <div id="framesLabel">Sampled Frames:</div>
            <div id="framesValue">${data.sampled_frames}</div>
          </div>
        `;
      }

      else if (type === "text" && data.selectedText) {
        container.innerHTML = `
            <div id="textWrapper">
              <p id="selectedText">${data.selectedText}</p>
            </div>
            <div id="result">
              <div id="labelLabel">Prediction:</div>
              <div id="labelValue">${data.label}</div>
              <div id="confidenceLabel">Confidence Level:</div>
              <div id="confidenceValue">${(data.probability * 100).toFixed(2)}%</div>
            </div>
          `;
      } 
      else {
        container.innerHTML = `<p>No data selected.</p>`;
      }
    });
  });
>>>>>>> Stashed changes
});