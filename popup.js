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
});