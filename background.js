// Updated background.js with video and AI detection support
chrome.runtime.onInstalled.addListener(() => {
  // Create context menu items for both images and videos
  chrome.contextMenus.create({
    id: "checkImage",
    title: "Check if this image is AI-generated",
    contexts: ["image"]
  });
  
  chrome.contextMenus.create({
    id: "checkVideo",
    title: "Check if this video is AI-generated",
    contexts: ["video"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "checkImage") {
    // Process a single image
    processImage(info.srcUrl, tab.id);
  } else if (info.menuItemId === "checkVideo") {
    // Process a video
    processVideo(info.srcUrl, tab.id);
  }
});

// Function to process a single image
function processImage(imageUrl, tabId) {
  // First, fetch the image
  fetch(imageUrl)
    .then(response => response.blob())
    .then(blob => {
      // Store the image URL for the popup
      chrome.storage.local.set({
        mediaType: 'image',
        mediaUrl: imageUrl,
        isProcessing: true,
        aiScore: null
      }, () => {
        // Open the popup to show "processing" status
        chrome.action.openPopup().catch(err => {
          console.error("Failed to open popup:", err);
        });
        
        // Create an offscreen document to process the image if needed
        return createImageBitmap(blob).then(bitmap => {
          // Send the image to our AI detection model
          return detectAIGenerated(bitmap);
        });
      });
    })
    .then(aiScore => {
      // Store the AI detection results
      chrome.storage.local.set({
        isProcessing: false,
        aiScore: aiScore
      }, () => {
        console.log("AI detection complete:", aiScore);
        // If popup is already open, it will update automatically via storage listener
      });
    })
    .catch(err => {
      console.error("Error processing image:", err);
      chrome.storage.local.set({
        isProcessing: false,
        error: err.message
      });
    });
}

// Function to process a video
function processVideo(videoUrl, tabId) {
  // Store initial data for the popup
  chrome.storage.local.set({
    mediaType: 'video',
    mediaUrl: videoUrl,
    isProcessing: true,
    framesProcessed: 0,
    totalFrames: 0,
    aiFrames: 0,
    aiScore: null
  }, () => {
    // Open the popup to show "processing" status
    chrome.action.openPopup().catch(err => {
      console.error("Failed to open popup:", err);
    });
  });

  // Inject a content script to extract video frames
  chrome.scripting.executeScript({
    target: { tabId: tabId },
    files: ['videoExtractor.js']
  }).then(() => {
    // Send a message to the content script to start frame extraction
    chrome.tabs.sendMessage(tabId, {
      action: "extractFrames",
      videoUrl: videoUrl
    });
  }).catch(err => {
    console.error("Error injecting video extractor script:", err);
    chrome.storage.local.set({
      isProcessing: false,
      error: err.message
    });
  });
}

// Listen for messages from content scripts
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === "frameExtracted") {
    // Process an individual video frame
    const frameDataUrl = message.frameDataUrl;
    
    // Convert data URL to bitmap
    fetch(frameDataUrl)
      .then(response => response.blob())
      .then(blob => createImageBitmap(blob))
      .then(bitmap => detectAIGenerated(bitmap))
      .then(frameAiScore => {
        // Update the running statistics
        chrome.storage.local.get([
          'framesProcessed', 
          'totalFrames', 
          'aiFrames'
        ], (data) => {
          const framesProcessed = data.framesProcessed + 1;
          const aiFrames = data.aiFrames + (frameAiScore > 0.5 ? 1 : 0);
          
          // Calculate overall confidence
          const overallConfidence = (aiFrames / framesProcessed) * 100;
          
          chrome.storage.local.set({
            framesProcessed: framesProcessed,
            aiFrames: aiFrames,
            aiScore: overallConfidence
          }, () => {
            // If this is the last frame, mark processing as complete
            if (framesProcessed >= data.totalFrames) {
              chrome.storage.local.set({ isProcessing: false });
            }
            sendResponse({ success: true });
          });
        });
      })
      .catch(err => {
        console.error("Error processing video frame:", err);
        sendResponse({ success: false, error: err.message });
      });
      
    // Return true to indicate we'll send a response asynchronously
    return true;
  }
  
  else if (message.action === "setTotalFrames") {
    chrome.storage.local.set({
      totalFrames: message.totalFrames
    });
    sendResponse({ success: true });
    return true;
  }
});

// AI Detection Model Function
// This is a placeholder function - implement your actual AI detection here
function detectAIGenerated(bitmap) {
  return new Promise((resolve) => {
    // Create a canvas to process the image
    const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(bitmap, 0, 0);
    
    // Get image data for processing
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    // Simple placeholder model (not a real AI detector)
    // In a real implementation, you would:
    // 1. Load a pre-trained model (e.g., using TensorFlow.js)
    // 2. Preprocess the image data
    // 3. Run inference to get a confidence score
    
    // For demonstration, this simulates a basic analysis using image statistics
    let score = analyzeImageStatistics(imageData);
    
    // Return confidence score (0-100)
    resolve(score);
  });
}

// Simple placeholder analysis - replace with actual AI detection model
function analyzeImageStatistics(imageData) {
  const data = imageData.data;
  let smoothness = 0;
  let artifactScore = 0;
  
  // Extremely simplified analysis of pixel patterns
  // (This is NOT a real AI detection algorithm)
  
  // Check local pixel variance as one signal
  for (let y = 1; y < imageData.height - 1; y++) {
    for (let x = 1; x < imageData.width - 1; x++) {
      const idx = (y * imageData.width + x) * 4;
      
      // Check neighbor pixels for smoothness
      const neighbors = [
        (y * imageData.width + (x-1)) * 4,      // left
        (y * imageData.width + (x+1)) * 4,      // right
        ((y-1) * imageData.width + x) * 4,      // top
        ((y+1) * imageData.width + x) * 4       // bottom
      ];
      
      // Calculate local variance (simplified)
      let localVariance = 0;
      for (const nIdx of neighbors) {
        localVariance += Math.abs(data[idx] - data[nIdx]) +
                        Math.abs(data[idx+1] - data[nIdx+1]) +
                        Math.abs(data[idx+2] - data[nIdx+2]);
      }
      
      // Accumulate smoothness score
      if (localVariance < 30) {  // Low variance threshold
        smoothness++;
      }
      
      // Check for repeating patterns (oversimplified)
      if (x > 2 && y > 2) {
        const pattern1 = data[idx] + data[idx+1] + data[idx+2];
        const pattern2 = data[idx - 8] + data[idx - 7] + data[idx - 6];
        
        if (Math.abs(pattern1 - pattern2) < 5) {
          artifactScore++;
        }
      }
    }
  }
  
  // Calculate scores relative to image size
  const totalPixels = imageData.width * imageData.height;
  const smoothnessRatio = smoothness / totalPixels;
  const artifactRatio = artifactScore / totalPixels;
  
  // Combine factors (simplified model)
  const score = (smoothnessRatio * 50) + (artifactRatio * 70);
  
  // Return a score between 0-100
  return Math.min(Math.max(score * 100, 0), 100);
}