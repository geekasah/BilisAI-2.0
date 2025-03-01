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
<<<<<<< Updated upstream
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
=======

  // New context menu item for videos
  chrome.contextMenus.create({
    id: "fetchVideo",
    title: "Analyze this video",
    contexts: ["video"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  // Set loading state before analysis
  const setLoadingState = (context) => {
    chrome.storage.local.set({
      isAnalyzing: true,
      analysisContext: context
    });
  };

  // Clear loading state after analysis
  const clearLoadingState = () => {
    chrome.storage.local.remove(['isAnalyzing', 'analysisContext']);
  };

  // IMAGE FLOW
  if (info.menuItemId === "fetchImage") {
    const imageUrl = info.srcUrl;
    
    // Set loading state
    setLoadingState('image');

    // Fetch the image as a blob
    fetch(imageUrl)
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.blob();
      })
      .then(blob => {
        // Create a FormData instance and append the blob as a file
        const formData = new FormData();
        formData.append('image', blob, 'image.jpg'); // filename can be adjusted

        // Send the POST request to the Python API endpoint
        return fetch('http://localhost:5000/detect', {
          method: 'POST',
          body: formData
        });
      })
      .then(response => {
        if (!response.ok) {
          throw new Error(`API error! status: ${response.status}`);
        }
        return response.json();
      })
      .then(result => {
        // Clear loading state
        clearLoadingState();

        // The API returns a JSON object with keys:
        // 'prediction', 'confidence', 'is_ai_generated', and 'status'
        chrome.storage.local.set({
          dataType: "image",
          selectedImage: imageUrl,
          prediction: result.prediction,
          confidence: result.confidence,
          is_ai_generated: result.is_ai_generated
        }, () => {
          console.log("Image analysis saved:", result);
          // Optionally, open the popup automatically if allowed by user gesture:
          chrome.action.openPopup().catch(err => console.error("openPopup() failed:", err));
        });
      })
      .catch(error => {
        // Clear loading state
        clearLoadingState();

        console.error("Error in image analysis:", error);
        
        // Store error in local storage for user feedback
        chrome.storage.local.set({
          dataType: "error",
          errorMessage: error.toString(),
          errorContext: "Image Analysis"
        }, () => {
          chrome.action.openPopup().catch(err => console.error("openPopup() failed:", err));
        });
      });
  }

  // VIDEO FLOW
  else if (info.menuItemId === "fetchVideo") {
    const videoUrl = info.srcUrl;
    
    // Set loading state
    setLoadingState('video');
    
    console.log("Starting video analysis for:", videoUrl);

    // Fetch the video as a blob
    fetch(videoUrl)
      .then(response => {
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        console.log("Video fetched successfully, converting to blob");
        return response.blob();
      })
      .then(blob => {
        // Log blob info for debugging
        console.log(`Blob received: ${blob.size} bytes, type: ${blob.type}`);
        
        // Create a FormData instance and append the blob as a file
        const formData = new FormData();
        formData.append('video', blob, 'video.mp4');

        // Send the POST request to the Python API endpoint
        console.log("Sending video to API for analysis");
        return fetch('http://localhost:5000/detect-video', {
          method: 'POST',
          body: formData,
          // Add these headers to help with potential CORS issues
          headers: {
            'Access-Control-Allow-Origin': '*'
          }
        });
      })
      .then(response => {
        if (!response.ok) {
          console.error(`API returned error status: ${response.status}`);
          throw new Error(`API error! status: ${response.status}`);
        }
        console.log("API response received, parsing JSON");
        return response.json();
      })
      .then(result => {
        // Clear loading state
        clearLoadingState();
        
        console.log("Video analysis complete:", result);

        // Store video analysis results
        chrome.storage.local.set({
          dataType: "video",
          selectedVideo: videoUrl,
          prediction: result.prediction,
          confidence: result.confidence,
          is_ai_generated: result.is_ai_generated,
          sampled_frames: result.sampled_frames
        }, () => {
          console.log("Video analysis saved to storage");
          // Open the popup automatically
          chrome.action.openPopup().catch(err => console.error("openPopup() failed:", err));
        });
      })
      .catch(error => {
        // Clear loading state
        clearLoadingState();

        console.error("Error in video analysis:", error);
        
        // Store error in local storage for user feedback
        chrome.storage.local.set({
          dataType: "error",
          errorMessage: error.toString(),
          errorContext: "Video Analysis"
        }, () => {
          chrome.action.openPopup().catch(err => console.error("openPopup() failed:", err));
        });
      });
  }

  // TEXT FLOW
  else if (info.menuItemId === "analyzeText") {
    const selectedText = info.selectionText || "";
    
    // Set loading state
    setLoadingState('text');

    fetch('http://localhost:5000/detect-text', {
      method: 'POST',
      headers: { 
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      },
      body: JSON.stringify({ text: selectedText })
    })
      .then(response => {
        if (!response.ok)
          throw new Error("API request failed with status " + response.status);
        return response.json();
      })
      .then(result => {
        // Clear loading state
        clearLoadingState();

        chrome.storage.local.set({
          dataType: "text",
          selectedText: selectedText,
          label: result.label,
          probability: result.probability
        }, () => {
          console.log("Text analysis saved:", result);
          chrome.action.openPopup().catch(err => console.error("openPopup() failed:", err));
        });
      })
      .catch(error => {
        // Clear loading state
        clearLoadingState();

        console.error("Error sending text to API:", error);
        
        // Store error in local storage for user feedback
        chrome.storage.local.set({
          dataType: "error",
          errorMessage: error.toString(),
          errorContext: "Text Analysis"
        }, () => {
          chrome.action.openPopup().catch(err => console.error("openPopup() failed:", err));
        });
      });
  }
});
>>>>>>> Stashed changes
