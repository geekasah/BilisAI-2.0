// videoExtractor.js - Content script for extracting frames from videos

// Listen for messages from the background script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.action === "extractFrames") {
      console.log("Starting video frame extraction for:", message.videoUrl);
      extractVideoFrames(message.videoUrl)
        .then(() => {
          console.log("Video frame extraction complete");
          sendResponse({ success: true });
        })
        .catch(err => {
          console.error("Error extracting video frames:", err);
          sendResponse({ success: false, error: err.message });
        });
      return true; // Indicates we'll send a response asynchronously
    }
  });
  
  // Main function to extract frames from a video
  async function extractVideoFrames(videoUrl) {
    // Create video element
    const video = document.createElement('video');
    video.crossOrigin = "anonymous"; // Try to handle CORS
    video.style.display = "none";
    
    // Add video to page temporarily
    document.body.appendChild(video);
    
    try {
      // Set up video
      video.src = videoUrl;
      await new Promise((resolve, reject) => {
        video.onloadedmetadata = resolve;
        video.onerror = reject;
      });
      
      // Get video duration and calculate number of frames to sample
      const duration = video.duration;
      const frameCount = calculateFrameCount(duration);
      
      // Inform background script about total frame count
      chrome.runtime.sendMessage({
        action: "setTotalFrames",
        totalFrames: frameCount
      });
      
      // Load the video
      video.load();
      await new Promise((resolve) => {
        video.oncanplay = resolve;
      });
      
      // Extract frames at regular intervals
      const interval = duration / frameCount;
      
      // Create a canvas for frame extraction
      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext('2d');
      
      // Extract each frame
      for (let i = 0; i < frameCount; i++) {
        const time = i * interval;
        await extractFrame(video, time, ctx, canvas);
      }
    } finally {
      // Clean up
      if (video && video.parentNode) {
        video.parentNode.removeChild(video);
      }
    }
  }
  
  // Helper function to determine how many frames to sample based on video duration
  function calculateFrameCount(duration) {
    // For short videos (< 10s), sample more frames per second
    if (duration < 10) {
      return Math.min(Math.ceil(duration * 5), 50);
    }
    // For medium videos (10-60s), use moderate sampling
    else if (duration < 60) {
      return Math.min(Math.ceil(duration * 2), 100);
    }
    // For longer videos, use sparser sampling with a maximum
    else {
      return Math.min(Math.ceil(duration), 100);
    }
  }
  
  // Extract a single frame at the specified time
  async function extractFrame(video, time, ctx, canvas) {
    return new Promise((resolve, reject) => {
      // Seek to the specified time
      video.currentTime = time;
      
      // Wait for the seek to complete
      video.onseeked = () => {
        try {
          // Draw the current frame to the canvas
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          
          // Convert canvas to data URL
          const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
          
          // Send the frame to the background script
          chrome.runtime.sendMessage({
            action: "frameExtracted",
            frameDataUrl: dataUrl,
            timestamp: time
          }, (response) => {
            if (response && response.success) {
              resolve();
            } else {
              reject(new Error(response?.error || "Failed to process frame"));
            }
          });
        } catch (err) {
          reject(err);
        }
      };
      
      // Handle errors
      video.onerror = reject;
    });
  }