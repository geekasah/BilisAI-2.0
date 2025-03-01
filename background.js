chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "fetchImage",
    title: "Select this image for Popup",
    contexts: ["image"]
  });

  chrome.contextMenus.create({
    id: "analyzeText",
    title: "Analyze selected text",
    contexts: ["selection"]
  });

});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  // IMAGE FLOW
  if (info.menuItemId === "fetchImage") {
    const imageUrl = info.srcUrl;

    // Fetch the image as a blob
    fetch(imageUrl)
      .then(response => response.blob())
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
      .then(response => response.json())
      .then(result => {
        // The API returns a JSON object with keys:
        // 'prediction', 'confidence', 'is_ai_generated', and 'status'
        
        if (result.prediction && result.prediction.toLowerCase() === "artificial") {
          result.prediction = "AI Generated";
        } else {
          result.prediction = "Not AI Generated";
        }

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
      .catch(error => console.error("Error sending request to API:", error));
  }

  // TEXT FLOW
  else if (info.menuItemId === "analyzeText") {
    const selectedText = info.selectionText || "";
    fetch('http://localhost:5000/detect-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: selectedText })
    })
      .then(response => {
        if (!response.ok)
          throw new Error("API request failed with status " + response.status);
        return response.json();
      })
      .then(result => {
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
      .catch(error => console.error("Error sending text to API:", error));
  }
});
