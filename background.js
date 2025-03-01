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

    // Naive word count: split on whitespace. Trim first to avoid extra spaces.
    const wordCount = selectedText.trim().split(/\s+/).filter(Boolean).length;

    // Store the data in Chrome storage, marking it as 'text'
    chrome.storage.local.set({
      dataType: "text",
      selectedText: selectedText,
      wordCount: wordCount
    }, () => {
      console.log("Text + word count saved:", selectedText, wordCount);

      // Attempt to open the popup automatically
      chrome.action.openPopup().catch(err => {
        console.error("openPopup() failed:", err);
      });
    });
  }
});
