document.addEventListener('DOMContentLoaded', () => {
  const container = document.getElementById('content');

   chrome.storage.local.get(['dataType', 'selectedImage', 'selectedVideo', 'prediction', 'confidence', 'selectedText', 'label', 'probability', 'is_ai_generated', 'sampled_frames'], (data) => {
    const type = data.dataType;

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
        </div>
      `;
    }

    else if (type === "text" && data.selectedText) {
      // Render text-related UI
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
    else if (type === "video" && data.selectedVideo) {
      container.innerHTML = `
        <div id="videoWrapper">
          <video controls id="selectedVideo">
            <source src="${data.selectedVideo}" type="video/mp4">
            Your browser does not support the video tag.
          </video>
        </div>
        <div id="result">
          <div id="labelLabel">Prediction:</div>
          <div id="labelValue">${data.prediction}</div>
          <div id="confidenceLabel">Confidence Level:</div>
          <div id="confidenceValue">${(data.confidence * 100).toFixed(2)}%</div>
        </div>
      `;
    }

    else {
      // Default message if nothing was stored
      container.innerHTML = `<p>No data selected.</p>`;
    }
  });
});
