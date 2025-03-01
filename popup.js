document.addEventListener('DOMContentLoaded', () => {
  const container = document.getElementById('content');

   chrome.storage.local.get(['dataType', 'selectedImage', 'prediction', 'confidence', 'selectedText', 'wordCount'], (data) => {
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
        <div id="wordCountLabel">The number of words in this sentence is</div>
        <div id="wordCount">${data.wordCount} words</div>
      `;
    } 
    else {
      // Default message if nothing was stored
      container.innerHTML = `<p>No data selected.</p>`;
    }
  });
});
