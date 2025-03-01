document.addEventListener('DOMContentLoaded', () => {
  const container = document.getElementById('content');

  chrome.storage.local.get(['selectedImage', 'pixelCount'], (data) => {
    if (data.selectedImage) {
      container.innerHTML = `
        <div id="imageWrapper">
          <img src="${data.selectedImage}" alt="Selected Image" id="selectedImage">
        </div>
        <div id="pixelCountLabel">The number of pixels in this image is</div>
        <div id="pixelCount">${data.pixelCount.toLocaleString()} pixels</div>
      `;
    } else {
      container.innerHTML = `<p>No image selected.</p>`;
    }
  });
});
