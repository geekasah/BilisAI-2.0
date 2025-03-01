chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({
    id: "fetchImage",
    title: "Select this image for Popup",
    contexts: ["image"]
  });
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "fetchImage") {
    const imageUrl = info.srcUrl;

    fetch(imageUrl)
      .then(response => response.blob())
      .then(blob => createImageBitmap(blob))
      .then(bitmap => {
        const pixelCount = bitmap.width * bitmap.height;

        // Store the data in Chrome storage
        chrome.storage.local.set({
          selectedImage: imageUrl,
          pixelCount: pixelCount
        }, () => {
          console.log("Image and pixel count saved:", imageUrl, pixelCount);

          // Attempt to open the popup automatically
          chrome.action.openPopup().catch(err => {
            console.error("openPopup() failed:", err);
          });
        });
      })
      .catch(err => console.error("Error processing image:", err));
  }
});
