# Riyal or Fakeh

## Project Overview
Riyal or Fakeh is a project that involves a Chrome extension and a server to process data. Follow the instructions below to set up the project and get it running.

## Instructions

### 1. Clone the Repository
Start by cloning the repository to your local machine using the following command:
```bash
git clone https://github.com/geekasah/BilisAI-2.0/tree/main
```

### 2. Create a Virtual Environment
After cloning the repository, create a virtual environment to isolate the project dependencies:

```bash
python -m venv venv
```

### 3. Install the Required Dependencies
Install the project dependencies using pip. It’s recommended to install each dependency one by one:

```bash
pip install -r requirements.txt
```

### 4. Enable Developer Mode in Chrome
To load the Chrome extension, follow these steps:

Open Chrome and go to `chrome://extensions`.
Enable `Developer Mode` at the top-right of the page.
### 5. Load the Unpacked Extension
Click the `Load Unpacked` button and select the directory where you cloned the GitHub repository. This will load the extension in your Chrome browser.

### 6. Start the Server
To process the data, you need to run the server. Open your terminal in the project directory and execute the following command:

```bash
python server.py
```
Now, the extension should be fully functional and ready to use!
