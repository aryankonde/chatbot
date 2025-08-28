# Smart Support Chatbot with RAG, Gemini, and Google Sheets Integration

A sophisticated, document-aware chatbot that provides accurate answers using a Retrieval-Augmented Generation (RAG) pipeline. This project features a dynamic web interface and a seamless escalation path for users to contact human support, with all requests automatically logged into a Google Sheet.

\<br\>

*(Recommendation: Record a short GIF of your application and replace the link above to showcase its functionality.)*

## Features

  * **Accurate, Context-Aware AI:** Utilizes a **RAG pipeline** to ground responses in your own documents, preventing the AI from hallucinating or providing irrelevant information.
  * **High-Speed Semantic Search:** Employs **FAISS** for lightning-fast retrieval of the most relevant document chunks based on the user's query.
  * **Powered by Google Gemini:** Leverages the state-of-the-art **Gemini 1.5 Flash** model for intelligent and coherent answer generation.
  * **Dynamic Chat Interface:** A fluid, single-page web application built with **Flask** and vanilla **JavaScript**, providing a temporary, scrollable chat history that resets on refresh.
  * **Human Support Escalation:** A user-friendly "Contact Us" feature allows users to submit their name, email, and question if they are not satisfied with the AI's response.
  * **Automated Ticket Logging:** Seamlessly integrates with **Google Sheets**, automatically logging all human support requests in real-time, creating an instant ticketing system.

## System Architecture

The application follows a modern web architecture where the frontend and backend are decoupled. The RAG pipeline is the core of the backend logic.

1.  **User Interaction:** The user types a message in the JavaScript-powered frontend.
2.  **API Call:** The frontend sends the query to a `/api/ask` endpoint on the Flask server.
3.  **RAG Pipeline Execution:**
      * The user's query is converted into a numerical vector.
      * FAISS performs a similarity search on the pre-indexed document vectors.
      * The most relevant document chunks are retrieved.
      * The chunks (context) and the original query are sent to the Gemini API.
4.  **Response Generation:** Gemini generates an answer based on the provided context.
5.  **Display:** The answer is sent back to the frontend and displayed in the chat window.
6.  **Escalation Flow:** If the user submits the contact form, the data is sent to a `/contact-support` endpoint, which authenticates with Google and appends a new row to the specified Google Sheet.

## Technology Stack

  * **Backend:** Python, Flask, Gspread, oauth2client
  * **AI / ML:** Google Gemini API, Sentence Transformers (`all-MiniLM-L6-v2`), Facebook AI Similarity Search (FAISS)
  * **Frontend:** HTML5, CSS3, JavaScript (Fetch API)
  * **Database:** Google Sheets (for ticket logging)

-----

## Setup and Installation

Follow these steps to get the project running on your local machine.

### Prerequisites

  * Python 3.8 or higher
  * Git

### 1\. Clone the Repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2\. Create a Python Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3\. Install Dependencies

Install all the required Python libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

*(Note: If you don't have a `requirements.txt` file, create one with `pip freeze > requirements.txt` after installing the libraries manually.)*

### 4\. Prepare Your Knowledge Base & Index

This chatbot relies on your custom documents. You must create the FAISS index from your data.

1.  Place your source text files (e.g., `.txt`, `.md`) in a designated folder.
2.  Run the script responsible for processing these documents and creating the index. (You would typically have a separate `create_index.py` script for this).
3.  This process should generate two files in your root directory:
      * `faiss_index.bin`: The FAISS vector index.
      * `faiss_meta.pkl`: A file containing the metadata (text content, headings) for each chunk.

### 5\. Google Cloud & Sheets Setup

This is required for the "Contact Us" feature.

\<details\>
\<summary\>\<b\>Click to expand for Google Cloud setup instructions\</b\>\</summary\>

1.  **Create a Google Cloud Project:** Go to the [Google Cloud Console](https://console.cloud.google.com/) and create a new project.
2.  **Enable APIs:** In your project, go to "APIs & Services" -\> "Library" and enable the **Google Drive API** and the **Google Sheets API**.
3.  **Create a Service Account:**
      * Go to "APIs & Services" -\> "Credentials".
      * Click "+ Create Credentials" -\> "Service account".
      * Give it a name (e.g., `chatbot-sheets-writer`) and click "Create and Continue".
      * Assign it the role of **Editor**.
4.  **Generate a JSON Key:**
      * Click on your newly created service account, go to the "KEYS" tab.
      * Click "Add Key" -\> "Create new key", select **JSON**, and create it.
      * A `.json` file will download. **Rename it to `credentials.json`** and place it in the root of your project directory.
      * **IMPORTANT:** Add `credentials.json` to your `.gitignore` file to keep it private.
5.  **Create and Share the Google Sheet:**
      * Create a new Google Sheet. Let's say you name it `Chatbot Support Tickets`.
      * Add headers in the first row: `Name`, `Email`, `Question`, `Timestamp`.
      * Open `credentials.json` and copy the `client_email` address.
      * Click the "Share" button in your Google Sheet and share it with that email address, giving it **Editor** permissions.

\</details\>

### 6\. Configure API Keys

1.  **Gemini API Key:**

      * Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
      * Open `app.py` and replace the placeholder value for `GEMINI_API_KEY`.

    <!-- end list -->

    ```python
    GEMINI_API_KEY = "YOUR_API_KEY_HERE"
    ```

2.  **Google Sheet Name:**

      * In `app.py`, make sure the `GOOGLE_SHEET_NAME` variable exactly matches the name of the Google Sheet you created.

    <!-- end list -->

    ```python
    GOOGLE_SHEET_NAME = "Chatbot Support Tickets"
    ```

-----

## Running the Application

Once the setup is complete, you can start the Flask development server.

```bash
python app.py
```

The application will be running on port 5001 by default. Open your web browser and navigate to:

**[http://127.0.0.1:5001](https://www.google.com/search?q=http://127.0.0.1:5001)**

You can now interact with your chatbot\!