# 🦾 RAG Gemini Flask App

A Retrieval-Augmented Generation (RAG) application that lets you upload a Word (.docx) file, generates semantic embeddings, stores them in a FAISS vector database, and allows you to ask questions powered by Google's Gemini API.

---

## 🚀 Features

- 📄 **Upload any Word (.docx) file**
- 🧩 **Automatic chunking** by Heading 1 and Heading 2
- 🧠 **Embeddings** generated using Sentence Transformers
- 🗃️ **FAISS vector search** for fast, semantic retrieval
- 🤖 **Gemini API** for context-aware answer generation
- 🌐 **Modern Flask web UI** (Bootstrap)
- 🔄 **Dynamic workflow**: Upload, embed, and query new documents on the fly

---

## 🛠️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/prithvirajjadhav2266/RAG_GEMINI.git
cd RAG_GEMINI
```

### 2. Create and activate a virtual environment (Python 3.10+ recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your Gemini API key
- Set your Gemini API key as an environment variable:
	```bash
	export GEMINI_API_KEY=your-gemini-api-key  # On Mac/Linux
	set GEMINI_API_KEY=your-gemini-api-key     # On Windows
	```
- Or edit `app.py` to insert your key directly (not recommended for production).

---

## 🏃‍♂️ Running the App

```bash
python app.py
```
- Visit [http://127.0.0.1:5000/](http://127.0.0.1:5000/) in your browser.

---

## 📤 Uploading a Word File
- Go to `/upload` or click the "Upload new Word file" button.
- Select your `.docx` file and upload.
- The app will process the file, generate embeddings, and update the FAISS database.
- You can now ask questions about the new document!

---

## 🧩 How it Works

1. **Upload**: User uploads a `.docx` file.
2. **Chunking**: The file is split into chunks by Heading 1 and Heading 2.
3. **Embedding**: Each chunk is embedded using Sentence Transformers.
4. **Indexing**: Embeddings are stored in a FAISS vector database.
5. **Query**: User asks a question; the app retrieves the most relevant chunks using cosine similarity.
6. **Answer**: The context and question are sent to Gemini API, which generates a context-aware answer.

---

## 📁 Project Structure

```
RAG_GEMINI/
├── app.py                # Main Flask app
├── embed_word_to_faiss.py# Embedding script
├── requirements.txt      # Python dependencies
├── .gitignore            # Git ignore rules
├── faiss_index.bin       # FAISS vector index (auto-generated)
├── faiss_meta.pkl        # Metadata for chunks (auto-generated)
├── uploads/              # Uploaded Word files
└── ...
```

---

## 📝 Notes
- Do **not** commit your `venv/` or large model files to GitHub.
- The app is designed for `.docx` files with clear Heading 1 and Heading 2 structure.
- For large files or production, use Render or a similar ML-friendly host.

---

## 🙏 Credits
- [FAISS](https://github.com/facebookresearch/faiss)
- [Sentence Transformers](https://www.sbert.net/)
- [Flask](https://flask.palletsprojects.com/)
- [Google Gemini API](https://ai.google.dev/)

---

## 💡 License
MIT

