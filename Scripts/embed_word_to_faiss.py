import os
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

def extract_headings_and_chunks(doc_path):
    doc = Document(doc_path)
    chunks = []
    current_h1 = None
    current_h2 = None
    current_content = []
    for para in doc.paragraphs:
        if para.style.name.startswith('Heading 1'):
            # Save previous Heading 2 chunk
            if current_h2 and current_content:
                chunks.append({
                    'heading1': current_h1,
                    'heading2': current_h2,
                    'content': '\n'.join(current_content)
                })
                current_content = []
            # Save previous Heading 1 chunk (content directly under Heading 1)
            elif current_h1 and current_content:
                chunks.append({
                    'heading1': current_h1,
                    'heading2': None,
                    'content': '\n'.join(current_content)
                })
                current_content = []
            current_h1 = para.text.strip()
            current_h2 = None
        elif para.style.name.startswith('Heading 2'):
            if current_h2 and current_content:
                chunks.append({
                    'heading1': current_h1,
                    'heading2': current_h2,
                    'content': '\n'.join(current_content)
                })
                current_content = []
            current_h2 = para.text.strip()
        else:
            # Collect content for Heading 2 or directly under Heading 1
            if current_h2:
                current_content.append(para.text.strip())
            elif current_h1 and para.text.strip():
                current_content.append(para.text.strip())
    # Save last chunk
    if current_h2 and current_content:
        chunks.append({
            'heading1': current_h1,
            'heading2': current_h2,
            'content': '\n'.join(current_content)
        })
    elif current_h1 and current_content:
        chunks.append({
            'heading1': current_h1,
            'heading2': None,
            'content': '\n'.join(current_content)
        })
    return chunks

def embed_and_store(chunks, model_name='all-MiniLM-L6-v2', faiss_path='faiss_index.bin', meta_path='faiss_meta.pkl'):
    model = SentenceTransformer(model_name)
    texts = [f"{c['heading1']} | {c['heading2']} | {c['content']}" for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))
    faiss.write_index(index, faiss_path)
    # Save metadata
    meta = [{'heading': f"{c['heading1']} | {c['heading2']}", 'content': c['content']} for c in chunks]
    with open(meta_path, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Stored {len(chunks)} chunks in FAISS DB.")

if __name__ == '__main__':
    docx_file = 'AetherX Dynamics.docx'  # Update if needed
    chunks = extract_headings_and_chunks(docx_file)
    embed_and_store(chunks)
