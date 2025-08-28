import pickle

def view_chunks(meta_path='faiss_meta.pkl'):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    for i, chunk in enumerate(meta, 1):
        print(f"Chunk {i}:")
        print(f"  Heading: {chunk['heading']}")
        print(f"  Content:\n{chunk['content']}\n{'-'*40}")

if __name__ == '__main__':
    view_chunks()
    