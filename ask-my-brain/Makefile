.PHONY: setup run dev init-db clean

setup:
	pip install -r requirements.txt
	python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
	@echo "Setup complete. Copy .env.example to .env and fill in your MIMO_API_KEY."

run:
	streamlit run app.py

dev:
	set ASK_MY_BRAIN_DEV=1 && streamlit run app.py

init-db:
	python -c "import chromadb; chromadb.PersistentClient(path='./chroma_db'); print('ChromaDB initialized.')"

clean:
	rm -rf chroma_db/
	rm -rf data/documents/*
	rm -rf logs/*
	@echo "Cleaned database, documents and logs."
