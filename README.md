# Semantic Search Engine

This project does semantic search built using embeddings. It searches documents based on their meaning using cosine similarity instead of matching exact keywords- as it done by LLM's

At the end it also visualises the document embeddings using PCA to show how different topics are grouped together, I have used 6 different topics across.

While working on the project, encountered two errors- the API key was incorrect, and the plot embedding was not getting impemented. Checked the .env file and the plot function.

## How to run

Install the required packages:

```bash
pip install -r requirements.txt
```

Add your NVIDIA API key to the `.env` file.

You can run the project in two ways:

- Open and run `semantic_search_starter.ipynb`
- Or run the program directly:

```bash
python search.py
```

Type your query to search the documents and type `quit` to exit.