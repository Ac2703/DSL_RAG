from langchain_huggingface import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
import uuid

class Embedder:
    def __init__(
        self,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        collection_name="course_syllabi",
        host="localhost",
        port=6333,
        vector_size=384,
    ):
        self.embedder = HuggingFaceEmbeddings(model_name=model_name)
        self.collection_name = collection_name
        self.client = QdrantClient(host=host, port=port)
        self.vector_size = vector_size
        self._ensure_collection()

    def _ensure_collection(self):
        if not self.client.get_collection(self.collection_name, allow_missing=True):
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config={"size": self.vector_size, "distance": "Cosine"},
            )

    def embed_documents(self, docs):
        texts = [doc.page_content for doc in docs]
        vectors = self.embedder.embed_documents(texts)

        embeddings = []
        for doc, vector in zip(docs, vectors):
            embeddings.append({
                "embedding": vector,
                "metadata": doc.metadata,
                "text": doc.page_content
            })
        return embeddings

    def upsert_embeddings(self, embedded_chunks):
        points = []
        for chunk in embedded_chunks:
            point_id = str(uuid.uuid4())
            payload = chunk["metadata"].copy()
            payload["text"] = chunk["text"]
            points.append(PointStruct(id=point_id, vector=chunk["embedding"], payload=payload))

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"Upserted {len(points)} points to Qdrant collection '{self.collection_name}'.")
