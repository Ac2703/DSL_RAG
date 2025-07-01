from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import Filter, FieldCondition, MatchValue


class Embedder:
    def __init__(self, recreate=False, collection_name="syllabus_collection"):
        # docs (list[Document]): LangChain Document objects with .page_content and .metadata
        self.docs = ""
        self.collection_name = collection_name
        self.vectorstore = None

        # Step 1: Use LangChain-compatible embedding model
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # Step 2: Start Qdrant (in-memory for testing)
        self.client = QdrantClient(path="./qdrant_data")

        # If recreate is True, handle the old logic
        if recreate:
            self._handle_recreate_logic()

    def _handle_recreate_logic(self):
        """Handle the original recreate logic from __init__"""
        if self.check_collection_exists():
            recreate_answer = input("Collection exists. Recreate anyway? (y/n)")
            if recreate_answer.strip().lower().startswith("y"):
                # Delete existing collection
                self.client.delete_collection(self.collection_name)
                print(f"Deleted existing collection: {self.collection_name}")
                self.create_vectorstore()
            else:
                # Use existing collection
                self.vectorstore = QdrantVectorStore(
                    client=self.client,
                    collection_name=self.collection_name,
                    embeddings=self.embedding_model,
                )
        else:
            print("Collection not found. Creating from scratch.")
            self.create_vectorstore()

    def check_collection_exists(self):
        """Check if the collection exists in Qdrant"""
        collections = self.client.get_collections()
        return any(c.name == self.collection_name for c in collections.collections)

    def create_vectorstore(self):
        """Create the vectorstore (will create collection if it doesn't exist)"""
        try:
            # Try different parameter names based on version
            try:
                # Try with 'embedding' (singular) first
                self.vectorstore = QdrantVectorStore(
                    client=self.client,
                    collection_name=self.collection_name,
                    embedding=self.embedding_model,
                )
            except TypeError:
                # If that fails, try with 'embeddings' (plural)
                self.vectorstore = QdrantVectorStore(
                    client=self.client,
                    collection_name=self.collection_name,
                    embeddings=self.embedding_model,
                )
            print(f"Vectorstore setup complete for collection: {self.collection_name}")
        except Exception as e:
            print(f"Error creating vectorstore: {e}")
            # Alternative approach - create collection manually if it doesn't exist
            if not self.check_collection_exists():
                print("Attempting to create collection manually...")
                try:
                    # Create collection with vector configuration
                    vector_size = 384  # all-MiniLM-L6-v2 has 384 dimensions
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
                    )
                    print(f"Collection '{self.collection_name}' created manually.")
                    
                    # Now try creating vectorstore again
                    self.vectorstore = QdrantVectorStore(
                        client=self.client,
                        collection_name=self.collection_name,
                        embedding=self.embedding_model,
                    )
                    print(f"Vectorstore setup complete for collection: {self.collection_name}")
                except Exception as e2:
                    print(f"Failed to create collection manually: {e2}")
                    raise e2
            else:
                raise e

    def recreate_collection(self):
        """Delete existing collection and create a new one"""
        if self.check_collection_exists():
            self.client.delete_collection(self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
        
        self.create_vectorstore()
        print(f"Recreated collection: {self.collection_name}")

    def store_chunks(self, course_dict):
        """Store document chunks in the vector store"""
        if self.vectorstore is None:
            print("Error: Vectorstore not initialized. Call create_vectorstore() first.")
            return
            
        print(" >>>> storing chunks as vectors")
        self.docs = course_dict
        # Embed and store LangChain Document objects into Qdrant.
        self.vectorstore.add_documents(self.docs)

    def get_collection_info(self):
        """Get basic information about the collection"""
        if not self.check_collection_exists():
            print(f"Collection '{self.collection_name}' does not exist.")
            return None
        
        collection_info = self.client.get_collection(self.collection_name)
        print(f"\n=== Collection Info: {self.collection_name} ===")
        print(f"Points count: {collection_info.points_count}")
        print(f"Vector size: {collection_info.config.params.vectors.size}")
        print(f"Distance metric: {collection_info.config.params.vectors.distance}")
        print("=" * 50)
        return collection_info

    def get_sample_documents(self, limit=5):
        """Get a sample of documents from the vector store"""
        if self.vectorstore is None:
            print("Error: Vectorstore not initialized.")
            return []
        
        try:
            # Use similarity search with a generic query to get some documents
            results = self.vectorstore.similarity_search("", k=limit)
            
            print(f"\n=== Sample Documents (showing {len(results)} documents) ===")
            for i, doc in enumerate(results, 1):
                print(f"\nDocument {i}:")
                print(f"Content preview: {doc.page_content[:200]}...")
                print(f"Metadata: {doc.metadata}")
                print("-" * 50)
            
            return results
        except Exception as e:
            print(f"Error retrieving sample documents: {e}")
            return []

    def get_unique_metadata_values(self):
        """Get unique values for metadata fields"""
        if self.vectorstore is None:
            print("Error: Vectorstore not initialized.")
            return {}
        
        try:
            # Get all points to analyze metadata
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=1000,  # Adjust based on your collection size
                with_payload=True,
                with_vectors=False
            )
            
            metadata_summary = {
                "course_nums": set(),
                "course_sems": set(),
                "course_names": set(),
                "sources": set()
            }
            
            for point in points:
                payload = point.payload
                if "course_num" in payload:
                    metadata_summary["course_nums"].add(payload["course_num"])
                if "course_sem" in payload:
                    metadata_summary["course_sems"].add(payload["course_sem"])
                if "course_name" in payload:
                    metadata_summary["course_names"].add(payload["course_name"])
                if "source" in payload:
                    metadata_summary["sources"].add(payload["source"])
            
            print(f"\n=== Metadata Summary ===")
            print(f"Unique Course Numbers ({len(metadata_summary['course_nums'])}): {sorted(metadata_summary['course_nums'])}")
            print(f"Unique Course Semesters ({len(metadata_summary['course_sems'])}): {sorted(metadata_summary['course_sems'])}")
            print(f"Unique Course Names ({len(metadata_summary['course_names'])}): {sorted(metadata_summary['course_names'])}")
            print(f"Total Sources: {len(metadata_summary['sources'])}")
            print("=" * 50)
            
            return metadata_summary
            
        except Exception as e:
            print(f"Error analyzing metadata: {e}")
            return {}

    def search_by_course(self, course_num=None, course_sem=None, limit=10):
        """Search for all documents matching specific course criteria"""
        if self.vectorstore is None:
            print("Error: Vectorstore not initialized.")
            return []
        
        # Build metadata filter
        metadata_filter = {}
        if course_num:
            metadata_filter["course_num"] = course_num
        if course_sem:
            metadata_filter["course_sem"] = course_sem
        
        if metadata_filter:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key=key, match=MatchValue(value=value))
                    for key, value in metadata_filter.items()
                ]
            )
            
            try:
                # Get points with the filter
                points, _ = self.client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=qdrant_filter,
                    limit=limit,
                    with_payload=True,
                    with_vectors=False
                )
                
                print(f"\n=== Documents for Course: {course_num or 'Any'}, Semester: {course_sem or 'Any'} ===")
                print(f"Found {len(points)} documents")
                
                for i, point in enumerate(points, 1):
                    payload = point.payload
                    content = payload.get("page_content", "No content")
                    print(f"\nDocument {i}:")
                    print(f"Content preview: {content[:150]}...")
                    print(f"Course: {payload.get('course_num', 'N/A')} - {payload.get('course_sem', 'N/A')}")
                    print(f"Source: {payload.get('source', 'N/A')}")
                    print("-" * 50)
                
                return points
                
            except Exception as e:
                print(f"Error searching by course: {e}")
                return []
        else:
            print("Please provide at least course_num or course_sem")
            return []

    def search(self, query_text, sem_filter=None, course_filter=None, k=3):
        """Search for similar documents with optional metadata filtering"""
        if self.vectorstore is None:
            print("Error: Vectorstore not initialized. Call create_vectorstore() first.")
            return []
            
        print(" > embedding query and retrieving relevant docs now")

        # Build metadata filter
        metadata_filter = {}
        if sem_filter:
            metadata_filter["metadata.course_sem"] = sem_filter
        if course_filter:
            metadata_filter["metadata.course_num"] = course_filter

        if metadata_filter:
            qdrant_filter = Filter(
                must=[
                    FieldCondition(key=key, match=MatchValue(value=value))
                    for key, value in metadata_filter.items()
                ]
            )
        else:
            qdrant_filter = None

        # Run a vector search with optional metadata filter
        results = self.vectorstore.similarity_search(
            query_text,
            k=k,
            filter=qdrant_filter
        )

        return results
    

if __name__ == "__main__":
    from langchain_core.documents import Document

    # Create an instance
    embedder = Embedder()

    # Check if collection exists
    exists = embedder.check_collection_exists()
    print(f"Collection exists? {exists}")

    # Ask user whether to recreate the collection if it exists
    if exists:
        choice = input("Collection exists. Recreate? (y/n): ").strip().lower()
        if choice.startswith("y"):
            embedder.recreate_collection()
        else:
            embedder.create_vectorstore()
    else:
        embedder.create_vectorstore()

    # Prepare some dummy documents (LangChain Document objects)
    dummy_docs = [
        Document(page_content="Qdrant is a vector database.", metadata={"course_sem": "Fall 2023", "course_no": "CSc101"}),
        Document(page_content="LangChain helps build LLM apps.", metadata={"course_sem": "Spring 2023", "course_no": "CSc102"}),
        Document(page_content="Embeddings convert text into vectors.", metadata={"course_sem": "Fall 2023", "course_no": "CSc101"}),
    ]

    # Store dummy documents
    embedder.store_chunks(dummy_docs)

    # Search without filter
    print("\nSearch results without filter:")
    results = embedder.search("vector database")
    for doc in results:
        print(f"- {doc.page_content} | metadata: {doc.metadata}")

    # Search with metadata filter
    print("\nSearch results with semester filter:")
    results = embedder.search("vector database", sem_filter="Fall 2023")
    for doc in results:
        print(f"- {doc.page_content} | metadata: {doc.metadata}")