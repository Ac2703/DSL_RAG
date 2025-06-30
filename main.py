from modules.webscraping import WebScraper
from modules.preprocessing import Chunker
from modules.embedding import Embedder
import os

class RAG:
    def __init__(self, main_url):
        self.main_url = main_url
        self.scraper = WebScraper(main_url)
        self.chunker = Chunker()
        self.embedder = None  # Initialize later

    def print_banner(self):
        print(r"""
            ╔═══════════════════════════════════════╗
            ║       DSL RAG SYSTEM INITIALIZED      ║
            ╠═══════════════════════════════════════╣
            ║   A Document-Retrieval Pipeline for   ║
            ║       Semantic Search + Filtering     ║
            ╠═══════════════════════════════════════╣
            ║     Powered by LangChain + Qdrant     ║
            ╚═══════════════════════════════════════╝
            """)

    def setup_embedder_and_check_collection(self):
        """Initialize embedder and check if collection exists"""
        # Initialize embedder without recreating
        self.embedder = Embedder(recreate=False)
        
        if self.embedder.check_collection_exists():
            recreate_answer = input("Collection exists. Do you want to re-scrape and chunk and create a vector store from scratch? (y/n): ")
            if recreate_answer.strip().lower().startswith("y"):
                # Delete existing collection and recreate
                self.embedder.recreate_collection()
                return True  # Need to index docs
            else:
                print("Using existing vector store.")
                # Initialize vectorstore to use existing collection
                self.embedder.create_vectorstore()
                return False  # Use existing store
        else:
            print("Collection not found. Will create from scratch.")
            # Create vectorstore (will create collection if it doesn't exist)
            self.embedder.create_vectorstore()
            return True  # Need to index docs

    def Index_Docs(self, csv_file="./data/course_chunks.csv"):
        # Delete CSV once before starting batch processing
        if os.path.exists(csv_file):
            os.remove(csv_file)
            print(f"> {csv_file} deleted at program start to start fresh\n")
        else:
            print(f"> {csv_file} does not exist, starting fresh\n")

        ############### MAIN LOOP ###############
        # STEP 1. GET ALL COURSES
        main_page_links = self.scraper.scrape_main_page() 

        for course_dict in main_page_links:
            # STEP 2. CHUNK DATA
            chunks = self.chunker.process(course_dict) # create chunks
            self.chunker.save_chunks_to_csv(chunks, filename=csv_file) # save chunks

            # STEP 3. EMBED DATA
            self.embedder.store_chunks(chunks)
        
        ##############################

    def Retrieve_Docs(self, query, sem_filter, course_filter):
        if self.embedder is None:
            print("Error: Embedder not initialized. Please run setup first.")
            return
            
        results = self.embedder.search(query, sem_filter, course_filter)
        
        print(f"\nResults for query: '{query}'")
        for doc in results:
            print("Text:", doc.page_content)
            print("Metadata:", doc.metadata)
            print("---")


def main():
    main_url = "https://tinman.cs.gsu.edu/~raj/past2.html"   
    rag = RAG(main_url) 

    rag.print_banner()

    # Check collection and determine if we need to index
    need_to_index = rag.setup_embedder_and_check_collection()
    
    if need_to_index:
        print("Indexing documents...")
        rag.Index_Docs()
        print("Indexing complete!")
    else:
        print("Ready to search with existing data!")

    # GET QUERY
    while True:
        course_num = input("Enter Course Num (e.g. CSc 1302): ")
        if course_num == "":  # Check if the input is an empty string
            print("Exiting program.")
            break  # Exit the loop and terminate the program

        sem_filter = input("Enter Fall/Spring/Summer and 4-digit year (e.g. Fall 2016): ")
        if sem_filter == "":  # Check if the input is an empty string
            print("Exiting program.")
            break  # Exit the loop and terminate the program

        query = input("Enter Question: ")
        if query == "":  # Check if the input is an empty string
            print("Exiting program.")
            break  # Exit the loop and terminate the program
        else: 
            rag.Retrieve_Docs(query, sem_filter, course_num)


if __name__ == "__main__":
    main()