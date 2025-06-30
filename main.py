from modules.webscraping import WebScraper, Chunker, Embedder
import os


def main():
    csv_file = "./data/course_chunks.csv"

    # Delete CSV once before starting batch processing
    if os.path.exists(csv_file):
        os.remove(csv_file)
        print(f"{csv_file} deleted at program start to start fresh")
    else:
        print(f"{csv_file} does not exist, starting fresh")

    main_url = "https://tinman.cs.gsu.edu/~raj/past2.html"
    scraper = WebScraper(main_url)

    main_page_links = scraper.scrape_main_page()  # ALL LINKS FROM MAIN PAGE
    for course_dict in main_page_links:
        pdf_chunker = Chunker(course_dict)
        chunks = pdf_chunker.process()
        if chunks:
            pdf_chunker.save_chunks_to_csv(chunks, filename=csv_file)

        embedder = Embedder()
        embedded_chunks = embedder.embed_documents(chunks)

        embedder.upsert_embeddings(embedded_chunks)

main()