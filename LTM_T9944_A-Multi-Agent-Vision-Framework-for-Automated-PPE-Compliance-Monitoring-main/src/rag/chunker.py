class TextChunker:
    """
    Splits loaded documents into overlapping chunks.
    """

    def __init__(
        self,
        chunk_size=800,
        chunk_overlap=100,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, documents):

        chunks = []

        for document in documents:

            text = document["text"]

            start = 0

            while start < len(text):

                end = start + self.chunk_size

                chunk_text = text[start:end]

                if chunk_text.strip():

                    chunks.append(
                        {
                            "text": chunk_text,
                            "source": document["source"],
                            "page": document["page"],
                            "category": document["category"],
                        }
                    )

                start = (
                    end - self.chunk_overlap
                )

        return chunks