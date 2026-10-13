from pathlib import Path

from pypdf import PdfReader


class DocumentLoader:
    """
    Loads PDF documents from the knowledge base.
    """

    def load_pdf(self, pdf_path):
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        reader = PdfReader(str(pdf_path))

        pages = []

        for page_number, page in enumerate(
            reader.pages,
            start=1,
        ):

            text = page.extract_text()

            if not text:
                continue

            pages.append(
                {
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_number,
                }
            )

        return pages

    def load_directory(self, directory):
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(
                f"Knowledge base directory not found: "
                f"{directory}"
            )

        documents = []

        for pdf_path in directory.rglob("*.pdf"):

            pages = self.load_pdf(pdf_path)

            category = pdf_path.parent.name

            for page in pages:
                page["category"] = category
                documents.append(page)

        return documents