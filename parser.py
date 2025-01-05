from llama_parse import LlamaParse
from pathlib import Path
import asyncio  # Required to run async functions

async def parse_document():
    instruction = """The document is a research paper implementing a specific methodology in the field of Machine Learning. Retrieve all the present text, equations, and tables in the paper.
    The paper includes detailed methodologies, results from numerous tests, discussions on the impact, and conclusions. 
    It contains many tables. Answer questions using the information in this article and be precise. Don't miss any line in the paper.
    """

    parser = LlamaParse(
        api_key="llx-h662nb1dXzMC7r7KMTLbhwpv3pXLE19MK1pqF1Cogx7O9AyM",
        result_type="markdown",
        parsing_instruction=instruction,
        max_timeout=5000,
    )

    llama_parse_documents = await parser.aload_data('<path_to_pdf>')

    parsed_doc = llama_parse_documents[0]

    document_path = Path('parsed_doc.md')
    with document_path.open('w') as f:
        f.write(parsed_doc.text)

# Run the async function
if __name__ == "__main__":
    asyncio.run(parse_document())
