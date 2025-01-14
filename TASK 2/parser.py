from llama_parse import LlamaParse
from pathlib import Path
import asyncio  # Required to run async functions


async def parse_document(input_folder: str, output_folder: str):
    # Prepare paths
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

    # Instruction for parsing
    instruction = """The document is a research paper implementing a specific methodology in the field of Machine Learning. 
        Retrieve all the present text, equations, and tables in the paper.
        The paper includes detailed methodologies, results from numerous tests, discussions on the impact, and conclusions. 
        It contains many tables. Answer questions using the information in this article and be precise. 
        Don't miss any line in the paper.
    """

    # Initialize parser
    parser = LlamaParse(
        api_key="llx-h662nb1dXzMC7r7KMTLbhwpv3pXLE19MK1pqF1Cogx7O9AyM",
        result_type="markdown",
        parsing_instruction=instruction,
        max_timeout=50000,
    )

    # Process each PDF in the input folder
    for pdf_file in input_path.glob("*.pdf"):
        print(f"Processing {pdf_file.name}...")
        try:
            # Parse the PDF
            llama_parse_documents = await parser.aload_data(str(pdf_file))
            parsed_doc = llama_parse_documents[0]

            # Save the parsed output
            output_file = output_path / f"{pdf_file.stem}_parsed.md"
            with output_file.open("w") as f:
                f.write(parsed_doc.text)
            print(f"Saved parsed document to {output_file}")
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")
        

