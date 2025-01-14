import os
from extract_and_format_sections import extract_and_format_sections
from langgraph.graph import StateGraph
from agentic_workflow import State, conference_recommendation_node, final_decision_node
from parser import parse_document
import asyncio
import pandas as pd


outputs = {}

def convert_to_list_of_dicts(file_dict):
    """
    Convert a dictionary with file paths as keys and content as values
    into a list of dictionaries with filename and content fields.
    
    Parameters:
    file_dict (dict): Dictionary with file paths as keys and content as values
    
    Returns:
    list: List of dictionaries with filename and content fields
    """
    result = []
    
    for filepath, content in file_dict.items():
        # Extract just the filename from the path
        filename = filepath.split('/')[-1]
        
        # Create a dictionary for this file
        file_info = {
            'filename': filename,
            'content': content
        }
        
        result.append(file_info)
    
    return result


def process_md_file(file_path):
    # Load the markdown content
    with open(file_path, 'r') as file:
        content = file.read()

    # Extract and format sections
    formatted_text = extract_and_format_sections(content)
    print(f"Processed content from {file_path}:\n{formatted_text}")

    # Prepare the input state
    input_state = {
        "paper_text": formatted_text,
        "all_analyses": [],
        "final_recommendation": ""
    }

    # Invoke the app
    result = app.invoke(input_state)
    print("Final Recommendation for", file_path, ":", result["final_recommendation"])
    outputs[file_path] = result["final_recommendation"]
    print("-" * 80)

if __name__ == "__main__":

    # Run the async function
    input_folder = "input"  # Replace with the path to your folder containing PDFs
    output_folder = 'target'  # Replace with the path to save the parsed markdown files

    asyncio.run(parse_document(input_folder, output_folder))
    # Build the graph
    graph = StateGraph(State)
    graph.add_node("conference_recommendation", conference_recommendation_node)
    graph.add_node("final_decision", final_decision_node)
    graph.add_edge("conference_recommendation", "final_decision")
    graph.set_entry_point("conference_recommendation")

    # Compile the graph
    app = graph.compile()

    # Iterate over all files in the folder
    for filename in os.listdir(output_folder):
        if filename.endswith(".md"):
            file_path = os.path.join(output_folder, filename)
            process_md_file(file_path)
    print(outputs)
    converted_list = convert_to_list_of_dicts(outputs)
    df = pd.DataFrame(converted_list)
    df.to_csv('results.csv', index=False)