def extract_and_format_sections(markdown_text):
    """
    Extract sections from markdown text and format them into a single text with section headers.
    
    Args:
        markdown_text (str): The content of the markdown paper
        
    Returns:
        str: Formatted text with all sections
    """
    import re

    try:
        # Extract title
        title_match = re.search(r'^#\s+(.+)$', markdown_text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else ""
        
        # Extract Abstract
        abstract_match = re.search(r'#\s*Abstract\s*\n\n(.*?)(?=\n#)', markdown_text, re.DOTALL)
        abstract = abstract_match.group(1).strip() if abstract_match else ""
        
        # Extract Introduction
        intro_match = re.search(r'#\s*(?:\d+\s+)?Introduction\s*\n\n(.*?)(?=\n#)', markdown_text, re.DOTALL)
        intro = intro_match.group(1).strip() if intro_match else ""
        
        # Extract Conclusion
        conclusion_match = re.search(r'#\s*(?:\d+\s+)?Conclusions?\s*\n\n(.*?)(?=\n#|$)', markdown_text, re.DOTALL)
        conclusion = conclusion_match.group(1).strip() if conclusion_match else ""
        
        # Format the output
        formatted_text = f"""Title:
{title}

Abstract:
{abstract}

Introduction:
{intro}

Conclusion:
{conclusion}
"""
        return formatted_text
        
    except Exception as e:
        print(f"Error extracting sections: {str(e)}")
        return None