import os
import pandas as pd
from langchain_groq import ChatGroq
import json
import markdown
import bs4
from typing import Dict, List, Any
import yaml
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import hashlib
from tqdm import tqdm
import time
import asyncio
from collections import deque
from datetime import datetime, timedelta
import os
import asyncio
import pandas as pd


class TokenBucket:
    def __init__(self, tokens_per_minute: int = 5000):
        self.capacity = tokens_per_minute
        self.tokens = tokens_per_minute
        self.last_updated = datetime.now()
        self.tokens_per_minute = tokens_per_minute
        self.lock = asyncio.Lock()

    async def get_tokens(self, requested_tokens: int) -> bool:
        async with self.lock:
            now = datetime.now()
            time_passed = (now - self.last_updated).total_seconds() / 60.0
            
            # Refill tokens based on time passed
            self.tokens = min(
                self.capacity,
                self.tokens + (self.tokens_per_minute * time_passed)
            )
            self.last_updated = now

            if self.tokens >= requested_tokens:
                self.tokens -= requested_tokens
                return True
            return False

    async def wait_for_tokens(self, requested_tokens: int):
        while not await self.get_tokens(requested_tokens):
            await asyncio.sleep(1)

class PaperContentExtractor:
    def __init__(self, groq_api_key: str, max_workers: int = 3, cache_dir: str = None):
        self.llm = ChatGroq(
            temperature=0,
            api_key=groq_api_key,
            model_name="mixtral-8x7b-32768"
        )
        self.max_workers = max_workers
        self.token_bucket = TokenBucket(tokens_per_minute=5000)
        self.cache_dir = cache_dir or os.path.join(os.getcwd(), 'feature_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.section_keywords = {
            'introduction': ['introduction', 'background', 'overview', 'abstract'],
            'methodology': ['method', 'approach', 'implementation', 'proposed', 'architecture', 'system', 'model'],
            'results': ['result', 'evaluation', 'experiment', 'performance', 'analysis', 'finding'],
            'conclusion': ['conclusion', 'discussion', 'future', 'summary']
        }

    @staticmethod
    def _get_section_hash(content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()
    
    def _split_by_sections(self, content: str) -> List[tuple]:
        sections = []
        lines = content.split('\n')
        current_section = None
        current_content = []
        
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            
            # Check for markdown headers (# style)
            header_match = re.match(r'^#+\s+(.+)$', line)
            if header_match:
                if current_section and current_content:
                    sections.append((current_section, '\n'.join(current_content)))
                current_section = header_match.group(1).strip()
                current_content = []
            elif line and current_section is not None:
                current_content.append(line)
            elif line and not sections:  # Content before first heading
                current_section = "Introduction"
                current_content.append(line)
            
            i += 1
        
        # Add the last section
        if current_section and current_content:
            sections.append((current_section, '\n'.join(current_content)))
        
        print(f"Found sections: {[section[0] for section in sections]}")
        return sections

    def _get_cache_path(self, section_hash: str) -> str:
        return os.path.join(self.cache_dir, f"{section_hash}.json")

    def _load_from_cache(self, section_hash: str) -> Dict:
        cache_path = self._get_cache_path(section_hash)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except:
                return None
        return None

    def _save_to_cache(self, section_hash: str, features: Dict):
        cache_path = self._get_cache_path(section_hash)
        with open(cache_path, 'w') as f:
            json.dump(features, f)

    def _get_empty_features(self) -> Dict:
        return {
            "Topic of Research": "",
            "Research Objective": "",
            "Methodology": "",
            "Results": "",
            "Novelty Claims": "",
            "Evaluation Metrics": "",
            "Category of Research": ""
        }

    async def _extract_section_features(self, section: str) -> Dict:
        # Estimate tokens (rough approximation)
        estimated_tokens = len(section.split()) * 1.5
        
        await self.token_bucket.wait_for_tokens(estimated_tokens)
        
        prompt = f"""Analyze this research paper section and extract key information.
        Return a JSON object with these keys (use empty string if information is not found):
        {{
            "Topic of Research": "The main research topic or focus area",
            "Research Objective": "The specific goals or objectives",
            "Methodology": "Methods, approaches, or techniques used",
            "Results": "Key findings or outcomes",
            "Novelty Claims": "Claims about new contributions or innovations",
            "Evaluation Metrics": "Metrics used to evaluate results",
            "Category of Research": "Type of research (e.g., empirical, theoretical, applied)"
        }}

        Section content:
        {section}"""
        
        try:
            response = self.llm.invoke(prompt)
            try:
                return json.loads(response.content)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON response: {response.content[:200]}...")
                return self._get_empty_features()
        except Exception as e:
            if "rate_limit_exceeded" in str(e):
                print("Rate limit exceeded, waiting before retry...")
                await asyncio.sleep(2)
                return await self._extract_section_features(section)
            print(f"Error in LLM call: {str(e)}")
            return self._get_empty_features()

    async def _process_section(self, section_tuple: tuple) -> Dict:
        header, content = section_tuple
        if len(content.strip()) < 50:
            return self._get_empty_features()
        
        section_hash = self._get_section_hash(content)
        
        # Try to load from cache first
        cached_features = self._load_from_cache(section_hash)
        if cached_features:
            return cached_features
        
        features = await self._extract_section_features(content)
        self._save_to_cache(section_hash, features)
        
        print(f"Processed section: {header[:50]}...")
        return features

    def _merge_features(self, features_list: List[Dict]) -> Dict:
        if not features_list:
            return self._get_empty_features()
            
        merged = self._get_empty_features()
        
        for key in merged:
            values = [f[key] for f in features_list if f[key] and f[key].strip()]
            if values:
                merged[key] = max(values, key=len)
        
        return merged

    async def extract_paper_features(self, file_path: str) -> Dict[str, Any]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        sections = self._split_by_sections(content)
        print(f"Found {len(sections)} sections in {file_path}")
        
        features_list = []
        tasks = [self._process_section(section) for section in sections]
        features_list = await asyncio.gather(*tasks)
        
        merged_features = self._merge_features([f for f in features_list if any(f.values())])
        return merged_features

async def process_papers(papers_dir: str, output_dir: str, groq_api_key: str = None, max_workers: int = 3):
    if not groq_api_key:
        raise ValueError("Please provide GROQ_API_KEY")
            
    extractor = PaperContentExtractor(groq_api_key, max_workers)
    os.makedirs(output_dir, exist_ok=True)
    
    columns = [
        "Paper Code", "Topic of Research", "Research Objective", "Methodology",
        "Results", "Novelty Claims", "Evaluation Metrics",
        "Category of Research"
    ]
    
    paper_data = []
    filenames = []
    md_files = [f for f in os.listdir(papers_dir) if f.endswith('.md')]
    
    if not md_files:
        print(f"No markdown files found in {papers_dir}")
        return pd.DataFrame(columns=columns)
    
    for filename in tqdm(md_files, desc="Processing papers"):
        paper_path = os.path.join(papers_dir, filename)
        output_path = os.path.join(output_dir, f"{filename[:-3]}_features.yaml")
        
        try:
            features = await extractor.extract_paper_features(paper_path)
            paper_data.append(features)
            
            with open(output_path, 'w') as f:
                yaml.dump(features, f)
                
            print(f"Successfully processed {filename}")
            filenames.append(filename)
            df_paper = pd.DataFrame(paper_data, columns=columns)
            df_paper['filename'] = filenames
            df_paper.to_csv('paper_df.csv', index=False)
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # df_paper = pd.DataFrame(paper_data, columns=columns)
    # df_paper['filename'] = filenames
    return df_paper


async def main():
    # Configuration
    PAPERS_DIR = "/path/to/markdown/papers"  # Directory containing markdown papers
    OUTPUT_DIR = "/path/for/output/features"  # Directory for output features
    GROQ_API_KEY = "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"
    MAX_WORKERS = 3


    if not os.path.exists(PAPERS_DIR):
        raise FileNotFoundError(f"Papers directory '{PAPERS_DIR}' not found")

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        # Process papers and get results DataFrame
        df = await process_papers(
            papers_dir=PAPERS_DIR,
            output_dir=OUTPUT_DIR,
            groq_api_key=GROQ_API_KEY,
            max_workers=MAX_WORKERS
        )

        # Save results
        results_path = os.path.join(OUTPUT_DIR, "paper_analysis_results.csv")
        df.to_csv(results_path, index=False)
        print(f"Analysis complete. Results saved to {results_path}")

        # Print summary
        print("\nProcessing Summary:")
        print(f"Total papers processed: {len(df)}")
        print(f"Research categories found: {df['Category of Research'].unique()}")
        
    except Exception as e:
        print(f"Error during paper processing: {str(e)}")
        raise

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
