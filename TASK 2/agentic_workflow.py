from langchain import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain_groq import ChatGroq
from langchain.utilities import ArxivAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph
from typing import TypedDict, List
from cachetools import TTLCache
import hashlib
from langchain_together import ChatTogether
from concurrent.futures import ThreadPoolExecutor
from pathway.xpacks.llm.vector_store import VectorStoreClient


PATHWAY_PORT = 8765


client = VectorStoreClient(
    host="127.0.0.1",
    port=PATHWAY_PORT,
)

# Cache setup
_CACHE_MAX_SIZE = 50
_CACHE_TTL = 3600
_cache = TTLCache(maxsize=_CACHE_MAX_SIZE, ttl=_CACHE_TTL)

API_KEYS = [
    "<API_KEY1>",
    "<API_KEY1>",
    "<API_KEY2>",
    "<API_KEY2>"
]

class State(TypedDict):
    paper_text: str
    all_analyses: List[str]
    final_recommendation: str

# Initialize the three LLMs with different API keys
llm1 = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key = API_KEYS[0]
)

llm1 = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key = API_KEYS[1]
)
llm1 = ChatTogether(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key = API_KEYS[2]
)

conferences = ["NeurIPS", "CVPR", "EMNLP", "KDD", "TMLR"]

# Conference instructions dictionary remains the same
conference_instructions = {
    "NeurIPS": """
    NeurIPS focuses on advances in neural information processing systems, including but not limited to:
    - Deep learning and representation learning.
    - Reinforcement learning and decision-making.
    - Generative models and unsupervised learning.
    - AI ethics, fairness, and societal impact.
    - Optimization methods for machine learning.
    - Applications of machine learning in science, healthcare, and industry.
    """,
    "CVPR": """
    CVPR focuses on computer vision, image processing, and pattern recognition, including but not limited to:
    - Image and video understanding (e.g., object detection, segmentation, tracking).
    - 3D vision and reconstruction.
    - Vision for robotics and autonomous systems.
    - Generative models for images and videos.
    - Applications of computer vision in healthcare, surveillance, and entertainment.
    """,
    "EMNLP": """
    EMNLP focuses on natural language processing (NLP), including but not limited to:
    - Text analysis (e.g., sentiment analysis, named entity recognition).
    - Machine translation and multilingual NLP.
    - Question answering and dialogue systems.
    - Text summarization and generation.
    - Applications of NLP in healthcare, education, and social media.
    """,
    "KDD": """
    KDD focuses on knowledge discovery and data mining, including but not limited to:
    - Data mining algorithms (e.g., clustering, classification, association rule mining).
    - Anomaly detection and outlier analysis.
    - Graph mining and network analysis.
    - Applications of data mining in finance, healthcare, and social networks.
    - Big data analytics and scalable machine learning.
    """,
    "TMLR": """
    TMLR focuses on theoretical and applied machine learning research, including but not limited to:
    - Theoretical foundations of machine learning (e.g., generalization, optimization).
    - Federated learning and distributed machine learning.
    - Robustness, interpretability, and fairness in machine learning.
    - Applications of machine learning in science, engineering, and industry.
    - Novel machine learning models and algorithms.
    """
}

react_prompt = PromptTemplate(
    input_variables=["paper_text", "conference", "similar_papers"],
    template="""
    You are a conference recommendation agent. Your task is to analyze whether the given research paper aligns with the theme of the {conference} conference.

    Paper Text: {paper_text}

    Similar Papers from {conference}:
    {similar_papers}

    {conference_instructions}

    Instructions:
    1. Identify the key topics and methodologies in the paper.
    2. Compare the paper's content with the typical themes of {conference}.
    3. Analyze how the paper relates to the similar papers retrieved from {conference}.
    4. Provide reasons why the paper should belong to {conference}.
    5. Provide reasons why the paper should not belong to {conference}.
    6. Make a decision based on the alignment of the paper with the conference theme.

    Output your analysis and decision below:
    """
)

# Define sufficiency judgment prompt
sufficiency_prompt = PromptTemplate(
    input_variables=["analysis"],
    template="""
    You are a sufficiency judgment agent. Your task is to evaluate whether the provided analysis is sufficient to make a recommendation.

    Analysis: {analysis}

    Instructions:
    1. Determine if the analysis is comprehensive and well-reasoned.
    2. If the analysis is insufficient, provide suggestions for improvement.
    3. If the analysis is sufficient, confirm its validity.

    Output your judgment below:
    """
)

# Define final decision-making prompt
decision_prompt = PromptTemplate(
    input_variables=["all_analyses"],
    template="""
    You are a final decision-making agent. Your task is to analyze all the critiques and recommend the most suitable conference for the research paper.

    All Analyses: {all_analyses}

    Instructions:
    1. Compare the analyses for each conference.
    2. Identify the conference that best aligns with the paper's content.
    3. Provide a final recommendation with detailed reasoning.

    Output your recommendation below:
    """
)

class ConferenceAnalyzer:
    def __init__(self, api_key):
        self.llm =  ChatTogether(
            model="meta-llama/Llama-3-70b-chat-hf",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key = api_key
        )
        self.arxiv = ArxivAPIWrapper()
        self.search = DuckDuckGoSearchRun()
        self.tools = [
            Tool(
                name="Arxiv",
                func=self.arxiv.run,
                description="Useful for retrieving academic papers and verifying references."
            )
        ]
        self.fact_checking_agent = initialize_agent(
            self.tools, 
            self.llm, 
            verbose=False, 
            handle_parsing_errors=True
        )
        
        self.conference_chain = LLMChain(llm=self.llm, prompt=react_prompt)

    def get_cache_key(self, paper_text, k):
        text_hash = hashlib.sha256(paper_text.encode()).hexdigest()
        return f"{text_hash}_{k}"
    
    def cached_client_query(self, paper_text, k):
        cache_key = self.get_cache_key(paper_text, k)
        if cache_key in _cache:
            print(f"Cache hit for key: {cache_key}")
            return _cache[cache_key]
        else:
            print(f"Cache miss for key: {cache_key}")
            docs = client.query(paper_text, k=k)
            _cache[cache_key] = docs
        return docs

    def analyze_conference(self, conference: str, paper_text: str) -> str:
        # Get similar papers (implement your cached_client_query here)
        cache_key = f"{hashlib.sha256(paper_text.encode()).hexdigest()}_{conference}"
        if cache_key in _cache:
            similar_papers_text = _cache[cache_key]
        else:
            # Implement your similarity search here
            similar_papers_text = self.cached_client_query(paper_text, 10)  # Replace with actual implementation
            _cache[cache_key] = similar_papers_text

        # Generate analysis
        analysis = self.conference_chain.run(
            paper_text=paper_text,
            conference=conference,
            similar_papers=similar_papers_text,
            conference_instructions=conference_instructions[conference]
        )

        # Fact checking
        fact_check_result = self.fact_checking_agent.run(
            f"Verify the following analysis: {analysis}"
        )

        return analysis

# Create a mapping of conferences to API keys
conference_to_api_key = {
    "NeurIPS": API_KEYS[0],
    "CVPR": API_KEYS[1],
    "EMNLP": API_KEYS[2],
    "KDD": API_KEYS[3],
    "TMLR": API_KEYS[0]  # Reuse the first key if you have more conferences than keys
}

def conference_recommendation_node(state: State) -> State:
    paper_text = state["paper_text"]
    
    all_analyses = []
    
    # Process conferences in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(conferences)) as executor:
        # Submit tasks for initial analysis
        future_to_conf = {
            executor.submit(ConferenceAnalyzer(conference_to_api_key[conf]).analyze_conference, conf, paper_text): conf 
            for conf in conferences
        }
        
        # Collect results
        for future in future_to_conf:
            conference = future_to_conf[future]
            try:
                analysis = future.result()
                
                # Check sufficiency using a different API key
                sufficiency_llm = llm1 = ChatTogether(
                    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                    temperature=0,
                    max_tokens=None,
                    timeout=None,
                    max_retries=2,
                    api_key = API_KEYS[1]
                )
                sufficiency_chain = LLMChain(llm=sufficiency_llm, prompt=sufficiency_prompt)
                
                sufficiency_result = sufficiency_chain.run(analysis=analysis)
                
                if "insufficient" in sufficiency_result.lower():
                    # Re-analyze with more detail
                    analysis = ConferenceAnalyzer(conference_to_api_key[conference]).analyze_conference(conference, paper_text)
                
                all_analyses.append(analysis)
                
            except Exception as e:
                print(f"Error processing conference {conference}: {str(e)}")
                all_analyses.append(f"Error analyzing {conference}")
    
    return {"all_analyses": all_analyses}

def final_decision_node(state: State) -> State:
    # Use a different API key for final decision
    final_llm = ChatTogether(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=2,
            api_key = API_KEYS[2]
        )
    decision_chain = LLMChain(llm=final_llm, prompt=decision_prompt)
    
    final_recommendation = decision_chain.run(all_analyses=state["all_analyses"])
    return {"final_recommendation": final_recommendation}

# Build the graph
graph = StateGraph(State)
graph.add_node("conference_recommendation", conference_recommendation_node)
graph.add_node("final_decision", final_decision_node)
graph.add_edge("conference_recommendation", "final_decision")
graph.set_entry_point("conference_recommendation")

# Compile the graph
app = graph.compile()

