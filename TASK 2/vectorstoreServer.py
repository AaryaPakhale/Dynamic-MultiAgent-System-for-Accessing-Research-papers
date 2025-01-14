
import numpy as np
from sentence_transformers import SentenceTransformer
from pathway.internals import udfs
from pathway.xpacks.llm.embedders import BaseEmbedder
import logging
import pathway as pw
from pathway.xpacks.llm.splitters import TokenCountSplitter
from pathway.xpacks.llm.vector_store import VectorStoreClient, VectorStoreServer
import time

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)

PATHWAY_PORT = 8765

class EmbeddingClient(BaseEmbedder):

    def __init__(
        self,
        *,
        capacity: int = 10,
        max_retries: int = 5,
        cache_strategy: udfs.CacheStrategy = None,
    ):
        retry_strategy = udfs.ExponentialBackoffRetryStrategy(max_retries=max_retries)
        
        executor = udfs.async_executor(
            capacity=capacity,
            retry_strategy=retry_strategy,
            timeout=20.0,
        )
      
        super().__init__(
            executor=executor,
            cache_strategy=cache_strategy,
        )
   
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    async def __wrapped__(self, input: str, **kwargs) -> np.ndarray:
        
        embedding = self.model.encode(input)
        logger.info(embedding[:10])
        return embedding

data_sources = []
data_sources.append(
    pw.io.fs.read(
        "Parsed_Docs/Parsed_Docs/Reference",
        format="binary", 
        mode="streaming",
        with_metadata=True,
    )
)

text_splitter = TokenCountSplitter()
embedder = EmbeddingClient(cache_strategy=udfs.DefaultCache())

vector_server = VectorStoreServer(
    *data_sources,
    embedder=embedder,
    splitter=text_splitter,
)
vector_server.run_server(host="127.0.0.1", port=PATHWAY_PORT, threaded=True, with_cache=False)
time.sleep(5)