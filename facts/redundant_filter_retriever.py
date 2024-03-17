from typing import Dict, List
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from langchain.schema import BaseRetriever
from langchain_core.documents import Document

class RedundantFilterRetriever(BaseRetriever):
    embeddings: Embeddings
    chroma: Chroma
    def get_relevant_documents(self, query):
        emb = self.embeddings.embed_query(query)
        return self.chroma.max_marginal_relevance_search_by_vector(
            embedding=emb,
            lambda_mult=0.8
        )
    
    async def aget_relevant_documents(self, query): 
        return []
