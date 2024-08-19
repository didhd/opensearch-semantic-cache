import os
import boto3
from dotenv import load_dotenv
from langchain_core.caches import BaseCache
from langchain_aws import ChatBedrock
from langchain_aws import BedrockEmbeddings
from opensearchpy import RequestsAWSV4SignerAuth, RequestsHttpConnection
from opensearchpy.exceptions import NotFoundError
from typing import Dict, Any, List, Optional
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain.schema import Generation
from json import dumps, loads
import logging
import hashlib
import time

# .env 파일에서 환경 변수 로드
load_dotenv()

# OpenSearch 엔드포인트 및 인증 정보 설정
opensearch_url = os.getenv("OPENSEARCH_URL")

# AWS 설정
aws_region = os.getenv("AWS_REGION", "ap-northeast-1")

# Bedrock 클라이언트 설정
session = boto3.Session(region_name=aws_region)
bedrock_runtime = session.client(service_name="bedrock-runtime")

# AWS4Auth 설정
credentials = session.get_credentials()
aoss_auth = RequestsAWSV4SignerAuth(
    credentials=credentials,
    region=session.region_name,
    service="aoss",
)

logger = logging.getLogger(__name__)


def _hash(_input: str) -> str:
    return hashlib.md5(_input.encode()).hexdigest()


def _load_generations_from_json(json_string: str) -> List[Generation]:
    json_object = loads(json_string)
    return [Generation(**gen) for gen in json_object]


class OpenSearchSemanticCache(BaseCache):
    """Cache that uses OpenSearch vector store backend"""

    def __init__(
        self, opensearch_url: str, embedding: Any, score_threshold: float = 0.2
    ):
        """
        Args:
            opensearch_url (str): URL to connect to OpenSearch.
            embedding (Embedding): Embedding provider for semantic encoding and search.
            score_threshold (float, 0.2):
        """
        self._cache_dict: Dict[str, OpenSearchVectorSearch] = {}
        self.opensearch_url = opensearch_url
        self.embedding = embedding
        self.score_threshold = score_threshold

    def _index_name(self, llm_string: str) -> str:
        hashed_index = _hash(llm_string)
        return f"cache_{hashed_index}"

    def _get_llm_cache(self, llm_string: str) -> OpenSearchVectorSearch:
        index_name = self._index_name(llm_string)

        # return vectorstore client for the specific llm string
        if index_name in self._cache_dict:
            return self._cache_dict[index_name]

        # create new vectorstore client for the specific llm string
        self._cache_dict[index_name] = OpenSearchVectorSearch(
            opensearch_url=self.opensearch_url,
            connection_class=RequestsHttpConnection,
            http_auth=aoss_auth,
            index_name=index_name,
            embedding_function=self.embedding,
        )

        # create index for the vectorstore
        vectorstore = self._cache_dict[index_name]
        try:
            if not vectorstore.index_exists():
                _embedding = self.embedding.embed_query(text="test")
                vectorstore.create_index(len(_embedding), index_name)
                print(f"Created new index: {index_name}")
        except NotFoundError:
            # 인덱스가 없을 경우 생성
            _embedding = self.embedding.embed_query(text="test")
            vectorstore.create_index(len(_embedding), index_name)
            print(f"Created new index: {index_name}")
        except Exception as e:
            print(f"Error checking/creating index: {e}")
        return vectorstore

    def lookup(self, prompt: str, llm_string: str) -> Optional[List[Generation]]:
        """Look up based on prompt and llm_string."""
        llm_cache = self._get_llm_cache(llm_string)
        generations: List[Generation] = []
        try:
            # Read from a Hash
            results = llm_cache.similarity_search(
                query=prompt,
                k=1,
                score_threshold=self.score_threshold,
            )
            if results:
                for document in results:
                    try:
                        generations.extend(loads(document.metadata["return_val"]))
                    except Exception:
                        logger.warning(
                            "Retrieving a cache value that could not be deserialized "
                            "properly. This is likely due to the cache being in an "
                            "older format. Please recreate your cache to avoid this "
                            "error."
                        )

                        generations.extend(
                            _load_generations_from_json(document.metadata["return_val"])
                        )
        except NotFoundError:
            logger.info(f"Index for {llm_string} not found or empty. Cache miss.")
        except Exception as e:
            logger.error(f"Error during cache lookup: {e}")

        return generations if generations else None

    def update(
        self, prompt: str, llm_string: str, return_val: List[Generation]
    ) -> None:
        """Update cache based on prompt and llm_string."""
        for gen in return_val:
            if not isinstance(gen, Generation):
                raise ValueError(
                    "OpenSearchSemanticCache only supports caching of "
                    f"normal LLM generations, got {type(gen)}"
                )
        llm_cache = self._get_llm_cache(llm_string)
        metadata = {
            "llm_string": llm_string,
            "prompt": prompt,
            "return_val": dumps([g.dict() for g in return_val]),
        }
        llm_cache.add_texts(texts=[prompt], metadatas=[metadata])

    def clear(self, **kwargs: Any) -> None:
        """Clear semantic cache for a given llm_string."""
        index_name = self._index_name(kwargs["llm_string"])
        if index_name in self._cache_dict:
            self._cache_dict[index_name].delete_index(index_name=index_name)
            del self._cache_dict[index_name]


# OpenSearchSemanticCache 초기화
bedrock_embeddings = BedrockEmbeddings(
    client=bedrock_runtime, model_id="cohere.embed-multilingual-v3"
)
semantic_cache = OpenSearchSemanticCache(
    opensearch_url=opensearch_url, embedding=bedrock_embeddings, score_threshold=0.95
)

# Bedrock Claude 3 Sonnet 모델 초기화 및 캐시 설정
llm = ChatBedrock(
    client=bedrock_runtime,
    model_id="anthropic.claude-3-haiku-20240307-v1:0",
    model_kwargs={"temperature": 0},
    cache=semantic_cache,
)

# 모델 사용 예시
messages = [
    # (
    #     "system",
    #     "You are a helpful assistant that explains concepts in Korean. Explain the given topic.",
    # ),
    ("human", "고양이에 대해 자세히 설명해줘"),
]

for i in range(2):
    start_time = time.time()
    ai_msg = llm.invoke(messages)
    end_time = time.time()

    print(f"Iteration {i+1}:")
    print(ai_msg.content)
    print(f"Usage: {ai_msg.response_metadata['usage']}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("---")
