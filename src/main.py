import hashlib
import logging
import os
import time
from json import loads
from typing import List

import boto3
from dotenv import load_dotenv
from langchain.schema import Generation
from langchain_aws import BedrockEmbeddings
from langchain_aws import ChatBedrock
from langchain_community.cache import OpenSearchSemanticCache
from opensearchpy import RequestsAWSV4SignerAuth

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

# OpenSearchSemanticCache 초기화
bedrock_embeddings = BedrockEmbeddings(
    client=bedrock_runtime, model_id="cohere.embed-multilingual-v3"
)
semantic_cache = OpenSearchSemanticCache(
    opensearch_url=opensearch_url,
    embedding=bedrock_embeddings,
    score_threshold=0.95,
    http_auth=aoss_auth,
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


def _hash(_input: str) -> str:
    return hashlib.md5(_input.encode()).hexdigest()


def _load_generations_from_json(json_string: str) -> List[Generation]:
    json_object = loads(json_string)
    return [Generation(**gen) for gen in json_object]


for i in range(2):
    start_time = time.time()
    ai_msg = llm.invoke(messages)
    end_time = time.time()

    print(f"Iteration {i + 1}:")
    print(ai_msg.content)
    print(f"Usage: {ai_msg.response_metadata['usage']}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("---")
