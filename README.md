# OpenSearch Semantic Cache

이 프로젝트는 LangChain과 OpenSearch를 사용하여 AI 모델의 응답을 캐싱하는 예제를 제공합니다.

## 설치

1. 이 저장소를 클론합니다:
   ```
   git clone https://github.com/yourusername/opensearch-langchain-cache.git
   cd opensearch-langchain-cache
   ```

2. 가상 환경을 생성하고 활성화합니다:
   ```
   python -m venv venv
   source venv/bin/activate  # Windows의 경우: venv\Scripts\activate
   ```

3. 필요한 패키지를 설치합니다:
   ```
   pip install -r requirements.txt
   ```

4. `.env` 파일을 생성하고 필요한 환경 변수를 설정합니다.

## 사용 방법

1. `.env` 파일에 필요한 환경 변수를 설정합니다.
2. `src/main.py` 파일을 실행합니다:
   ```
   python src/main.py
   ```

## 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.