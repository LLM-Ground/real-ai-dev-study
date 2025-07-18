# LLMGROUND: 실무형 LLM 기반 AI 서비스 개발 스터디 (30주)

> **스터디 목적**  
> - AI 백엔드 개발자로 전환하거나 역량을 확장하려는 백엔드 개발자를 위한 실무형 커리큘럼입니다.  
> - Python, 수학, LLM 기초부터 RAG, 벡터 DB, Agent, LangChain 실습, 서비스 설계 및 배포까지 다룹니다.  

---

## 📌 운영 가이드

- **진행 방식**: 1주 1회 스터디 진행 + 개인 과제 + 실습 코드 공유  
- **언어/도구**: Python, Jupyter Notebook, LangChain, LlamaIndex, OpenAI API, Pinecone, Weaviate 등  
- **공식 언어**: Python  
- **권장 인원**: 3~6명  
- **공유 장소**: GitHub, Discord, Notion  
- **형식**: `이론 학습 → 실습 → 발표 → 리뷰`  

---

## 🧠 커리큘럼 (30주)

---

### ✅ 1~4주차: 기초 수학 & Python

#### 1주차 - 파이썬 기초 (개발자용)
- 자료형, 함수, 클래스, 예외 처리
- [Python for Devs](https://www.w3schools.com/python/)
- 추천: [점프 투 파이썬](https://wikidocs.net/book/1)

#### 2주차 - 파이썬 데이터 처리 & 파일 IO
- List, Dict, 파일 읽기/쓰기, JSON 파싱
- [Built-in Functions](https://docs.python.org/3/library/functions.html)

#### 3주차 - 기초 수학: 선형대수, 벡터, 행렬
- 벡터 연산, 내적, 행렬곱
- [Essence of Linear Algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs)

#### 4주차 - 기초 수학: 미분, 확률 통계 개념
- 함수 미분, 확률 분포, 평균/분산/표준편차
- [Khan Academy - Statistics](https://www.khanacademy.org/math/statistics-probability)

---

### ✅ 5~8주차: LLM과 생성형 AI 이해

#### 5주차 - LLM의 구조와 Transformer 이해
- Attention, Decoder-only 구조, BERT vs GPT
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

#### 6주차 - Hugging Face와 LLM 모델 다루기
- Transformers 라이브러리, 모델 로딩, 토크나이저 실습
- [Hugging Face Course](https://huggingface.co/learn/nlp-course)

#### 7주차 - OpenAI API와 기본 텍스트 생성
- `gpt-3.5-turbo`, Prompt 구성, completion 실습
- [OpenAI Docs](https://platform.openai.com/docs)

#### 8주차 - Prompt Engineering
- Zero-shot, Few-shot, Chain-of-Thought, Role Prompting
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

---

### ✅ 9~12주차: RAG (Retrieval-Augmented Generation)

#### 9주차 - RAG 개념과 구조
- RAG 구성요소: Retriever + Generator
- [RAG 소개 논문](https://arxiv.org/abs/2005.11401)

#### 10주차 - LangChain 구조 이해 및 실습
- LangChain core, prompt chain, retriever chain 실습
- [LangChain Examples](https://github.com/langchain-ai/langchain-examples)

#### 11주차 - 벡터 DB 이해 및 Pinecone, FAISS 실습
- Embedding, 벡터화, cosine similarity
- [FAISS](https://github.com/facebookresearch/faiss), [Pinecone](https://docs.pinecone.io/)

#### 12주차 - LlamaIndex 실습 + LangChain 비교
- Document loader, query engine
- [LlamaIndex Docs](https://docs.llamaindex.ai/)

---

### ✅ 13~16주차: 고급 사용법 및 Agent

#### 13주차 - LangChain Memory & Conversation
- BufferMemory, TokenBufferMemory, ConversationChain
- [LangChain Memory Docs](https://docs.langchain.com/docs/components/memory)

#### 14주차 - LangChain Tools & Agents
- ReAct, Zero-shot Agent, Tool 사용법
- [LangChain Agent Docs](https://python.langchain.com/docs/modules/agents/)

#### 15주차 - Self-Querying Retrieval & HyDE
- Self-querying retriever, Hypothetical Document Embeddings
- [LangChain - Self Query](https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/)

#### 16주차 - Vector Store 최적화 및 Hybrid Search
- Dense + Sparse Hybrid, BM25 + Embedding
- [Weaviate Hybrid Search](https://weaviate.io/developers/weaviate/search/hybrid)

---

### ✅ 17~20주차: 실무 연계 및 서비스 설계

#### 17주차 - Multi-turn 대화 시스템 설계
- Dialogue flow, 대화 컨텍스트 유지
- 실습: FAQ 봇 만들기

#### 18주차 - 외부 API와 LangChain 연동
- Weather API, Search API 사용해 Agent 확장
- 실습: 검색 + 답변 봇

#### 19주차 - LangSmith & Observability
- LangSmith를 통한 트레이스 분석
- [LangSmith](https://smith.langchain.com/)

#### 20주차 - LangServe로 서비스 배포
- LangChain을 FastAPI 서버로 배포
- [LangServe](https://docs.langchain.com/docs/guides/deploy/langserve)

---

### ✅ 21~25주차: 응용 및 실전 프로젝트 준비

#### 21주차 - 문서 요약 & QA 실습
- 문서 분할, 요약, RAG 기반 질문 응답
- 실습: PDF에서 질문받고 답변하기

#### 22주차 - 코드 보조 봇 만들기
- GPT로 코드 리팩토링 / 설명 봇 만들기
- 실습: 코드 리뷰 Agent

#### 23주차 - JSON/구조화 응답 처리
- function calling, schema 기반 응답 설계
- [Function Calling Docs](https://platform.openai.com/docs/guides/function-calling)

#### 24주차 - 이미지/음성 등 멀티모달 연동
- Whisper, DALL·E, Gemini Pro Vision 등 API 실습
- [OpenAI Whisper](https://github.com/openai/whisper)

#### 25주차 - Vector DB & RAG 성능 벤치마크
- Latency 측정, top-k tuning, embedding 성능 비교
- 실습: FAISS vs Pinecone vs Weaviate

---

### ✅ 26~30주차: 실전 프로젝트 진행

#### 26~28주차 - 실전 미니 프로젝트 진행
예시:
- 쇼핑몰용 LLM FAQ 챗봇
- 개인 검색 보조 Agent
- 팀 내 문서 요약 봇
- 코드 리뷰 어시스턴트

#### 29주차 - 배포 및 성능 최적화
- FastAPI + Docker + Cloud 환경 배포
- Latency 개선, 비용 절감 팁 정리

#### 30주차 - 발표 및 회고
- 결과 발표, 피드백 공유, 기술블로그 작성
- 기술회고 작성 & 포트폴리오 등록

---

## 📚 추천 도서 & 자료

- [The Hundred-Page Machine Learning Book](https://themlbook.com/)
- [Deep Learning with Python - François Chollet](https://www.manning.com/books/deep-learning-with-python)
- [Transformers for NLP - Hugging Face](https://huggingface.co/transformers/)
- [GPTs are GPTs 논문](https://arxiv.org/abs/2303.12712)
- [Full Stack LLM Bootcamp - Pinecone](https://www.pinecone.io/learn/series/llm-bootcamp/)


---
