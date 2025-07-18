# LLM Ground: 실무형 AI 서비스 개발 스터디 (30주)

> Java/Kotlin 백엔드 개발자를 위한 실전 LLM AI 서비스 학습 로드맵  
> 기간: 30주 (주 1회 진행 기준)  
> 대상: AI 서비스 실무 개발을 목표로 하는 백엔드 개발자  
> 목표: 직접 AI 서비스를 설계, 구축, 배포할 수 있는 역량 확보

---

## 🧭 커리큘럼 개요

- 총 30주, 이론/실습/프로젝트 병행
- 실무 중심으로 Prompt, RAG, Embedding, API, Vector DB 등 전반 학습
- Python 및 LangChain, LlamaIndex 실습 포함
- 기초 수학 포함: 선형대수, 미분, 통계 기초

---

## 📅 주차별 커리큘럼

### 🔰 Phase 1: AI 서비스 개발 기초 (1~5주)

| 주차 | 주제 | 내용 |
|------|------|------|
| 1주차 | OT & LLM 개요 | 스터디 목표 설정, 생성형 AI 개요, 주요 용어 정리 |
| 2주차 | ChatGPT, GPT 구조 | Transformer, Self-Attention 등 핵심 구조 개념 |
| 3주차 | LLM 활용 아키텍처 | LLM API 호출 구조, 비즈니스 접목 사례 분석 |
| 4주차 | Prompt Engineering | 프롬프트 설계 기초, few-shot, chain-of-thought |
| 5주차 | Python 기초 | 실습용 Python 문법 정리 (Lambda, Dict 등) |

---

### 🔍 Phase 2: LLM 활용 및 실전 설계 (6~14주)

| 주차 | 주제 | 내용 |
|------|------|------|
| 6주차 | LangChain 기초 | LangChain 개념, PromptTemplate, Chains 실습 |
| 7주차 | Embedding 기초 | 텍스트 벡터화 이해, cosine similarity 등 |
| 8주차 | Vector DB 기초 | FAISS, Chroma 등 실습, 검색 정확도 비교 |
| 9주차 | Retrieval 기반 RAG 개념 | RAG 구조 이해, Retrieval + Generation 설계 |
| 10주차 | LLM 기반 챗봇 설계 | 유저 질문에 응답하는 챗봇 설계 실습 |
| 11주차 | LangChain Agents | Tool 사용, OpenAPI 연동, AgentAction 이해 |
| 12주차 | 문서 기반 RAG 서비스 | PDF/Notion 등 문서 기반 질의응답 서비스 설계 |
| 13주차 | LlamaIndex 기본 | LlamaIndex vs LangChain 비교 및 실습 |
| 14주차 | Agent vs RAG | RAG + Agent 통합 아키텍처 설계 패턴 소개 |

---

### ⚙️ Phase 3: 실무형 서비스 개발 및 운영 (15~24주)

| 주차 | 주제 | 내용 |
|------|------|------|
| 15주차 | FastAPI 기초 | AI 서비스용 Python API 서버 구축 실습 |
| 16주차 | OpenAI API 연동 | GPT API 연동, 속도 및 사용량 측정 |
| 17주차 | 사용자 Context 관리 | Session 기반 대화 흐름 저장 구조 설계 |
| 18주차 | RAG 성능 개선 전략 | HyDE, Reranking, Multi-query 기법 |
| 19주차 | 메타데이터 필터링 | Query filtering, tag 기반 검색 정교화 |
| 20주차 | 벡터 인덱스 최적화 | Index type 비교 (IVF, HNSW), latency 측정 |
| 21주차 | Redis Vector Store 연동 | Redis w/ vector search 연동 실습 |
| 22주차 | LangSmith 사용법 | LangChain 디버깅 및 Trace 관리 |
| 23주차 | Open Source LLM (Mistral 등) | Llama3, Mistral, Ollama 환경 구축 |
| 24주차 | 모델 경량화 전략 | Quantization, GGUF, LoRA 개념 실습 |

---

### 🚀 Phase 4: 실전 프로젝트 (25~30주)

| 주차 | 주제 | 내용 |
|------|------|------|
| 25주차 | 프로젝트 설계 | 주제 선정, 요구사항 정의, 아키텍처 초안 |
| 26주차 | 프롬프트/검색 설계 | LLM Prompt & Retrieval 구조 확정 |
| 27주차 | 벡터화 & Index 구축 | Embedding, Vector DB 구축 |
| 28주차 | API 구축 & 통합 | API, 프론트 연동, 오류 대응 설계 |
| 29주차 | 테스트 및 고도화 | Evaluation, 개선 반복 |
| 30주차 | 발표 및 회고 | 프로젝트 발표 및 회고 공유 |

---

## 🛠️ 실습 도구 및 환경

- Python 3.10+
- LangChain / LlamaIndex
- ChromaDB / FAISS / Redis Vector
- OpenAI API / Ollama / Huggingface Hub
- FastAPI / Docker

---

## 📚 추천 도서 & 자료

### 기본 이론
- [Deep Learning with Python (François Chollet)](https://www.oreilly.com/library/view/deep-learning-with/9781617294433/)
- [LLM University by Cohere](https://university.cohere.com/)
- [Dive into Deep Learning](https://d2l.ai/index.html)

### 실습 & 프레임워크
- [LangChain 공식 문서](https://docs.langchain.com/)
- [LlamaIndex 공식 문서](https://docs.llamaindex.ai/)
- [FastAPI 공식 문서](https://fastapi.tiangolo.com/)

### RAG & Vector DB
- [Retrieval-Augmented Generation paper (Lewis et al.)](https://arxiv.org/abs/2005.11401)
- [Pinecone RAG Guide](https://www.pinecone.io/learn/rag/)
- [Weaviate RAG 튜토리얼](https://weaviate.io/blog/rag-langchain)

---

## 🧑‍💻 운영 가이드

- **진행방식**: 매주 1회 발표 + 실습 공유
- **공통과제**: 매주 실습 과제 PR 또는 노션에 제출
- **피드백**: 코드 리뷰 및 PR 기반 피드백
- **문서화**: 매주 회고 작성 및 정리

---

## 📩 문의

스터디/운영/내용 관련하여 궁금한 점은 이슈 또는 PR로 남겨주세요!
