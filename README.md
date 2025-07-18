# 💡 LLM 실무형 AI 서비스 개발 스터디 (30주)

## 🎯 목표
- Java 개발자도 실무에 적용 가능한 LLM 서비스 개발 역량 확보
- Python 기초 → AI 기초 수학 → 머신러닝/딥러닝 → RAG 기반 서비스 개발까지 실습 중심 학습
- 검색 + 챗봇 + 생성형 AI 융합 프로젝트 구현

---

## 📅 커리큘럼 개요

| 기간 | 주제 | 내용 | 추천 리소스 |
|------|------|------|-------------|
| 1주차 | Python 기초 I | 문법, 자료형, 조건문, 반복문 | [파이썬 300제](https://wikidocs.net/book/922), [점프 투 파이썬](https://wikidocs.net/book/1) |
| 2주차 | Python 기초 II | 함수, 클래스, 예외, 패키지 | [Python 공식 튜토리얼](https://docs.python.org/ko/3/tutorial/index.html) |
| 3주차 | 개발 환경 세팅 | VSCode, Jupyter, Conda, Git | [Jupyter 설치](https://jupyter.org/install), [Python 가상환경](https://wikidocs.net/14155) |
| 4주차 | 기초 수학 I | 선형대수 (벡터, 행렬), numpy | [Youtube - 선형대수 쉽게 배우기](https://www.youtube.com/playlist?list=PLkZYeFmDuaNn2iVoHqVdQre8nwuCksLZx), [numpy 튜토리얼](https://numpy.org/learn/) |
| 5주차 | 기초 수학 II | 미분, 확률, 통계 | [기초통계 강의](https://www.kmooc.kr/courses/course-v1:SMU+SMU_Stats101+2020_T1/about) |
| 6주차 | Pandas & 시각화 | Pandas, Matplotlib, Seaborn | [Pandas 공식](https://pandas.pydata.org/docs/), [Seaborn 튜토리얼](https://seaborn.pydata.org/tutorial.html) |
| 7주차 | LLM 개념 입문 | GPT, Transformer, Tokenization | [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/), [ChatGPT 논문 정리](https://youtu.be/jwoy5B3mW8Y) |
| 8주차 | OpenAI API 실습 | Completion / ChatCompletion / Embedding 사용법 | [OpenAI Docs](https://platform.openai.com/docs/), [Langchain + OpenAI 실습](https://github.com/pinecone-io/examples) |
| 9주차 | Prompt Engineering I | 기본 프롬프트 설계 원칙 | [Prompt Engineering Guide](https://www.promptingguide.ai/) |
| 10주차 | Prompt Engineering II | Few-shot, Chain-of-Thought, System Prompt | [Langchain PromptTemplate Docs](https://docs.langchain.com/docs/components/prompts/prompt-templates) |
| 11주차 | Vector DB 개념 & 실습 | 벡터 유사도 검색, FAISS, ChromaDB | [FAISS 튜토리얼](https://github.com/facebookresearch/faiss/wiki/Getting-started), [Chroma](https://docs.trychroma.com/) |

---

## 🧠 머신러닝 & 딥러닝

| 주차 | 주제 | 내용 | 추천 리소스 |
|------|------|------|-------------|
| 12주차 | 머신러닝 개념 | 지도학습/비지도학습, 과적합/일반화 | [모두를 위한 머신러닝](https://www.youtube.com/playlist?list=PLlMkM4tgfjnLSOjrEJN31gZATbcj_MpUm) |
| 13주차 | Scikit-Learn 실습 | 선형회귀, 분류, KNN, SVM | [Scikit-learn 튜토리얼](https://scikit-learn.org/stable/tutorial/index.html) |
| 14주차 | 머신러닝 모델 평가 | Confusion matrix, ROC, Precision/Recall | [Machine Learning Crash Course (Google)](https://developers.google.com/machine-learning/crash-course) |
| 15주차 | 딥러닝 개념 | Perceptron, MLP, CNN 개념 | [모두를 위한 딥러닝](https://www.youtube.com/playlist?list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG) |
| 16주차 | PyTorch 실습 I | Tensor, Dataset, Dataloader, 모델 정의 | [PyTorch 튜토리얼](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) |
| 17주차 | PyTorch 실습 II | CNN 실습 (MNIST), 과제: 간단한 이미지 분류 | [PyTorch 예제](https://github.com/pytorch/examples) |

---

## 🛠️ 실전 LLM 서비스 구현

| 주차 | 주제 | 내용 | 추천 리소스 |
|------|------|------|-------------|
| 18주차 | LangChain 개요 | 체인, 프롬프트, 메모리 구조 이해 | [LangChain CookBook](https://github.com/hwchase17/langchain-cookbook), [Docs](https://docs.langchain.com/) |
| 19주차 | LangChain 실습 I | LLMChain, SequentialChain, RetrievalQA | [LangChain Tutorials](https://python.langchain.com/docs/get_started/introduction) |
| 20주차 | LangChain 실습 II | Tools, Agents, OutputParser | [LangChain Agent Guide](https://python.langchain.com/docs/modules/agents/) |
| 21주차 | 문서 임베딩 & RAG 개념 | Embedding → VectorDB → Retrieval | [RAG 개념 설명](https://www.pinecone.io/learn/retrieval-augmented-generation/) |
| 22주차 | RAG 서비스 실습 | 나만의 문서 기반 QA 챗봇 만들기 | [RAG 실습 예제](https://github.com/langchain-ai/langchain/blob/master/cookbook/RAG_QA.ipynb) |
| 23주차 | Multi-turn QA & Streaming | LangChain Memory, 스트리밍 출력 | [Streamlit + LangChain](https://github.com/streamlit/llm-examples) |
| 24주차 | LangServe / FastAPI 연동 | API 서비스 배포형태로 전환 | [LangServe Docs](https://docs.langchain.com/docs/guides/deploying/langserve) |

---

## 🚀 프로젝트 집중 기간

| 주차 | 주제 | 내용 | 추천 도구 |
|------|------|------|-----------|
| 25주차 | 아이디어 선정 | 검색+AI 챗봇 또는 문서 QnA 등 주제 선정 | Notion, Miro |
| 26주차 | 데이터 구축 | 나만의 PDF/DB/FAQ 등 수집 및 벡터화 | pdfplumber, beautifulsoup4 |
| 27주차 | 백엔드 구성 | LangChain + FastAPI + VectorDB 구축 | LangServe, FAISS, Qdrant |
| 28주차 | 프론트 + UI | Streamlit or React 연동 | Streamlit Docs |
| 29주차 | 테스트 & 배포 | Docker, Cloud (Railway, Render 등) | Docker, GitHub Actions |
| 30주차 | 데모 & 발표 | 스터디원 앞 발표 / GitHub 정리 | GitHub Pages, Notion |

---

## 📚 추천 도서 & 자료

- 《파이썬 라이브러리를 활용한 데이터 분석》 – 웨슬리 맥키니
- 《밑바닥부터 시작하는 딥러닝》 – 사이토 고키
- 《딥러닝 일잘러 노트》 – (추천 실무서, 쉽고 실용적)
- [LangChain 공식 문서](https://docs.langchain.com/)
- [OpenAI 공식 문서](https://platform.openai.com/docs/)
- [Deep Learning Specialization – Andrew Ng (Coursera)](https://www.coursera.org/specializations/deep-learning)

---

## 🧑‍💻 스터디 운영 가이드

- **진행 방식**: 주 1회 모임 (Zoom or 오프라인), 매주 과제 제출 & 발표
- **학습 자료 공유**: Notion, GitHub Repository 활용
- **깃허브 활용법**:
  - `study-ai/llm-ground` (추천 리포지토리 이름)
  - `/week01`, `/week02`, `/project` 구조로 커밋

---

## 📌 스터디명 추천

> **LLM Ground: 실무형 생성형 AI 스터디**

---

🏁 *Java 개발자도, AI를 처음 접해도 끝까지 완주할 수 있도록 구성된 커리큘럼입니다. 꾸준히 진행하면 실무 프로젝트를 구현할 수 있는 수준까지 도달할 수 있어요!*
