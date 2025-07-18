# 🤖 AI 서비스 개발자 실무형 스터디 커리큘럼 (6개월, 24주)

AI 트렌드에 맞춘 실무형 커리큘럼입니다.  
백엔드 개발자를 위한 AI/LLM 실전 중심 스터디입니다.

---

## ✅ 목표
- 머신러닝/딥러닝/LLM의 핵심 개념을 익히고  
- 실제 모델을 활용한 서비스 개발까지 직접 경험

---

## 🧭 전체 구성 개요

| 기간 | 주차 | 주제 | 목표 |
|-------|------|-------|-------|
| 1-4주 | 1~4주 | 수학 기초 완전 정복 | AI/딥러닝 기초 수학 완벽 이해 |
| 5-8주 | 5~8주 | Python & 데이터 처리 실무 | Python 문법 + 데이터분석 라이브러리 숙달 |
| 9-12주 | 9~12주 | 머신러닝 기본과 모델링 실습 | 머신러닝 알고리즘 이해 및 실습 |
| 13-16주 | 13~16주 | 딥러닝 심화 및 프레임워크 실습 | 신경망 원리 + TensorFlow/PyTorch 심화 |
| 17-20주 | 17~20주 | NLP & LLM 구조 이해와 실무 활용 | Transformer/LLM 개념 완전 정복 및 API 실습 |
| 21-24주 | 21~24주 | LLM 미세조정 & AI 서비스 구현 및 배포 | 파인튜닝, 프롬프트, 서비스 기획과 배포까지 |

---

## 📌 주차별 세부 계획

### 1~4주차 : 수학 기초 완전 정복

| 주차 | 주제 | 세부 내용 | 실습 | 추천 자료 |
|------|------|-----------|------|-----------|
| 1주차 | 벡터와 행렬 기초 | 벡터, 행렬 개념과 연산, 전치행렬, 단위행렬 | Python으로 벡터/행렬 연산 구현 | [모두의연구소 블로그](https://blog.naver.com/PostView.naver?blogId=moids&logNo=222345429144) |
| 2주차 | 행렬곱과 응용 | 행렬곱 계산 원리, 특징, 행렬 곱을 활용한 데이터 변환 | 행렬곱 직접 구현 후 결과 검증 | [StatQuest YouTube](https://www.youtube.com/user/joshstarmer) |
| 3주차 | 미분과 편미분 | 함수 미분, 편미분, 체인 룰 개념과 경사하강법 이해 | Python으로 미분, 경사하강법 구현 | [StatQuest YouTube](https://www.youtube.com/user/joshstarmer), [나도코딩 통계 기초](https://www.youtube.com/watch?v=7cPhj0FkhHU) |
| 4주차 | 확률 및 통계 기초 | 정규분포, 평균, 분산, 소프트맥스 함수 이해 | Softmax 함수 구현, 간단한 확률 문제 | [나도코딩 통계 기초](https://www.youtube.com/watch?v=7cPhj0FkhHU) |

---

### 5~8주차 : Python & 데이터 처리 실무

| 주차 | 주제 | 세부 내용 | 실습 | 추천 자료 |
|------|------|-----------|------|-----------|
| 5주차 | Python 기초 문법 | 리스트, 딕셔너리, 함수, 클래스, 예외처리 | 문제풀이, 기본 문법 숙달 | [점프 투 파이썬](https://wikidocs.net/book/1) |
| 6주차 | 함수형 프로그래밍 | 람다, map/filter, 제너레이터, 데코레이터 | 간단한 함수형 프로그램 작성 | [점프 투 파이썬](https://wikidocs.net/book/1) |
| 7주차 | Numpy 기초 | 배열, 브로드캐스팅, 벡터 연산 | Numpy로 데이터 연산 실습 | [Numpy 10분 완성](https://wikidocs.net/146935) |
| 8주차 | Pandas & 데이터 전처리 | Series, DataFrame, 데이터 필터링/집계 | 타이타닉 데이터셋 전처리 | [Pandas 공식 튜토리얼](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html) |

---

### 9~12주차 : 머신러닝 기본과 모델링 실습

| 주차 | 주제 | 세부 내용 | 실습 | 추천 자료 |
|------|------|-----------|------|-----------|
| 9주차 | 머신러닝 개념 | 지도/비지도 학습, 회귀/분류, 데이터셋 구성 | 사이킷런 기본 사용법 학습 | [FastCampus 머신러닝 무료 강의](https://www.youtube.com/watch?v=ESg9g6_5U1M) |
| 10주차 | 모델 평가 및 과적합 | 교차검증, 평가 지표(정확도, F1, RMSE 등) | 타이타닉 생존자 예측 모델 구현 | [Kaggle Titanic](https://www.kaggle.com/c/titanic) |
| 11주차 | 주요 알고리즘1 | 선형회귀, 로지스틱 회귀 | Python으로 알고리즘 직접 구현 및 sklearn 비교 | [사이킷런 튜토리얼](https://scikit-learn.org/stable/tutorial/index.html) |
| 12주차 | 주요 알고리즘2 | 의사결정나무, 랜덤포레스트, SVM | sklearn 모델 학습 및 튜닝 실습 | [사이킷런 튜토리얼](https://scikit-learn.org/stable/tutorial/index.html) |

---

### 13~16주차 : 딥러닝 심화 및 프레임워크 실습

| 주차 | 주제 | 세부 내용 | 실습 | 추천 자료 |
|------|------|-----------|------|-----------|
| 13주차 | 신경망 기본 | 퍼셉트론, 다층 신경망, 활성화 함수 | Python으로 기본 신경망 구현 | [모두의 딥러닝](https://book.naver.com/bookdb/book_detail.nhn?bid=11982967) |
| 14주차 | 손실 함수와 역전파 | 손실 함수 종류, 역전파 알고리즘 원리 | 역전파 수식 직접 계산 | [모두의 딥러닝](https://book.naver.com/bookdb/book_detail.nhn?bid=11982967) |
| 15주차 | 프레임워크 입문 | TensorFlow or PyTorch 설치 및 기본 사용법 | MNIST 숫자 분류 기본 코드 작성 | [PyTorch 기초 강의](https://www.youtube.com/watch?v=7dum3nCzX1A) |
| 16주차 | 심화 실습 | CNN 기본 구조 이해 및 구현 | Fashion MNIST 분류기 완성 | [Colab 실습 노트북](https://colab.research.google.com/notebooks/intro.ipynb) |

---

### 17~20주차 : NLP & LLM 구조 이해와 실무 활용

| 주차 | 주제 | 세부 내용 | 실습 | 추천 자료 |
|------|------|-----------|------|-----------|
| 17주차 | NLP 기초 | 토큰화, 임베딩, RNN, LSTM 개념 | 텍스트 전처리 코드 작성 | [한입에 쏙 NLP](https://wikidocs.net/book/2155) |
| 18주차 | Transformer 구조 | Self-Attention, Multi-head Attention, Positional Encoding | 작은 Transformer 모듈 직접 구현 | [Jay Alammar 시각자료](https://jalammar.github.io/illustrated-transformer/) |
| 19주차 | HuggingFace 활용 | 사전학습 모델 불러오기, 텍스트 생성/분류 | HuggingFace API로 간단 챗봇 만들기 | [HuggingFace 튜토리얼](https://huggingface.co/course) |
| 20주차 | 프롬프트 튜닝 | 프롬프트 작성법, 요약/분석 API 만들기 | ChatGPT API 활용 실습 | [OpenAI Prompt Engineering Guide](https://github.com/openai/openai-cookbook) |

---

### 21~24주차 : LLM 미세조정 & AI 서비스 구현 및 배포

| 주차 | 주제 | 세부 내용 | 실습 | 추천 자료 |
|------|------|-----------|------|-----------|
| 21주차 | 미세조정 개념 | LoRA, QLoRA, PEFT 개념과 활용 | 간단한 파인튜닝 코드 실습 | [HuggingFace PEFT 문서](https://huggingface.co/docs/peft) |
| 22주차 | 프롬프트 엔지니어링 | Few-shot, Chain-of-thought 기법 | 다양한 프롬프트 작성 및 평가 | [OpenAI Prompt Engineering](https://github.com/openai/openai-cookbook) |
| 23주차 | AI 서비스 설계 | SpringBoot + Kotlin + AI API 연동 | 자동완성, 리뷰 요약 서비스 설계 | 내부 문서, 샘플 코드 |
| 24주차 | 배포 및 발표 | Docker, FastAPI/Flask 배포 | AWS Lambda/EC2 연동, 최종 프로젝트 발표 | [Docker 공식 문서](https://docs.docker.com/), [AWS 공식 문서](https://aws.amazon.com/documentation/) |

---

## 🛠 운영 가이드

- **진행 주기**: 주 1회, 2~3시간 스터디 진행  
- **스터디 방식**  
  - 각 주차별 학습 + 실습 과제 제출  
  - 과제 결과 공유 및 코드 리뷰  
  - 실무 연계 프로젝트 단계별 점검  
- **커뮤니케이션 채널**: Slack/Discord/Notion 등으로 자료와 코드 공유  
- **프로젝트**  
  - 17주차부터 프로젝트 기획, 21주차부터 구현 및 배포  
  - 24주차 최종 발표 및 코드 리뷰  

---

궁금한 점이나 개선 요청 언제든 환영합니다!  
함께 AI 서비스 개발자로 성장해요 🚀
