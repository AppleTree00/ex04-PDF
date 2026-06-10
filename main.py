# pip install --upgrade langchain langchain-community langchain-text-splitters langchain-openai langchain-chroma pypdf python-dotenv

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

loader = PyPDFLoader("unsu01.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 300,           # 하나의 청크가 가질 최대 글자 수
    chunk_overlap  = 20,        # 청크 간 문맥 연결을 위해 겹칠 글자 수
    length_function = len,      # 길이 측정 기준 (기본 문자열 길이)
    is_separator_regex = False, # 구분 기호의 정규표현식 해석 여부
)
texts = text_splitter.split_documents(pages)


embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


db = Chroma.from_documents(texts, embeddings_model)


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(), 
    llm=llm
)


system_prompt = (
    "너는 질문-답변을 돕는 유능한 비서야. "
    "아래 제공된 맥락(context)만을 사용하여 질문에 답해줘. "
    "답을 모르면 모른다고 하고, 절대 답변을 지어내지 마.\n\n"
    "{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])


question_answer_chain = create_stuff_documents_chain(llm, prompt)

rag_chain = create_retrieval_chain(retriever_from_llm, question_answer_chain)


# question = "아내가 사다 달라고 했던 모든 음식은 무엇이야?"
question = "김첨지는 왜 운수가 좋은날 이라고 생각한거야?"

response = rag_chain.invoke({"input": question})

# 결과 출력
print(f"검색된 참조 문서 개수: {len(response.get('context', []))}")
print(f"답변: {response['answer']}")
