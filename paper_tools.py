import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, ArxivLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from config import PAPER_OUTPUT_DIR, PROJECT_OUTPUT_DIR, LMSTUDIO_URL

from llm import getllm


class PaperProcessor:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            base_url=LMSTUDIO_URL,
            api_key="lm-studio",
            model="nomic-embed-text",
            check_embedding_ctx_length=False,
        )
        os.makedirs(PAPER_OUTPUT_DIR, exist_ok=True)
    
    def load_paper(self, paper_input: str) -> List[str]:
        """Load paper from PDF path or arXiv ID"""
        if paper_input.endswith('.pdf'):
            loader = PyPDFLoader(paper_input)
        else:  # arxiv ID like "1706.03762"
            loader = ArxivLoader(arxiv_id=paper_input, load_pdf=True)
        
        docs = loader.load()
        return [doc.page_content for doc in docs]
    
    def create_paper_rag(self, paper_path_or_id: str):
        raw_pages = self.load_paper(paper_path_or_id)
        
        texts = []
        for page in raw_pages:
            if isinstance(page, str):
                clean_page = page.strip()
                if clean_page and len(clean_page) > 50:
                    texts.append(clean_page)
        
        print(f"📄 Loaded {len(texts)} clean pages")
        
        #Manual chunking → GUARANTEED strings
        all_text = "\n\n".join(texts[:5])
        if len(all_text) < 100:
            print("Paper too short, using raw text")
            chunks = [all_text]
        else:
            chunks = [all_text[i:i+800] for i in range(0, len(all_text), 700)]
        
        #FINAL string check
        final_chunks = []
        for chunk in chunks:
            if isinstance(chunk, str) and len(chunk.strip()) > 20:
                final_chunks.append(chunk.strip())
        
        print(f"Final chunks: {len(final_chunks)}")
        print(f"First chunk preview: {final_chunks[0][:100]}...")
        
        #Chroma
        vectorstore = Chroma.from_texts(
            texts=final_chunks,
            embedding=self.embeddings,
            persist_directory=f"{PAPER_OUTPUT_DIR}/paper_rag"
        )
        print("RAG ready!")
        return vectorstore

    
    def query_paper(self, vectorstore, question: str):
        """Ask questions about the paper via RAG"""
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        prompt = ChatPromptTemplate.from_template(
            """Based on this research paper context, answer the question.
            
Context: {context}
Question: {question}
Answer:"""
        )
        llm = getllm()
        chain = prompt | llm
        docs = retriever.invoke(question)
        context = "\n\n".join([d.page_content for d in docs])
        result = chain.invoke({
            "context": context,
            "question": question
        })
        return result.content
