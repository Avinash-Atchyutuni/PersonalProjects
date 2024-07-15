from PyPDF2 import PdfReader
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
import pandas as pd 
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from Query_Analyzer import queryAnalysis
from langchain.agents import initialize_agent,tool,AgentType
import os
from get_score import score
#from sklearn.metrics.pairwise import cosine_similarity

class RAG:
    def __init__(self,require_reranker=False,require_score=False):
        self.llm = Ollama(model = "llama3")
        self.require_reranker = require_reranker
        self.require_score = require_score
        self.qe = queryAnalysis()

    def preprocess_files(self, path):
        elements = []
        for filename in os.listdir(path):
            if filename.endswith(".pdf"):
                pdf_path = os.path.join(path, filename)
                ele = partition_pdf(pdf_path)
                elements.extend(ele)
        return elements

    def get_chunks(self, elements):
        chunks = chunk_by_title(elements)
        return chunks

    def get_documents(self, chunks):
        documents = []
        for element in chunks:
            metadata = element.metadata.to_dict()
            documents.append(Document(metadata=metadata, page_content=element.text))
        return documents

    def get_vector_store(self, documents):
        db = FAISS.from_documents(documents, OllamaEmbeddings(model="nomic-embed-text",show_progress=True))
        db.save_local("RAG/faiss_index")
        db = FAISS.load_local("RAG/faiss_index", OllamaEmbeddings(model="nomic-embed-text",show_progress=True), allow_dangerous_deserialization=True)
        retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
        return retriever

    def get_rag_chain(self, retriever):
        prompt_template = """
    <|start_header_id|>user<|end_header_id|>
    Responds user questions taking into account the given context, give a precise 
    answer. if you do not know the answer, just say "I do not know the answer". do not use your internal knowledge to answer the question.
    Question: {question}
    Context: {context}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
        prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        def process_and_retrieve(query):
            decomposed_queries = self.qe.query_decomposition(query)
            all_docs = []
            for sub_query in decomposed_queries["sub_queries"]:
                docs = retriever.get_relevant_documents(sub_query["query"])
                all_docs.extend(docs)
            # Remove duplicates and limit the number of documents if needed
            unique_docs = list({doc.page_content: doc for doc in all_docs}.values())
            return unique_docs[:5] 

        rag_chain = (
            {
                "context": RunnableLambda(process_and_retrieve) | format_docs, 
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def get_answer(self,question,pdf_path):
        elements = self.preprocess_files(pdf_path)
        chunks = self.get_chunks(elements)
        documents = self.get_documents(chunks)
        retriever = self.get_vector_store(documents)
        if self.require_reranker:
            print("Reranker is required")
            compressor = FlashrankRerank()
            compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
            rag_chain = self.get_rag_chain(compression_retriever)
        else:
            rag_chain = self.get_rag_chain(retriever)
        answer = rag_chain.invoke(question)
        return answer

def create_answers(questions,rag,path,file_name):
    answers = []
    df = {"Questions": questions}
    for question in questions:
        answer = rag.get_answer(question,path)
        answers.append(answer)
    df["Answers"] = answers
    df = pd.DataFrame(df)
    df.to_csv(file_name,index=False)

if __name__ == "__main__":
    rag = RAG(require_reranker=True)
    rag.get_answer(question="question",pdf_path="path")#enter the question and path to the pdf file
    
    
    
    