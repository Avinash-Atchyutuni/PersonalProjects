from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

class SubQuery(BaseModel):
    query: str = Field(description="A specific sub-query")

class QueryDecomposition(BaseModel):
    sub_queries: List[SubQuery] = Field(description="List of sub-queries")

class ParaphrasedQuery(BaseModel):
    query: str = Field(description="A paraphrased version of the query")

class QueryExpansion(BaseModel):
    queries: List[ParaphrasedQuery] = Field(description="List of paraphrased queries")

class queryAnalysis:
    def __init__(self):
        self.llm = Ollama(model="llama3")
        self.parser = JsonOutputParser(pydantic_object=QueryDecomposition)
        self.qe_parser = JsonOutputParser(pydantic_object=QueryExpansion)

    def query_decomposition(self, query: str):
        system = """You are an expert at converting user questions into database queries.
        You have access to a database of tutorial videos about a software library for building LLM-powered applications.
        Perform query decomposition. Given a user question, break it down into distinct sub questions that
        you need to answer in order to answer the original question.
        If there are acronyms or words you are not familiar with, do not try to rephrase them.
        
        {format_instructions}
        
        Provide at least 3 sub-queries for the given question."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{query}"),
        ])
        
        query_analyzer = prompt | self.llm | self.parser
        answer = query_analyzer.invoke({
            "query": query, 
            "format_instructions": self.parser.get_format_instructions()
        })
        return answer

    def query_expansion(self, query: str):
        system = """You are an expert at converting user questions into queries.
        You have access to a database of tutorial videos about a software library for building LLM-powered applications.
        Perform query expansion. If there are multiple common ways of phrasing a user question
        or common synonyms for key words in the question, make sure to return multiple versions
        of the query with the different phrasings.
        If there are acronyms or words you are not familiar with, do not try to rephrase them.
        Return at most 3 versions of the question.
        for example: "Compare the share of Tesla and Disney".
        sub-queries: "what are the shares of Tesla?", "what are the shares of Disney?", "compare the shares of Tesla and Disney".

        {format_instructions}

        Your response should be a JSON object that can be parsed into the QueryExpansion model."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system),
            ("human", "{query}"),
        ])
        query_analyzer = prompt | self.llm | self.parser
        answer = query_analyzer.invoke({
            "query": query,
            "format_instructions": self.parser.get_format_instructions()
        })
        return answer


if __name__ == "__main__":
    query = queryAnalysis()
    result = query.query_decomposition("Compare share prices of Tesla and Disney")
    print(result)