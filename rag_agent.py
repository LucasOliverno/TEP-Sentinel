import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
# import google.generativeai as genai # REMOVED: Deprecated & Unused 

load_dotenv()

# Settings
DB_DIR = "chroma_db"

class RAGAgent:
    def __init__(self):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not found.")
            
        # Initialize Embeddings (Standard LangChain Class)
        # Using a modern embedding model
        # Initialize Embeddings (Standard LangChain Class)
        # Using a modern embedding model
        # Available model: models/gemini-embedding-001
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", 
            google_api_key=self.api_key
        )
        
        # Load Vector DB
        self.vectorstore = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
        
        # Define Prompt
        template = """
        You are an Expert Industrial Operator Assistant for the Tennessee Eastman Process (TEP).
        
        Context from Technical Manuals:
        {context}
        
        Current Situation:
        Alert Code: {fault_code}
        Sensor Readings: {sensor_data}
        
        Task:
        1. Identify the technical name of the fault based on the code.
        2. Explain the root cause based on the context.
        3. Recommend specific actions for the operator (Valves to check, Setpoints to adjust).
        
        Format your response in Markdown with clear headers.
        """
        self.prompt = ChatPromptTemplate.from_template(template)
        
        # Build Chain
        self.chain = (
            {"context": self.retriever, "fault_code": RunnablePassthrough(), "sensor_data": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def diagnose(self, fault_code, sensor_data):
        print(f"Analyzing Fault {fault_code} with RAG...")
        # We pass the fault code as the query to retrieve relevant docs
        response = self.chain.invoke(f"Fault {fault_code} description and cause") 
        # Note: The chain as defined above is slightly tricky because 'invoke' input maps to the first RunnablePassthrough
        # Let's fix the chain call to be more explicit or adjust the input.
        
        # Simpler invocation for clarity:
        # Augment query with synonyms for better retrieval
        query = f"{fault_code} Fault Falha description"
        docs = self.retriever.invoke(query)
        context_text = "\n\n".join([d.page_content for d in docs])
        
        chain = self.prompt | self.llm | StrOutputParser()
        return chain.invoke({
            "context": context_text,
            "fault_code": fault_code,
            "sensor_data": sensor_data
        })

if __name__ == "__main__":
    # Test Run
    agent = RAGAgent()
    report = agent.diagnose("IDV(1)", "XMEAS(1) Feed A is Low")
    print("\n--- Diagnostic Report ---\n")
    print(report)
