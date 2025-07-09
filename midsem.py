#Jai Mata Di
# Load environment variables from .env file
from dotenv import load_dotenv
import os
import json
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"

file1 = "computer_network.txt"
file2 = "data_structures.txt"
file3 = "databases.txt"
file4 = "machine_learning.txt"
file5 = "operating_systems.txt"

# Get API key from environment variable
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file or environment.")

# Initialize the OpenAI model
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=.8
)

def create_rag_system(file_path):
    """Initialize the RAG system components"""
    
    # Initialize the LLM and embeddings model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    
    # Load and split the document
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # Split documents into chunks
    # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Create and populate the vector store
    vector_store = Chroma(embedding_function=embeddings)
    vector_store.add_documents(documents=splits)
    
    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    return retriever

def list_gen(filepath):
    with open("final_evaluations.txt", "r") as f:
        str=f.read()
    prompt = [
        SystemMessage(content="""Generate a list of 10 topics for the prompt submitted by the user
        Give the solution in a structured format in json:
        {
            "list":[List of topics]
        }
        """),
        HumanMessage(content=str)
    ]
    response = llm.invoke(prompt).content
    json_res=json.loads(response)
    return json_res["list"]

def secondQueryGen(conv_History, l1,l2,l3,l4,l5):
    prompt1 = [
        SystemMessage(content=conv_History),
        HumanMessage(content="This is the previous conversation, based on this Generate an optimized Search Query")
    ]
    llm_res1=llm.invoke(prompt1).content
    # Based on this search query we will find the relevant database
    prompt2 = [
        SystemMessage(content="""Based on the user query scan the following lists of topic and tell me the vector data base that i should select for the query solution
                      list1 ({l1}): Reply "computer_network"
                      list1 ({l2}): Reply "data_structures"
                      list1 ({l3}): Reply "databases"
                      list1 ({l4}): Reply "machine_learning"
                      list1 ({l5}): Reply "operating_systems"
                      If none of the lists match the search query Reply "NONE"
        """),
        HumanMessage(content=llm_res1)
    ]
    llm_res2=llm.invoke(prompt2).content
    if(llm_res2=="NONE"):
        print("LLM does not have any documents related to user query")
        return "NONE"
    else:
        return llm_res1, llm_res2
    
def retrieveChunks(retriver, question):
    """Answer a question using the RAG system"""
    
    # Retrieve relevant documents
    retrieved_docs = retriver.get_relevant_documents(question)
    
    # Combine retrieved documents into context
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context

def answer_question(retriever, question):
    """Answer a question using the RAG system"""
    
    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(question)
    
    # Combine retrieved documents into context
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # Get response from LLM
    answer = get_response_from_llm(llm, context, question)

    prompt1 = [
        SystemMessage(content=f"""Break {answer} into a list of claims/facts
        Respond in the following JSON format:
        {{
            "list": "list of facts"
        }}
        Ensure the response is valid JSON.
        """),
        HumanMessage()
    ]
    res1=json.loads(llm.invoke(prompt1).content)
    for claim in res1["list"]:
        prompt = [
            SystemMessage(content=f"""Based on the claim {claim} check if it is supported by the retrived chunks {retrieveChunks(retriver=retriever,question=claim )}
            Respond "YES": if claim supported or "NO": if claim not supported
            """),
            HumanMessage()
        ]
        resp=llm.invoke(prompt).content
        if(resp=="NO"):
            return "try again", context

    
    return answer, context

def answer_question_removeUnSupported(retriever, question):
    """Answer a question using the RAG system"""
    
    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(question)
    
    # Combine retrieved documents into context
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # Get response from LLM
    answer = get_response_from_llm(llm, context, question)

    prompt1 = [
        SystemMessage(content=f"""Break {answer} into a list of claims/facts
        Respond in the following JSON format:
        {{
            "list": "list of facts"
        }}
        Ensure the response is valid JSON.
        """),
        HumanMessage()
    ]
    res1=json.loads(llm.invoke(prompt1).content)
    i=0
    list=res1["list"]
    for claim in list:
        prompt = [
            SystemMessage(content=f"""Based on the claim {claim} check if it is supported by the retrived chunks {retrieveChunks(retriver=retriever,question=claim )}
            Respond "YES": if claim supported or "NO": if claim not supported
            """),
            HumanMessage()
        ]
        resp=llm.invoke(prompt).content
        if(resp=="NO"):
            list.__delitem__(i)
        i+=1
    chunks=retrieveChunks(retriver=retriever,question=list )
    prompt2 = [
        SystemMessage(content=f"""Based on the list of claims {list} generate appropriate answer again for the question: {question} 
                      and context: {chunks}
        """),
        HumanMessage()
    ]
    answer= llm.invoke(prompt2).content
    return answer, chunks


def main():
    # Initialize the RAG system
    computer_network = create_rag_system(file1)
    data_structures = create_rag_system(file2)
    databases = create_rag_system(file3)
    machine_learning = create_rag_system(file4)
    operating_systems = create_rag_system(file5)
    list_compNet=list_gen(file1)
    list_datStr=list_gen(file2)
    list_databases=list_gen(file3)
    list_machLea=list_gen(file4)
    list_opSys=list_gen(file5)
    
    # Conversation history retturned by appropriate fn
    conv_History=""
    exit="NO"
    while exit!="YES":
        in1=input("Start talking to LLM")
        prompt = [
            SystemMessage(content=f"""You are an LLM agent designed to keep track of the conversation history and talk to the user based on the inputs provided by him/her
                        Perform this task in a good way
                        Previous Conversation History: {conv_History}
            """),
            HumanMessage(in1)
        ]
        response=llm.invoke(prompt).content
        conv_History+="HUMAN RESPONSE: "+in1+"/nLLM Response: "+response
        searchQuery, docname = secondQueryGen(conv_History, list_compNet, list_datStr, list_databases, list_machLea, list_opSys)
        while(docname=="NONE"):
            searchQuery, docname = secondQueryGen(conv_History, list_compNet, list_datStr, list_databases, list_machLea, list_opSys)
            pass
        
        i=0
        while(i<3):
            resp, context = answer_question(eval(docname), searchQuery)
            if resp == "try again":
                i = i + 1
            else:
                break
        if i == 3:
            resp, context = answer_question_removeUnSupported(eval(docname), searchQuery)
        print("LLM:", resp)
        conv_History += "\n Response to LLM generated query: " + str(resp)
        exit = input("want to continue, reply yes or no").strip().upper()

