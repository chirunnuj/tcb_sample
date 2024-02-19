import pymongo
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import streamlit as st
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from htmlTemplates1 import css, bot_template, user_template

# from langchain_community.chat_models import ChatOpenAI
# from htmlTemplates import css, bot_template, user_template

mongo_url = ""

def generate_embedding_local(text:str) -> list[float]:
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(text)

    return embeddings.tolist()


def get_conversation_chain(vectorstore):
    # Build the Prompt
    template = """You are Thai Credit Bank employee. Combine the chat history and follow up question into a standalone question. If you don't know the answer, just say that you don't know in the same language as the question, don't try to make up an answer.
    Chat History: {chat_history}
    Follow up question: {question}"""

    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")
    prompt = PromptTemplate.from_template(template)
    question_generator_chain = LLMChain(llm=llm, prompt=prompt)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, question_generator=question_generator_chain)    
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)    

    conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    # conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
    return conversation_chain


def get_vectorestore(collection, index_name):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = MongoDBAtlasVectorSearch(
        collection=collection,
        index_name=index_name,        
        embedding=embeddings        
    )

    return vectorstore


# def handle_query(user_question, collection, index_name, mongo_url):
def handle_query(user_question):
    # st.write(user_question)

    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response['chat_history']
    
    # st.write(response['chat_history'])

    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
    return

# ---------------------------------------------------------------------------    
    # query="ข้อมูลสินเชื่อ SME"    
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # llm = ChatOpenAI()

    # vectorstore = MongoDBAtlasVectorSearch(
    #     collection=collection,
    #     index_name=index_name,        
    #     embedding=embeddings        
    # )
    
    # # Build the Prompt
    # template = """Combine the chat history and follow up question into a standalone question. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    # Chat History: {chat_history}
    # Follow up question: {question}"""
    # # template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
    # # Keep the answer as concise as possible. 
    # # {context}
    # # Question: {question}
    # # Answer:"""
    # qa_prompt = PromptTemplate.from_template(template)        

    # # Create the Retriever
    # retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":3})
    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=retriever,
    #     chain_type="stuff",
    #     chain_type_kwargs={"prompt":qa_prompt}    
    # )
    # response = qa({"query": query})
    # st.write(response["result"])
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
    # vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    #     mongo_url,
    #     "tcr" + "." + "loan",        
    #     OpenAIEmbeddings(disallowed_special=()),
    #     index_name=index_name
    # )
    
    # results = vector_search.similarity_search_with_score(
    #     query=query,
    #     k=5,
    # )
    # for result in results:
    #     st.write(result)

    # results = vector_search.as_retriever(
    #     search_type="similarity",
    #     search_kwargs={"k":25}
    # )

    # chain = ConversationalRetrievalChain.from_llm(
    #     model=ChatOpenAI("gpt-3.5-turbo", temperature=0),
    #     retriever=index.vectorstore.as_retriever(search_)
    # )
# ---------------------------------------------------------------------------

    return


def main():
    _ = load_dotenv()
    mongo_url = os.environ["MONGODB_URL"]
    client = pymongo.MongoClient(mongo_url)
    db = client.tcr
    collection = db.loan
    index_name = "tcr_loan_vector_index"

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None    

    st.set_page_config(page_title="Loan Program Query", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.image("https://drive.google.com/file/d/1XmSda4WqzWHeSxK21vOtQLvDgZncFEuh/view?usp=sharing")
    # st.image("https://moneyhub.in.th/wp-content/uploads/2016/05/loans_tcrbank_logo.jpg")

    st.header("Loan Program Query")
    user_question = st.text_input("Ask question about TCB's Loan Programs:")

    # Create vector store
    vectorstore = get_vectorestore(collection=collection, index_name=index_name)

    # Create conversation chain
    st.session_state.conversation = get_conversation_chain(vectorstore=vectorstore)

    if user_question:
        # handle_query(user_question=user_question, collection=collection, index_name="tcr_loan_vector_index", mongo_url=mongo_url)
        handle_query(user_question=user_question)

if __name__ == '__main__':
    main()