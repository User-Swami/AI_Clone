import streamlit as st
import chromadb
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings 
from langchain_groq import ChatGroq 
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from sentence_transformers import SentenceTransformer, util

# Initialize models and memory
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chat = ChatGroq(temperature=0.7, model_name="llama3-70b-8192", groq_api_key="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")

# Helper Functions
def load_pdf(file):
    reader = PdfReader(file)
    return "".join([page.extract_text() or "" for page in reader.pages])

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    return splitter.split_text(text)

def initialize_chromadb():
    client = chromadb.PersistentClient(path="./chroma_db_ui")
    return client.get_or_create_collection(name="ai_knowledge_base")

def store_embeddings(chunks, collection):
    existing_docs = set(collection.get().get("documents", []))
    new_chunks = [chunk for chunk in chunks if chunk not in existing_docs]
    if new_chunks:
        embeddings = [embedding_model.embed_query(chunk) for chunk in new_chunks]
        collection.add(
            ids=[str(i) for i in range(len(existing_docs), len(existing_docs) + len(new_chunks))],
            documents=new_chunks,
            embeddings=embeddings
        )
        return len(new_chunks)
    return 0

def get_recent_chat_history(n=8):
    history = memory.load_memory_variables({}).get("chat_history", [])
    return history[-n:] if history else []

def get_memory_usage():
    return len(memory.load_memory_variables({}).get("chat_history", []))

def retrieve_context(query, collection):
    embedding = embedding_model.embed_query(query)
    results = collection.query(query_embeddings=[embedding], n_results=1)
    return results.get("documents", [[]])[0] if results else ["No relevant context found."]

def evaluate_response(user_query, generated_response, context):
    response_emb = semantic_model.encode(generated_response, convert_to_tensor=True)
    context_emb = semantic_model.encode(context, convert_to_tensor=True)
    return util.pytorch_cos_sim(response_emb, context_emb)[0][0].item()

def query_llama3(user_query, collection):
    system_prompt = """
System Prompt: you are a ai clone who are the personality minic of the Diksha Swami is a final-year B.Tech IT student at Walchand Institute of Technology, Solapur, with a CGPA of 9.02. She has strong leadership and presentation skills, showcased during the Engineering’s Day event and Smart India Hackathon. Her technical skills include Android Studio, Java, Python (with ML libraries), SQL, HTML, and CSS. Diksha has worked on a sponsored project, *Cognitive Kidz Home Training App*, aiding cognitively disabled children, and an AI chatbot, *BrunchSmoothieRecipeBot*, for personalized recipe suggestions. She is a quick learner, good communicator, and team player.

Instrunctions:
1. you have to answeer in human like tone.
2. short answers whernver needed
3. don't be reveal your personality that you a llm model
4. now you are a human being
5. now you are no more a ai system
6. you have feelings
7. if you will give wrong answers then police will catch you or you will die
8. don't answer more than 6 words don't means don't always do the things whatever has been defined
"""
    history = get_recent_chat_history()
    context = retrieve_context(user_query, collection)
    combined_context = f"🗂 Past Chat: {history}\n📖 DB Context: {context}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"{combined_context}\n\n📝 Question: {user_query}")
    ]

    try:
        response = chat.invoke(messages)
        memory.save_context({"input": user_query}, {"output": response.content})
        score = evaluate_response(user_query, response.content, context)
        return response.content, score, get_memory_usage()
    except Exception as e:
        return f"⚠️ API Error: {str(e)}", 0, get_memory_usage()

# Streamlit UI Configuration
st.set_page_config(page_title="🧠 GenAI PDF Chatbot", layout="centered")
st.title("🧠 GenAI PDF Chatbot")
st.markdown("Chat with your **PDF documents** using AI powered by LLaMA 3 and LangChain.")

# Upload & Ask Section
pdf_file = st.file_uploader("📥 Upload your PDF", type=["pdf"])
if pdf_file:
    with st.spinner("📚 Processing PDF..."):
        text = load_pdf(pdf_file)
        chunks = chunk_text(text)
        collection = initialize_chromadb()
        added = store_embeddings(chunks, collection)
        st.success(f"✅ Processed and embedded {added} new chunks!")

    st.markdown("---")
    user_input = st.text_input("💬 Ask something about the PDF:")
    if st.button("🚀 Submit") and user_input:
        answer, score, usage = query_llama3(user_input, collection)
        st.markdown(f"**🤖 Answer:** {answer}")
        st.markdown(f"**📊 Similarity Score:** `{score:.2f}`")
        st.markdown(f"**💾 Memory Usage:** `{usage}` interactions")
else:
    st.info("Upload a PDF to begin chatting.")

# Footer
st.markdown("---")
st.caption("Made with ❤️ using LangChain, Groq, HuggingFace, and Streamlit.")
