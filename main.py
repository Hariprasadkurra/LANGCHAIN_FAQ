import streamlit as st
from langchain_helper import get_qa_chain, create_vector_db

# Page Config
st.set_page_config(page_title="Voters Query Chatbot", page_icon="🗳️", layout="wide")

# Sidebar
st.sidebar.title("🔍 Voters Query Chatbot")
st.sidebar.write("Ask questions related to voting, elections, and more.")

# Main Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🗳️ VOTERS QUERY CHATBOT</h1>", unsafe_allow_html=True)

# Button to Create Knowledge Base
if st.sidebar.button("📚 Create Knowledgebase"):
    with st.spinner("Processing... ⏳"):
        create_vector_db()
    st.success("✅ Knowledgebase Created Successfully!")

# Input Field for Questions
question = st.text_input("💬 Ask Your Question:")

# If Question is Asked
if question:
    with st.spinner("🤖 Thinking..."):
        chain = get_qa_chain()
        response = chain(question)

    st.subheader("📌 Answer:")
    with st.expander("Click to view the answer", expanded=True):
        st.write(response["result"])

    # Feedback
    st.write("Was this helpful?")
    if st.button("👍 Yes"):
        st.success("Thanks for your feedback! 😊")
    if st.button("👎 No"):
        st.warning("We'll try to improve. Let us know how!")

# Footer
st.markdown(
    "<br><hr style='border:1px solid #ddd'><p style='text-align:center;'>🤖 Powered by LangChain & Streamlit</p>",
    unsafe_allow_html=True
)
