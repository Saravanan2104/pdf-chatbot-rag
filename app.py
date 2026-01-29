import gradio as gr
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# Global objects
retrieval_chain = None

# Groq LLM
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0
)

def process_pdf(pdf_file):
    global retrieval_chain

    try:
        if pdf_file is None:
            return "Please upload a PDF file."

        # Gradio-safe file path
        file_path = pdf_file["path"] if isinstance(pdf_file, dict) else pdf_file.name

        loader = PyPDFLoader(file_path)
        documents = loader.load()

        # CHECK 1: PDF content
        if not documents or len(documents) == 0:
            return "No readable text found in PDF (scanned PDFs not supported)."

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        docs = splitter.split_documents(documents)

        # CHECK 2: Split content
        if not docs or len(docs) == 0:
            return "Text splitting failed. PDF may be empty."

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()

        prompt = ChatPromptTemplate.from_template(
            """
            Answer the question ONLY using the context below.
            If the answer is not in the context, say "I don't know".

            Context:
            {context}

            Question:
            {input}
            """
        )

        document_chain = create_stuff_documents_chain(llm, prompt)
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        return "Document processed successfully. You can now ask questions."

    except Exception as e:
        return f"Error processing document: {str(e)}"



def answer_question(question):
    if retrieval_chain is None:
        return "Please upload and process a document first."

    if not question.strip():
        return "Please enter a valid question."

    try:
        result = retrieval_chain.invoke({"input": question})

        # SAFE result handling (works for all versions)
        if isinstance(result, dict):
            return (
                result.get("answer")
                or result.get("output")
                or result.get("result")
                or "No answer generated."
            )

        return str(result)

    except Exception as e:
        return f"Error generating answer: {str(e)}"


with gr.Blocks(title="AI Customer Support Assistant") as demo:
    gr.Markdown("## AI PDF ChatBot Assistant (Groq + RAG)")
    gr.Markdown("Upload a PDF and ask questions based on its content.")

    pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
    process_btn = gr.Button("Process Document")
    status_output = gr.Textbox(label="Status")

    process_btn.click(
        fn=process_pdf,
        inputs=pdf_input,
        outputs=status_output
    )

    question_input = gr.Textbox(label="Ask a Question")
    answer_output = gr.Textbox(label="Answer")

    question_input.submit(
        fn=answer_question,
        inputs=question_input,
        outputs=answer_output
    )

demo.launch()
