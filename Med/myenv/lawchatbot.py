import streamlit as st
import os
import re
from pathlib import Path
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
from pydantic import PrivateAttr

# --- LangChain Imports ---
from langchain_core.language_models.llms import LLM
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Groq Model Import ---
from groq import Groq
from groq import AuthenticationError as GroqAuthenticationError

try:
    from docx import Document
except ImportError:
    Document = None

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except ImportError:
    letter = None
    canvas = None

# Set page configuration as the very first Streamlit command
st.set_page_config(page_title="AI Personal Research Assistant", layout="wide")

# Load environment variables (e.g., any API keys or configuration)
SCRIPT_DIR = Path(__file__).resolve().parent
load_dotenv(SCRIPT_DIR / ".env")
load_dotenv()

def apply_professional_theme(theme_mode):
    dark_mode = True
    app_bg = "radial-gradient(circle at 20% 10%, #0f172a 0%, #111827 45%, #020617 100%)" if dark_mode else "radial-gradient(circle at 20% 10%, #f3f7ff 0%, #eef2f9 35%, #f6f8fc 100%)"
    text_color = "#e5e7eb" if dark_mode else "#0f172a"
    panel_bg = "rgba(15, 23, 42, 0.65)" if dark_mode else "rgba(255, 255, 255, 0.9)"
    panel_border = "rgba(148, 163, 184, 0.35)" if dark_mode else "rgba(148, 163, 184, 0.25)"
    chip_bg = "#1e293b" if dark_mode else "#e2e8f0"
    chip_text = "#e5e7eb" if dark_mode else "#0f172a"
    subtle_note = "#cbd5e1" if dark_mode else "#334155"
    heading_color = "#f8fafc" if dark_mode else "#0b1220"
    section_title = "#e2e8f0" if dark_mode else "#0f172a"
    button_bg = "#1d4ed8" if dark_mode else "#0f172a"
    button_text = "#f8fafc"
    button_border = "#60a5fa" if dark_mode else "#0f172a"

    theme_css = """
        <style>
            .stApp {{
                background: {app_bg};
                color: {text_color};
            }}
            .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
            .stApp p, .stApp label, .stApp li,
            [data-testid="stMarkdownContainer"] p,
            [data-testid="stMarkdownContainer"] div,
            [data-testid="stMarkdownContainer"] span {{
                color: {text_color};
            }}
            [data-testid="stSidebar"] {{
                background: {panel_bg};
                border-right: 1px solid {panel_border};
            }}
            .section-title {{
                color: {section_title};
                font-size: 1.05rem;
                font-weight: 700;
                margin: 6px 0 10px 0;
            }}
            .chatgpt-title {{
                color: {heading_color};
                font-size: 1.6rem;
                font-weight: 700;
                margin: 4px 0 2px 0;
            }}
            .chatgpt-subtitle {{
                color: {subtle_note};
                font-size: 0.92rem;
                margin: 0 0 16px 0;
            }}
            .subtle-note {{
                color: {subtle_note};
                font-size: 0.94rem;
            }}
            .citation-chip {{
                display: inline-block;
                background: {chip_bg};
                color: {chip_text};
                border-radius: 999px;
                padding: 2px 10px;
                margin-bottom: 8px;
                font-size: 0.78rem;
                font-weight: 600;
            }}
            [data-testid="stChatMessage"] {{
                background: {panel_bg};
                border: 1px solid {panel_border};
                border-radius: 12px;
            }}
            [data-testid="stChatMessageContent"],
            [data-testid="stChatMessageContent"] p,
            [data-testid="stChatMessageContent"] div,
            [data-testid="stChatMessageContent"] span {{
                color: {text_color};
            }}
            .stTextArea textarea {{
                background: {panel_bg};
                color: {text_color};
                -webkit-text-fill-color: {text_color};
                border: 1px solid {panel_border};
            }}
            .stTextInput input {{
                color: {text_color};
                -webkit-text-fill-color: {text_color};
            }}
            .stButton > button {{
                border-radius: 10px;
                background: {button_bg};
                color: {button_text};
                border: 1px solid {button_border};
                font-weight: 600;
            }}
            .stButton > button:hover {{
                filter: brightness(1.08);
                color: {button_text};
            }}
            .stButton > button:focus {{
                color: {button_text};
            }}
            [data-testid="stChatInput"] textarea {{
                background: {panel_bg};
                color: {text_color};
                border: 1px solid {panel_border};
            }}
        </style>
        """.format(
        app_bg=app_bg,
        text_color=text_color,
        panel_bg=panel_bg,
        panel_border=panel_border,
        subtle_note=subtle_note,
        chip_bg=chip_bg,
        chip_text=chip_text,
        heading_color=heading_color,
        section_title=section_title,
        button_bg=button_bg,
        button_text=button_text,
        button_border=button_border,
    )
    st.markdown(theme_css, unsafe_allow_html=True)


# Sidebar configuration for the research assistant
st.sidebar.title("Terry")
st.sidebar.markdown("Chat with your legal PDFs in a ChatGPT-style interface.")

apply_professional_theme("Dark")


# =============================================================================
# Custom Groq LLM Wrapper for LangChain
# =============================================================================
class GroqLLM(LLM):
    """
    A simple LangChain LLM wrapper to use the Groq model.
    Adjust the _call method as needed based on your Groq model’s interface.
    """
    
    model_id: str
    _client: Groq = PrivateAttr()

    def __init__(self, model_id: str):
        super().__init__(model_id=model_id)
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is missing. Add it to Med/myenv/.env or your system environment.")
        self._client = Groq(api_key=groq_api_key)
    
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _call(self, prompt: str, stop=None, run_manager=None, **kwargs) -> str:
        system_prompt = (
            "You are an expert AI research assistant. "
            "Always provide thorough, well-structured, and detailed answers. "
            "Use headings, bullet points, or numbered lists where appropriate. "
            "Explain concepts fully with reasoning and examples where relevant. "
            "Never give one-line or incomplete answers — always elaborate as much as the context allows."
        )
        response = self._client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=2048,
            temperature=0.4,
            stream=False,
        )
        text = response.choices[0].message.content or ""
        if stop:
            for token in stop:
                if token in text:
                    text = text.split(token)[0]
        return text

# =============================================================================
# Document Ingestion and FAISS Vector Store Creation
#
# This section loads PDFs from a directory (e.g., "./docs") and builds a FAISS vector store.
# Adjust the directory and loaders if you use other document types or sources.
# =============================================================================
def resolve_docs_path(docs_dir: str = "docs"):
    docs_path = Path(docs_dir)
    if not docs_path.is_absolute():
        docs_path = SCRIPT_DIR / docs_path
    return docs_path


@st.cache_resource(show_spinner=False)
def load_documents_and_create_vectorstore(docs_dir: str = "docs"):
    docs_path = resolve_docs_path(docs_dir)
    if not docs_path.exists() or not docs_path.is_dir():
        raise FileNotFoundError(
            f"Documents directory not found: {docs_path}. Create this folder and add PDF files."
        )

    all_docs = []
    # Loop over files in the given directory and load PDFs
    for file_name in os.listdir(docs_path):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(docs_path, file_name)
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            all_docs.extend(docs)
    if not all_docs:
        raise ValueError(f"No PDF documents found in {docs_path}. Add at least one PDF file.")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        vectorstore = FAISS.from_documents(all_docs, embeddings)
    except ImportError:
        st.warning(
            "FAISS could not be imported at runtime. Falling back to InMemoryVectorStore."
        )
        vectorstore = InMemoryVectorStore.from_documents(all_docs, embeddings)
    return vectorstore


def get_document_catalog(docs_dir: str = "docs"):
    docs_path = resolve_docs_path(docs_dir)
    if not docs_path.exists() or not docs_path.is_dir():
        return []
    return sorted([f.name for f in docs_path.iterdir() if f.is_file() and f.suffix.lower() == ".pdf"])


BLOCKED_EXPORT_NAMES = {"research_answer.pdf", "research_answer.txt", "research_answer.docx"}


def save_uploaded_pdfs(uploaded_files, docs_dir: str = "docs"):
    docs_path = resolve_docs_path(docs_dir)
    docs_path.mkdir(parents=True, exist_ok=True)
    saved = []
    skipped = []
    for uploaded_file in uploaded_files or []:
        file_name = Path(uploaded_file.name).name
        if file_name.lower() in BLOCKED_EXPORT_NAMES:
            skipped.append(file_name)
            continue
        target_path = docs_path / file_name
        with open(target_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved.append(file_name)
    if skipped:
        st.sidebar.warning(f"Skipped export file(s): {', '.join(skipped)}. Do not upload app-generated exports as source documents.")
    return saved


def delete_pdf(file_name: str, docs_dir: str = "docs"):
    target = resolve_docs_path(docs_dir) / Path(file_name).name
    if target.exists() and target.suffix.lower() == ".pdf":
        target.unlink()


def build_qa_chain():
    """Build vectorstore and QA chain. Returns None if no PDFs are available."""
    catalog = get_document_catalog()
    if not catalog:
        return None, None
    try:
        vs = load_documents_and_create_vectorstore()
    except Exception:
        return None, None
    _llm = GroqLLM(model_id=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    _retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    _qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an expert AI research assistant. "
            "Use the following source excerpts to answer the question in full detail.\n"
            "Provide a comprehensive, well-structured answer with explanations, reasoning, "
            "and examples where relevant. Use headings or bullet points when helpful. "
            "Do not give a one-line answer — elaborate thoroughly based on the context provided.\n\n"
            "Source Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Detailed Answer:"
        ),
    )
    _chain = RetrievalQA.from_chain_type(
        llm=_llm,
        retriever=_retriever,
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": _qa_prompt},
    )
    return vs, _chain

# =============================================================================
# Utility Functions to Clean Output
# =============================================================================
def strip_ansi_codes(text):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
    return ansi_escape.sub('', text)

def remove_box_drawing(text):
    box_chars = "┏┓┗┛┃━"
    for ch in box_chars:
        text = text.replace(ch, "")
    return text

def clean_output(text):
    text = strip_ansi_codes(text)
    text = remove_box_drawing(text)
    return text


def normalize_page_number(page_value):
    if isinstance(page_value, int):
        return page_value + 1
    return page_value


def extract_citations(source_documents):
    citations = []
    for idx, doc in enumerate(source_documents or [], start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        source_path = metadata.get("source", "Unknown source")
        source_name = Path(str(source_path)).name if source_path else "Unknown source"
        page = normalize_page_number(metadata.get("page", "N/A"))
        excerpt = (getattr(doc, "page_content", "") or "").strip().replace("\n", " ")
        excerpt = excerpt[:360] + ("..." if len(excerpt) > 360 else "")
        citations.append(
            {
                "index": idx,
                "source": source_name,
                "page": page,
                "excerpt": excerpt or "No excerpt available.",
            }
        )
    return citations


def ask_research_assistant(query, chain):
    result = chain.invoke({"query": query})
    answer_text = result.get("result", "") if isinstance(result, dict) else str(result)
    source_documents = result.get("source_documents", []) if isinstance(result, dict) else []
    return clean_output(answer_text), extract_citations(source_documents)


def build_export_text(message):
    lines = [
        "AI Personal Research Assistant",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Answer:",
        message.get("text", ""),
        "",
        "Citations:",
    ]
    citations = message.get("citations", [])
    if not citations:
        lines.append("- No citations available")
    for citation in citations:
        lines.append(f"- [{citation['index']}] {citation['source']} (page {citation['page']})")
        lines.append(f"  Excerpt: {citation['excerpt']}")
    return "\n".join(lines)


def create_pdf_bytes(text):
    if canvas is None or letter is None:
        return None
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    x_margin = 40
    y = height - 40
    for raw_line in text.splitlines() or [""]:
        line = raw_line if raw_line else " "
        while len(line) > 110:
            pdf.drawString(x_margin, y, line[:110])
            line = line[110:]
            y -= 14
            if y < 40:
                pdf.showPage()
                y = height - 40
        pdf.drawString(x_margin, y, line)
        y -= 14
        if y < 40:
            pdf.showPage()
            y = height - 40
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


def create_docx_bytes(text):
    if Document is None:
        return None
    doc = Document()
    for line in text.splitlines() or [""]:
        doc.add_paragraph(line)
    buffer = BytesIO()
    doc.save(buffer)
    buffer.seek(0)
    return buffer.getvalue()


def render_citations(citations):
    if not citations:
        st.caption("No citations were returned for this response.")
        return
    st.markdown("**Sources**")
    for citation in citations:
        st.markdown(
            f"<span class='citation-chip'>[{citation['index']}] {citation['source']} | page {citation['page']}</span>",
            unsafe_allow_html=True,
        )
        with st.expander(f"Open source chunk [{citation['index']}]", expanded=False):
            st.write(citation["excerpt"])

# =============================================================================
# Streamlit UI Setup
# =============================================================================
# Initialize session state for chat history if not already set
if "messages" not in st.session_state:
    st.session_state.messages = []


def append_user_message(text):
    st.session_state.messages.append({"sender": "user", "text": text})


def append_bot_message(text, citations=None, question=""):
    st.session_state.messages.append(
        {
            "sender": "bot",
            "text": text,
            "citations": citations or [],
            "question": question,
        }
    )


def process_query(query, chain):
    append_user_message(query)
    with st.spinner("Searching documents and drafting answer..."):
        try:
            answer, citations = ask_research_assistant(query, chain)
            append_bot_message(answer, citations=citations, question=query)
        except GroqAuthenticationError:
            auth_message = (
                "Groq authentication failed: invalid GROQ_API_KEY. "
                "Update GROQ_API_KEY in Med/myenv/.env with a valid key from console.groq.com."
            )
            st.error(auth_message)
            append_bot_message(auth_message, citations=[], question=query)
        except Exception as exc:
            error_message = f"Request failed: {exc}"
            st.error(error_message)
            append_bot_message(error_message, citations=[], question=query)


def get_last_bot_message():
    for message in reversed(st.session_state.messages):
        if message.get("sender") == "bot":
            return message
    return None

def get_recent_user_prompts(limit=6):
    prompts = [m.get("text", "") for m in st.session_state.messages if m.get("sender") == "user"]
    prompts = [p for p in prompts if p]
    return prompts[-limit:][::-1]


if st.sidebar.button("+ New Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

catalog = get_document_catalog()
st.sidebar.markdown(f"**Indexed PDFs ({len(catalog)})**")
if catalog:
    for pdf_name in catalog:
        col_name, col_del = st.sidebar.columns([5, 1])
        col_name.caption(pdf_name[:34] + ("..." if len(pdf_name) > 34 else ""))
        if col_del.button("🗑", key=f"del_{pdf_name}", help=f"Remove {pdf_name}"):
            delete_pdf(pdf_name)
            load_documents_and_create_vectorstore.clear()
            st.rerun()
else:
    st.sidebar.caption("No PDFs indexed yet. Upload one below.")

recent_prompts = get_recent_user_prompts()
if recent_prompts:
    st.sidebar.markdown("**Recent prompts**")
    for prompt in recent_prompts:
        st.sidebar.caption(prompt[:80] + ("..." if len(prompt) > 80 else ""))

st.sidebar.markdown("**Upload PDFs**")
uploaded_files = st.sidebar.file_uploader(
    "Add one or more PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)
if st.sidebar.button("Save & Reindex", use_container_width=True):
    saved = save_uploaded_pdfs(uploaded_files)
    if saved:
        load_documents_and_create_vectorstore.clear()
        st.sidebar.success(f"Saved: {', '.join(saved)}")
        st.rerun()
    else:
        st.sidebar.warning("Select at least one PDF file first.")


st.markdown("<div class='chatgpt-title'>Terry</div>", unsafe_allow_html=True)
st.markdown("<div class='chatgpt-subtitle'>Ask anything about your uploaded legal documents.</div>", unsafe_allow_html=True)

catalog_main = get_document_catalog()
_vectorstore, _qa_chain = build_qa_chain() if catalog_main else (None, None)

if not catalog_main:
    st.warning("No PDFs indexed yet. Upload one or more PDF files using the sidebar to get started.")
elif not st.session_state.messages:
    st.info("Start by asking a legal question in the chat box below.")

for msg in st.session_state.messages:
    role = "assistant" if msg.get("sender") == "bot" else "user"
    with st.chat_message(role):
        st.write(msg.get("text", ""))
        if role == "assistant":
            render_citations(msg.get("citations", []))

if catalog_main:
    user_prompt = st.chat_input("Message Terry")
    if user_prompt and user_prompt.strip():
        process_query(user_prompt.strip(), _qa_chain)
        st.rerun()
else:
    st.chat_input("Upload a PDF first to start chatting", disabled=True)

last_bot = get_last_bot_message()

if last_bot:
    export_text = build_export_text(last_bot)
    pdf_bytes = create_pdf_bytes(export_text)
    docx_bytes = create_docx_bytes(export_text)
    st.sidebar.markdown("**Export latest answer**")
    st.sidebar.download_button(
        label="Export Answer to PDF",
        data=pdf_bytes if pdf_bytes else export_text.encode("utf-8"),
        file_name="research_answer.pdf" if pdf_bytes else "research_answer.txt",
        mime="application/pdf" if pdf_bytes else "text/plain",
        use_container_width=True,
    )
    st.sidebar.download_button(
        label="Export Answer to Word",
        data=docx_bytes if docx_bytes else export_text.encode("utf-8"),
        file_name="research_answer.docx" if docx_bytes else "research_answer.txt",
        mime=(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            if docx_bytes
            else "text/plain"
        ),
        use_container_width=True,
    )

st.markdown("<p class='subtle-note'>Built with Groq + LangChain retrieval.</p>", unsafe_allow_html=True)
