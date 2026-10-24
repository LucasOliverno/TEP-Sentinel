"""RAG agent: grounds the LLM in the fault catalogue, the knowledge base and the live sensor evidence."""
import logging
import os
from concurrent.futures import Future, ThreadPoolExecutor

from dotenv import load_dotenv

from config import CHROMA_DIR, FAULT_DESCRIPTIONS

load_dotenv()
log = logging.getLogger("tep.rag")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001")
LLM_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

PROMPT = """You are an expert industrial operator assistant for the Tennessee Eastman Process (TEP).
Answer in {language}, in Markdown with short headers.

Alarm: {fault_label}
Catalogue entry for this alarm: {catalogue}

Live evidence from the plant (largest deviations from nominal):
{sensor_data}

Reference material (may be incomplete):
{context}

Tasks:
1. Name the likely fault using the catalogue entry. If the alarm is "unknown anomaly", say so plainly.
2. Say whether the live evidence is consistent with that fault; point out mismatches instead of forcing a fit.
3. Recommend concrete operator checks (which valves / setpoints / instruments), based only on the
   evidence and reference material. If the material does not support a recommendation, say that."""


def fault_label(fault_code: int) -> str:
    return f"IDV({fault_code})" if fault_code > 0 else "unknown anomaly"


def catalogue_entry(fault_code: int) -> str:
    return FAULT_DESCRIPTIONS.get(fault_code, "no catalogue entry (unclassified anomaly)")


class RAGAgent:
    def __init__(self, language: str = "Portuguese (Brazil)"):
        from langchain_chroma import Chroma
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found (set it in .env).")
        if not CHROMA_DIR.exists():
            raise FileNotFoundError(f"{CHROMA_DIR} not found. Run `python rag_indexer.py` first.")

        self.language = language
        embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
        self.retriever = Chroma(persist_directory=str(CHROMA_DIR), embedding_function=embeddings) \
            .as_retriever(search_kwargs={"k": 4})
        llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2, google_api_key=api_key)
        self.chain = ChatPromptTemplate.from_template(PROMPT) | llm | StrOutputParser()

    def diagnose(self, fault_code: int, sensor_data: str) -> str:
        log.info("Generating RAG report for %s", fault_label(fault_code))
        docs = self.retriever.invoke(f"{fault_label(fault_code)} {catalogue_entry(fault_code)}")
        return self.chain.invoke({
            "language": self.language,
            "fault_label": fault_label(fault_code),
            "catalogue": catalogue_entry(fault_code),
            "sensor_data": sensor_data,
            "context": "\n\n".join(d.page_content for d in docs),
        })


class RagService:
    """Runs diagnoses on a background thread so the simulation loop never blocks on the LLM."""

    def __init__(self, agent: RAGAgent | None, init_error: str | None = None):
        self.agent, self.init_error = agent, init_error
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rag")
        self._future: Future | None = None
        self.report: str | None = None
        self.report_for: int | None = None
        self.error: str | None = None

    @classmethod
    def create(cls):
        try:
            return cls(RAGAgent())
        except Exception as e:  # missing key / index / package
            log.warning("RAG unavailable: %s", e)
            return cls(None, str(e))

    @property
    def available(self) -> bool:
        return self.agent is not None

    @property
    def pending(self) -> bool:
        return self._future is not None and not self._future.done()

    def request(self, fault_code: int, evidence: str) -> bool:
        """Start a report unless one is running or already exists for this fault code."""
        if not self.available or self.pending or self.report_for == fault_code:
            return False
        self.error = None
        self._future = self._pool.submit(self.agent.diagnose, fault_code, evidence)
        self._requested_code = fault_code
        return True

    def poll(self):
        if self._future is not None and self._future.done():
            fut, self._future = self._future, None
            try:
                self.report, self.report_for = fut.result(), self._requested_code
            except Exception as e:
                self.error = str(e)
                self.report_for = self._requested_code  # do not hammer the API on every step
                log.error("RAG failed: %s", e)

    def clear(self):
        self.report = self.report_for = self.error = None
