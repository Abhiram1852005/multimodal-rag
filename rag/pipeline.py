from typing import Dict, Any
from rag.retriever import retrieve
from rag.llm import generate_with_citations

def answer_question(question: str) -> Dict[str, Any]:
    contexts = retrieve(question)
    answer = generate_with_citations(question, contexts) if contexts else \
        "I couldn't find relevant context in the knowledge base. Please ingest files first."
    return {
        "answer": answer,
        "contexts": [{"source": s, "score": sc, "preview": t[:300]} for (t,s,sc) in contexts]
    }
