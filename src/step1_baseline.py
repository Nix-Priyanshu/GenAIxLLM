"""Step 1: answer a question with the LLM alone -- no documents, no tools.
This is the baseline that Step 2 (RAG) must beat.
"""
try:
    from src.config import SYSTEM_PROMPT_BASE, call_llm
except ImportError:
    from config import SYSTEM_PROMPT_BASE, call_llm


def answer(question: str) -> str:
    return call_llm(SYSTEM_PROMPT_BASE, question)


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) or "How long do I have to return a pair of shoes?"
    print(answer(q))
