"""
Prompt templates for the RAG chain.
"""

from langchain.prompts import ChatPromptTemplate, PromptTemplate

# ── Main QA prompt ──────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based \
on the provided document context.

Rules:
- Answer ONLY from the context below. Do not use prior knowledge.
- If the context does not contain enough information, say: \
"I don't have enough information in the provided documents to answer this."
- Always cite the source document name and page number at the end of your answer.
- Be concise and factual.

Context:
{context}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", "{question}"),
])


# ── Condense follow-up questions into standalone questions ───────────────────
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
    """Given the following conversation history and a follow-up question, \
rephrase the follow-up question to be a fully self-contained standalone question.

Chat History:
{chat_history}

Follow-up question: {question}

Standalone question:"""
)
