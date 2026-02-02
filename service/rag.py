from __future__ import annotations

import os
import time
from typing import Dict, Iterable, List, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.prompts import ChatPromptTemplate

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None


SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "If insufficient, say you don't know."
)
HUMAN_PROMPT = "Question: {question}\n\nContext:\n{context}"


class MockChatModel(BaseChatModel):
    model_name: str = "mock-rag"

    @property
    def _llm_type(self) -> str:
        return "mock"

    @property
    def _identifying_params(self) -> Dict[str, str]:
        return {"model_name": self.model_name}

    def _generate(  # type: ignore[override]
        self,
        messages: List[BaseMessage],
        stop: List[str] | None = None,
        run_manager=None,
        **kwargs,
    ) -> ChatResult:
        answer = self._build_answer(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=answer))])

    def _build_answer(self, messages: Iterable[BaseMessage]) -> str:
        last = None
        for last in messages:
            pass
        if last is None or not isinstance(last.content, str):
            return "I don't know based on the provided context."

        content = last.content
        context = ""
        if "Context:" in content:
            context = content.split("Context:", 1)[1].strip()

        snippet = _first_context_snippet(context)
        if not snippet:
            return "I don't know based on the provided context."
        return f"Based on the context, {snippet}"


def _first_context_snippet(context: str) -> str:
    segments = [seg.strip() for seg in context.split("\n\n") if seg.strip()]
    if not segments:
        return ""
    lines = [line.strip() for line in segments[0].splitlines() if line.strip()]
    if len(lines) > 1:
        return lines[1]
    return lines[0] if lines else ""


def select_llm() -> Tuple[BaseChatModel, str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key and ChatOpenAI is not None:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(model=model, temperature=0), f"openai:{model}"
    return MockChatModel(), "mock"


def build_context(hits: List[Dict[str, object]], max_chars: int) -> str:
    parts: List[str] = []
    current_len = 0
    separator = "\n\n"

    for hit in hits:
        doc_id = str(hit.get("doc_id", ""))
        rank = hit.get("rank", "")
        snippet = str(hit.get("text_snippet", "")).replace("\n", " ").strip()
        segment = f"[{rank}] doc_id={doc_id}\n{snippet}"
        addition = segment if not parts else f"{separator}{segment}"
        if current_len + len(addition) > max_chars:
            remaining = max_chars - current_len
            if remaining > 0:
                truncated = addition[:remaining].rstrip()
                if truncated:
                    parts.append(truncated)
            break
        parts.append(segment)
        current_len += len(addition)

    return separator.join(parts)


def generate_answer(question: str, context: str) -> Tuple[str, str, float]:
    llm, llm_id = select_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )
    chain = prompt | llm

    start = time.perf_counter()
    result = chain.invoke({"question": question, "context": context})
    generation_ms = (time.perf_counter() - start) * 1000.0

    if hasattr(result, "content"):
        answer = result.content
    else:
        answer = str(result)

    return answer, llm_id, generation_ms
