from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from ..schemas.llm_response import KernelsResponse


class LLMClient:
    """
    LangChain Groq-based LLM client that generates CUDA kernels with structured output.

    - Backend: Groq ChatGroq
    - Default model: "openai/gpt-oss-120b"
    - Output: Structured into KernelsResponse (pydantic) via with_structured_output
    - Auth: Reads GROQ_API_KEY from environment or accepts api_key in constructor
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        timeout: float | None = None,
        max_retries: int = 2,
    ) -> None:
        load_dotenv()

        # If api_key is provided explicitly, set env so ChatGroq can pick it up
        if api_key:
            os.environ["GROQ_API_KEY"] = api_key

        # Instantiate Groq chat model via LangChain
        self._model_name = model or "openai/gpt-oss-120b"
        self.llm = ChatGroq(
            model=self._model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )

        # Prepare a structured-output wrapper for KernelsResponse
        self.structured = self.llm.with_structured_output(KernelsResponse)

        # Prompt template guiding the model to return multiple candidate kernels
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You generate CUDA C++ .cu kernels for the given host-side task and constraints.\n"
                        "- Return only compilable CUDA translation units as strings.\n"
                        "- Provide up to {num_candidates} diverse candidate kernels that solve the task.\n"
                        "- Do NOT include prose, markdown, comments, or explanations in the outputs.\n"
                        "- Respect any architectural hints (e.g., {arch}) provided by the user prompt.\n"
                        "- Output must strictly follow the provided structured schema."
                    ),
                ),
                ("human", "{user_prompt}"),
            ]
        )

    def generate_kernels(self, prompt: str, num_candidates: int) -> List[str]:
        """
        Generate up to `num_candidates` CUDA kernels using structured output.

        Args:
            prompt: Full user prompt containing task description and host code.
            num_candidates: Maximum number of kernels to request.

        Returns:
            List[str]: CUDA kernel candidates (each a complete .cu translation unit).
        """
        # Compose chain: prompt -> structured LLM
        chain = self.prompt | self.structured

        # Allow callers to inject arch hint via the prompt if they included it;
        # we still pass a placeholder to keep the template variables consistent.
        variables = {
            "user_prompt": prompt,
            "num_candidates": max(1, int(num_candidates)),
            # No strong assumption about arch; it's extracted from user prompt by the model.
            "arch": "",
        }

        result: KernelsResponse = chain.invoke(variables)

        kernels = result.kernels if result and result.kernels else []
        # Enforce max size and sanitize
        cleaned = []
        for k in kernels[:num_candidates]:
            if not isinstance(k, str):
                continue
            s = k.strip()
            if s:
                cleaned.append(s)
        return cleaned
