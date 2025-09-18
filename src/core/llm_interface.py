from dotenv import load_dotenv, dotenv_values
from typing import List
from openai import OpenAI
from ..schemas.llm_response import KernelsResponse


class LLMClient(OpenAI):
    def __init__(self, api_key: str | None = None, base_url: str | None = None):
        load_dotenv()
        config = dotenv_values()
        api_key = api_key or config.get("OPENAI_API_KEY")
        base_url = base_url or config.get("OPENAI_BASE_URL")
        super().__init__(api_key=api_key, base_url=base_url)

    def generate_kernels(self, prompt: str, num_candidates: int) -> List[str]:
        response = self.responses.parse(
            model="moonshotai/kimi-k2-instruct-0905",
            instructions="You generate CUDA C++ .cu kernels. Return only code strings.",
            input=[{"role": "user", "content": prompt}],
            text_format=KernelsResponse,
        )
        parsed = response.output_parsed
        return [] if parsed is None else parsed.kernels
