"""Settings loaded from environment variables (or a local .env file)."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
    # GPT-OSS models "reason" before answering; routing and short answers need little of it.
    groq_reasoning_effort: str = os.getenv("GROQ_REASONING_EFFORT", "low")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    superhero_api_token: str = os.getenv("SUPERHERO_API_TOKEN", "")
    # The final address: superheroapi.com/api redirects here, so calling it directly saves a round trip.
    superhero_base_url: str = os.getenv("SUPERHERO_BASE_URL", "https://www.superheroapi.com/api.php")
    data_dir: Path = Path(os.getenv("DATA_DIR", BASE_DIR / "data" / "comics"))
    http_timeout_s: float = float(os.getenv("HTTP_TIMEOUT_S", "15"))
    max_heroes_per_question: int = 3
    top_k_passages: int = 3
    retrieval_min_score: float = float(os.getenv("RETRIEVAL_MIN_SCORE", "0.1"))
    memory_turns: int = 5          # question/answer pairs remembered per session
    memory_ttl_s: float = 1800     # forget a session after 30 minutes idle


settings = Settings()
