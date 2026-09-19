"""Runtime configuration, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str
    embed_model: str
    rerank_model: str
    gen_model: str
    vector_store: str
    postgres_dsn: str
    policy_path: Path
    embed_cache_path: Path
    answer_cache_path: Path
    cassette_dir: Path
    cassette_mode: str
    # Exact-match ask cache. Off disables it (eval / ablations that need cold timings).
    answer_cache: bool = True
    # Point this at a second Ollama process to stop the reranker and the generator from
    # evicting each other: with one slot, scoring six candidates and then generating
    # queues both behind the same weights.
    rerank_ollama_url: str = ""
    # Sent with every request, so both models stay resident between questions instead of
    # charging the next user for a cold load.
    keep_alive: str = "30m"

    @staticmethod
    def from_env() -> Settings:
        load_dotenv(REPO_ROOT / ".env")

        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")

        def path_of(key: str, default: str) -> Path:
            raw = Path(os.getenv(key, default))
            return raw if raw.is_absolute() else REPO_ROOT / raw

        return Settings(
            ollama_base_url=ollama_base_url,
            embed_model=os.getenv("EMBED_MODEL", "qwen3-embedding:0.6b"),
            # Qwen3-Reranker's community GGUF is degenerate on Ollama (uniform logprobs
            # for any input), so the generation model doubles as the cross-encoder.
            rerank_model=os.getenv("RERANK_MODEL", "qwen3:8b"),
            gen_model=os.getenv("GEN_MODEL", "qwen3:8b"),
            vector_store=os.getenv("VECTOR_STORE", "postgres"),
            postgres_dsn=os.getenv("POSTGRES_DSN", "postgresql://rag:rag@localhost:5433/ragpolicy"),
            policy_path=path_of("POLICY_PATH", "policy.md"),
            embed_cache_path=path_of("EMBED_CACHE_PATH", ".cache/embeddings.sqlite"),
            answer_cache_path=path_of("ANSWER_CACHE_PATH", ".cache/answers.sqlite"),
            cassette_dir=path_of("CASSETTE_DIR", "tests/cassettes"),
            cassette_mode=os.getenv("CASSETTE_MODE", "off"),
            rerank_ollama_url=os.getenv("RERANK_OLLAMA_URL", ollama_base_url).rstrip("/"),
            keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "30m"),
            answer_cache=os.getenv("ANSWER_CACHE", "on").lower() not in {"0", "off", "false", "no"},
        )
