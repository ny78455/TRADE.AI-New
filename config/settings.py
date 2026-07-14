"""
config/settings.py
------------------
Type-safe, environment-driven configuration using Pydantic Settings.
All secrets and tunable parameters are loaded from environment variables
or a .env file — no hardcoded values.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AlpacaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APCA_", env_file=".env", extra="ignore")

    api_key_id: str = Field(default="", alias="APCA_API_KEY_ID")
    api_secret_key: str = Field(default="", alias="APCA_API_SECRET_KEY")
    api_base_url: str = Field(
        default="https://paper-api.alpaca.markets",
        alias="APCA_API_BASE_URL",
    )

    @property
    def is_paper(self) -> bool:
        return "paper" in self.api_base_url


class BinanceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BINANCE_", env_file=".env", extra="ignore")

    api_key: str = Field(default="", alias="BINANCE_API_KEY")
    api_secret: str = Field(default="", alias="BINANCE_API_SECRET")
    base_url: str = "https://api.binance.com"
    ws_url: str = "wss://stream.binance.com:9443/ws"


class RiskSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RISK_", env_file=".env", extra="ignore")

    max_position_size_usd: float = Field(default=10_000.0, alias="RISK_MAX_POSITION_SIZE_USD")
    max_daily_loss_pct: float = Field(default=5.0, alias="RISK_MAX_DAILY_LOSS_PCT")
    max_open_orders: int = Field(default=10, alias="RISK_MAX_OPEN_ORDERS")
    max_orders_per_second: int = Field(default=5, alias="RISK_MAX_ORDERS_PER_SECOND")


class MLSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_path: Path = Field(default=Path("artifacts/model.h5"), alias="MODEL_PATH")
    faiss_index_path: Path = Field(default=Path("faiss_index.index"), alias="FAISS_INDEX_PATH")
    sbert_model_path: str = Field(default="sbert_model", alias="SBERT_MODEL_PATH")
    artifacts_dir: Path = Field(default=Path("artifacts"), alias="ARTIFACTS_DIR")

    @field_validator("artifacts_dir", "model_path", mode="before")
    @classmethod
    def make_path(cls, v: str | Path) -> Path:
        return Path(v)


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_",
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )

    env: Literal["development", "staging", "production"] = Field(
        default="development", alias="APP_ENV"
    )
    secret_key: str = Field(
        default="change_me_to_a_secure_random_string_min_32_chars",
        alias="APP_SECRET_KEY",
    )
    port: int = Field(default=8000, alias="APP_PORT")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", alias="LOG_LEVEL"
    )

    # JWT
    jwt_secret_key: str = Field(default="change_me", alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_access_token_expire_minutes: int = Field(
        default=60, alias="JWT_ACCESS_TOKEN_EXPIRE_MINUTES"
    )

    # Google
    google_api_key: str = Field(default="", alias="GOOGLE_API_KEY")
    search_engine_id: str = Field(default="", alias="SEARCH_ENGINE_ID")

    # Redis
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")

    # Trading Defaults
    default_ticker: str = Field(default="AVAXUSDT", alias="DEFAULT_TICKER")
    default_interval: str = Field(default="1MINUTE", alias="DEFAULT_INTERVAL")
    default_period: str = Field(default="10 day", alias="DEFAULT_PERIOD")
    scheduler_refresh_seconds: int = Field(default=8, alias="SCHEDULER_REFRESH_SECONDS")

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


class Settings(BaseSettings):
    """Root settings — aggregates all sub-settings."""

    app: AppSettings = AppSettings()
    alpaca: AlpacaSettings = AlpacaSettings()
    binance: BinanceSettings = BinanceSettings()
    risk: RiskSettings = RiskSettings()
    ml: MLSettings = MLSettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached singleton settings instance."""
    return Settings()
