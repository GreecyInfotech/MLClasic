from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "AI Loan Approval System"
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    environment: str = Field(default="dev", alias="APP_ENV")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8000, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")
    api_key: str | None = Field(default=None, alias="API_KEY")
    database_url: str = Field(
        default=f"sqlite:///{(BASE_DIR / 'loan_approval.db').as_posix()}",
        alias="DATABASE_URL",
    )
    model_path: str = Field(
        default=str(BASE_DIR / "ml" / "artifacts" / "loan_approval_model.pkl"),
        alias="MODEL_PATH",
    )
    model_threshold: float = Field(default=0.6, alias="MODEL_THRESHOLD")
    high_risk_threshold: float = Field(default=0.7, alias="HIGH_RISK_THRESHOLD")

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
