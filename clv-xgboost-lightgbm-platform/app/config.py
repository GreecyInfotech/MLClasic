from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "CLV XGBoost LightGBM Platform"
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    environment: str = Field(default="dev", alias="APP_ENV")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8001, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")
    api_key: str | None = Field(default=None, alias="API_KEY")
    database_url: str = Field(
        default=f"sqlite:///{(BASE_DIR / 'clv_platform.db').as_posix()}",
        alias="DATABASE_URL",
    )
    xgboost_model_path: str = Field(
        default=str(BASE_DIR / "ml" / "artifacts" / "xgboost_clv.json"),
        alias="XGBOOST_MODEL_PATH",
    )
    lightgbm_model_path: str = Field(
        default=str(BASE_DIR / "ml" / "artifacts" / "lightgbm_clv.txt"),
        alias="LIGHTGBM_MODEL_PATH",
    )
    default_model: str = Field(default="ensemble", alias="DEFAULT_MODEL")

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
