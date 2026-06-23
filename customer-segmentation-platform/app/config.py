from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    app_name: str = "Customer Segmentation Platform"
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    environment: str = Field(default="dev", alias="APP_ENV")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8002, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")
    api_key: str | None = Field(default=None, alias="API_KEY")
    database_url: str = Field(
        default=f"sqlite:///{(BASE_DIR / 'segmentation.db').as_posix()}",
        alias="DATABASE_URL",
    )
    model_path: str = Field(
        default=str(BASE_DIR / "ml-platform" / "artifacts" / "kmeans_segmentation.pkl"),
        alias="MODEL_PATH",
    )
    n_clusters: int = Field(default=4, alias="N_CLUSTERS")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
