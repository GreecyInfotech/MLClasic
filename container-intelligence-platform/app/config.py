from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
BASE_DIR = Path(__file__).resolve().parent.parent
class Settings(BaseSettings):
    app_name: str = "Container Intelligence Platform"
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    environment: str = Field(default="dev", alias="APP_ENV")
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8005, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    cors_origins: str = Field(default="*", alias="CORS_ORIGINS")
    api_key: str | None = Field(default=None, alias="API_KEY")
    database_url: str = Field(default=f"sqlite:///{(BASE_DIR / 'container_intelligence.db').as_posix()}", alias="DATABASE_URL")
    dwell_model_path: str = Field(default=str(BASE_DIR / "ml" / "artifacts" / "dwell_time_model.pkl"), alias="DWELL_MODEL_PATH")
    risk_threshold: float = Field(default=0.7, alias="RISK_THRESHOLD")
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    @property
    def cors_origin_list(self) -> list[str]:
        return ["*"] if self.cors_origins.strip() == "*" else [o.strip() for o in self.cors_origins.split(",") if o.strip()]
@lru_cache(maxsize=1)
def get_settings() -> Settings: return Settings()
