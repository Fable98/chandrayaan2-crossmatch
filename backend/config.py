"""
config.py — Application configuration using Pydantic BaseSettings.

Reads from environment variables and optional .env file with type validation
and sensible defaults for local development and Render production deployment.
"""

from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(
            str(BACKEND_DIR / ".env"),
            str(REPO_ROOT / ".env"),
            ".env",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server configuration
    HOST: str = Field(default="0.0.0.0", description="Host address to bind to")
    PORT: int = Field(default=8000, description="Port to listen on (injected by Render via $PORT)")
    ENVIRONMENT: str = Field(default="production", description="Environment mode: development | production")

    # CORS configuration
    # Comma-separated list of allowed origins, e.g. "https://lunar-frontend.onrender.com,http://localhost:3000"
    ALLOWED_ORIGINS: str = Field(
        default="*",
        description="Comma-separated allowed CORS origins (e.g. https://my-site.onrender.com,http://localhost:3000)",
    )
    CORS_ORIGIN: Optional[str] = Field(
        default=None,
        description="Legacy single origin fallback",
    )

    # Supabase / PostgreSQL configuration (optional — JSON flat files remain primary data store)
    DATABASE_URL: Optional[str] = Field(
        default=None,
        description="Supabase / PostgreSQL connection URI (postgresql://user:pass@host:port/db)",
    )
    SUPABASE_URL: Optional[str] = Field(
        default=None,
        description="Supabase project URL",
    )
    SUPABASE_ANON_KEY: Optional[str] = Field(
        default=None,
        description="Supabase anonymous public key",
    )

    # Data directory paths (defaults dynamically resolve to repo directories)
    DATA_DIR: Optional[str] = Field(
        default=None,
        description="Override path for preprocessed data directory (default: repo root / data_preprocessing_pipeline or processed_user)",
    )
    PROCESSED_TRIPLETS_DIR: Optional[str] = Field(
        default=None,
        description="Override path for processed triplets directory",
    )
    ML_OUTPUT_DIR: Optional[str] = Field(
        default=None,
        description="Override path for ML output directory (default: repo root / ML_model)",
    )

    # JWT Authentication
    JWT_SECRET_KEY: str = Field(
        default="chandrayaan2-crossmatch-dev-secret-change-in-production",
        description="Secret key for signing JWT tokens (override in production!)",
    )
    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="Algorithm for JWT token encoding",
    )
    JWT_EXPIRY_MINUTES: int = Field(
        default=1440,
        description="JWT token expiry time in minutes (default: 24 hours)",
    )

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse comma-separated ALLOWED_ORIGINS into a sanitized list of origins."""
        origins: List[str] = []
        if self.ALLOWED_ORIGINS:
            origins.extend([o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()])
        if self.CORS_ORIGIN:
            cleaned = self.CORS_ORIGIN.strip()
            if cleaned and cleaned not in origins:
                origins.append(cleaned)
        if not origins:
            origins = ["*"]
        return origins


settings = Settings()

# Re-export canonical ML configuration constants
try:
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.append(str(REPO_ROOT))
    if str(REPO_ROOT / "ML_model") not in sys.path:
        sys.path.append(str(REPO_ROOT / "ML_model"))
    from ML_model.config import (
        OHRC_GSD,
        TMC_GSD,
        IIRS_GSD,
        LRO_NAC_GSD,
        SEED,
        SENSOR_GSD_MAP,
        get_seed,
        get_sensor_gsd,
    )
except Exception:
    OHRC_GSD = 0.25
    TMC_GSD = 5.0
    IIRS_GSD = 80.0
    LRO_NAC_GSD = 0.5
    SEED = 42
    SENSOR_GSD_MAP = {"OHRC": 0.25, "TMC": 5.0, "TMC-2": 5.0, "IIRS": 80.0, "LRO_NAC": 0.5}
    def get_seed() -> int:
        return SEED
    def get_sensor_gsd(s: str, f: float = 1.0) -> float:
        return SENSOR_GSD_MAP.get(str(s).upper(), f)
