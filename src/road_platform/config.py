from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
class Settings(BaseSettings):
    app_env: str = 'development'
    database_url: str = 'postgresql+psycopg://roads:roads@localhost:5432/roads'
    data_path: Path = ROOT / 'data/processed/roads.csv'
    model_dir: Path = ROOT / 'models'
    yolo_weights: str | None = None
    api_url: str = 'http://localhost:8000'
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
settings = Settings()
settings.model_dir.mkdir(parents=True, exist_ok=True)
