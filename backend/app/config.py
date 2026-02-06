from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://capex:capex_dev_password@localhost:5432/ai_capex"
    database_url_sync: str = "postgresql://capex:capex_dev_password@localhost:5432/ai_capex"

    eia_api_key: str = ""
    cloudflare_api_token: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"

    alert_webhook_url: str = ""
    alert_email_to: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""

    class Config:
        env_file = ".env"


settings = Settings()
