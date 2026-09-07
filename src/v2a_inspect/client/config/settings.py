from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    server_host: str = "localhost"
    server_port: int = 8080
    timeout: float = 300.0
    max_retries: int = 3
    retry_backoff_seconds: float = 10.0

    @property
    def server_url(self) -> str:
        return f"http://{self.server_host}:{self.server_port}"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="V2A_INSPECT_CLIENT_",
        extra="ignore",
    )


settings = Settings()
