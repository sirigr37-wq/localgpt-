from datetime import datetime
from typing import Dict, Any
from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    status: str
    version: str
    environment: str
    database_connected: bool
    timestamp: datetime
    services: Dict[str, Any]
