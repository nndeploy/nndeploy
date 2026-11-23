"""Optional MCP transport and explicit per-entry export configuration."""
from pydantic import BaseModel, ConfigDict, Field, field_validator


class MCPConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    enabled: bool = False
    path: str = '/mcp'
    allowed_hosts: list[str] = Field(default_factory=lambda: ['127.0.0.1:*', 'localhost:*', '[::1]:*'])
    allowed_origins: list[str] = Field(default_factory=lambda: ['http://127.0.0.1:*', 'http://localhost:*', 'http://[::1]:*'])

    @field_validator('path')
    @classmethod
    def valid_path(cls, value):
        import re
        if not re.fullmatch(r'/[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*', value):
            raise ValueError('MCP path must be an absolute non-root path without a trailing slash')
        if value.split('/')[1] in ('api', 'v1', 'health', 'metrics'):
            raise ValueError('MCP path conflicts with Server routes')
        return value


class ToolConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    name: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_.-]+$')
    description: str = Field(min_length=1, max_length=4096)

    @field_validator('name', 'description', mode='before')
    @classmethod
    def trim(cls, value):
        return value.strip() if isinstance(value, str) else value

    @field_validator('name')
    @classmethod
    def reserved(cls, value):
        if value in ('get_task', 'cancel_task'):
            raise ValueError('MCP tool name is reserved')
        return value
