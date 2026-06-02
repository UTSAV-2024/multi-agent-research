from pydantic import BaseModel


from pydantic import BaseModel, Field


class ResearchRequest(BaseModel):

    topic: str = Field(
        ...,
        min_length=1,
        description="Research topic"
    )


class ResearchResponse(BaseModel):

    topic: str

    status: str

    report: str

    source_count: int

    timestamp: str

    metrics: dict

    class Config:
        from_attributes = True