from pydantic import BaseModel, Field

class MethodInfo(BaseModel):
    problem: str = Field(description="The problem addressed, one sentence")
    method: str = Field(description="The proposed approach, 1-2 sentences")
    quote: str = Field(description="One sentence copied exactly from the text describing the method")

class Metric(BaseModel):
    name: str
    value: str
    dataset: str

class ResultsInfo(BaseModel):
    datasets: list[str]
    metrics: list[Metric]
    quote: str = Field(description="One sentence copied exactly from the text reporting results")

class LimitInfo(BaseModel):
    limitations: list[str]
    future_work: list[str]
    quote: str = Field(description="One sentence copied exactly from the text about limitations or future work")

class ClusterLabel(BaseModel):
    name: str

class GapExplanation(BaseModel):
    explanation: str
    research_question: str

class Queries(BaseModel):
    queries: list[str] = Field(description="List of academic search query strings")

class RelevanceDecision(BaseModel):
    relevant: bool = Field(description="True if the paper is relevant to the topic, False otherwise")
    reason: str = Field(description="One sentence explaining relevance or lack thereof")
