from tabnanny import verbose
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import SerperDevTool
from pydantic import BaseModel, Field

class ProductReview(BaseModel):
    product_name: str = Field(..., description="The name of the product")
    reviews: List[str] = Field(..., description="The reviews of the product")

class ProductReviewList(BaseModel):
    product_list: List[ProductReview] = Field(..., description="The list of products and their reviews")

class ProductReviewSummary(BaseModel):
    product_name: str = Field(..., description="The name of the product")
    summary: str = Field(..., description="The summary of the product reviews")
    pros: List[str] = Field(..., description="The pros of the product reviews")
    cons: List[str] = Field(..., description="The cons of the product reviews")

class ProductReviewRecommendation(BaseModel):
    product_name: str = Field(..., description="The name of the product")
    summary: str = Field(..., description="The summary of the product reviews")
    recommendation: bool = Field(..., description="Whether the product is worth buying")
    reason: str = Field(..., description="The reason for the recommendation")

@CrewBase
class ProductReviewer():
    """Product Reviewer crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def product_review_searcher(self) -> Agent:
        return Agent(
            config=self.agents_config["product_review_searcher"], 
            verbose=True, 
            tools=[SerperDevTool()]
        )

    @agent
    def sentiment_analyser(self) -> Agent:
        return Agent(
            config=self.agents_config["sentiment_analyser"],
            verbose=True
        )
    
    @agent
    def advisor_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["advisor_agent"],
            verbose=True
        )

    @task
    def search_product_reviews(self) -> Task:
        return Task(config=self.tasks_config['search_product_reviews'], verbose=True, output_pydantic=ProductReviewList)

    @task
    def analyze_product_reviews(self) -> Task:
        return Task(config=self.tasks_config['analyze_product_reviews'], verbose=True, output_pydantic=ProductReviewSummary)

    @task
    def provide_recommendation(self) -> Task:
        return Task(config=self.tasks_config['provide_recommendation'], verbose=True, output_pydantic=ProductReviewRecommendation,)

    @crew
    def crew(self) -> Crew:
        """Creates the ProductReviewer crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
