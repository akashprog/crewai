from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from pydantic import BaseModel, Field
from crewai_tools import SerperDevTool
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

class TrendingCompany(BaseModel):
    name: str = Field(description="The name of the trending company")
    ticker: str = Field(description="The ticker symbol of the trending company")
    reason: str = Field(description="Reason why this company is trending in news")

class TrendingCompanyList(BaseModel):
    companies: List[TrendingCompany] = Field(description="The trending companies in the news")

class TrendingCompanyResearch(BaseModel):
    name: str = Field(description="The name of the trending company")
    ticker: str = Field(description="The ticker symbol of the trending company")
    market_position: str = Field(description="Current market position and competitive analysis")
    future_outlook: str = Field(description="Future outlook and growth prospects")
    investment_potential: str = Field(description="Investment potential and suitability for investment")

class TrendingCompanyResearchList(BaseModel):
    companies: List[TrendingCompanyResearch] = Field(description="The research reports for each trending company")

@CrewBase
class StockPicker():
    """StockPicker crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def trending_company_finder(self) -> Agent:
        return Agent(config=self.agents_config['trending_company_finder'], verbose=True, tools=[SerperDevTool()])

    @agent
    def financial_researcher(self) -> Agent:
        return Agent(config=self.agents_config['financial_researcher'], verbose=True, tools=[SerperDevTool()])

    @agent
    def stock_picker(self) -> Agent:
        return Agent(config=self.agents_config['stock_picker'], verbose=True)

    @task
    def find_trending_companies(self) -> Task:
        return Task(config=self.tasks_config['find_trending_companies'], output_pydantic=TrendingCompanyList)

    @task
    def research_trending_companies(self) -> Task:
        return Task(config=self.tasks_config['research_trending_companies'],output_pydantic=TrendingCompanyResearchList)

    @task
    def pick_best_company(self) -> Task:
        return Task(config=self.tasks_config['pick_best_company'])

    @crew
    def crew(self) -> Crew:
        """Creates the StockPicker crew"""
        
        manager = Agent(
            config=self.agents_config['manager'],
            allow_delegation=True
            )

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            verbose=True,
            manager_agent=manager
        )
