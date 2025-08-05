import os
from datetime import datetime
from typing import Dict

import gradio as gr
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# ✅ Tools
class TavilySearchTool(BaseTool):
    name: str = "Tavily Search"
    description: str = "Search the internet for information using Tavily"

    def _run(self, query: str) -> str:
        return f"Search results for: {query}"


class GeminiResearchTool(BaseTool):
    name: str = "Gemini Research"
    description: str = "Conduct detailed analysis using Gemini"

    def _run(self, query: str) -> str:
        try:
            model = genai.GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(query)
            return response.text if hasattr(response, "text") else str(response)
        except Exception as e:
            return f"Error in Gemini Research: {e}"


# ✅ Crew Setup
class EVMarketResearchCrew:
    def __init__(self, topic: str):
        self.topic = topic
        self.tools = {
            "tavily": TavilySearchTool(),
            "gemini": GeminiResearchTool()
        }

        self.agents = self._create_agents()
        self.tasks = self._create_tasks()
        self.crew = self._create_crew()

    def _create_agents(self) -> Dict[str, Agent]:
        return {
            "market_research": Agent(
                role="Market Research Specialist",
                goal="Gather comprehensive market data about the EV industry",
                backstory="You are an expert in market trends and competitive landscapes in the EV sector.",
                tools=[self.tools["tavily"], self.tools["gemini"]],
                verbose=True,
                allow_delegation=True,
            ),
            "tech_analysis": Agent(
                role="Technology Analyst",
                goal="Analyze technological developments in the EV space",
                backstory="You specialize in EV battery tech, infrastructure, and innovations.",
                tools=[self.tools["gemini"]],
                verbose=True,
                allow_delegation=True,
            ),
            "strategy": Agent(
                role="Strategic Insights Analyst",
                goal="Synthesize research into actionable market insights",
                backstory="You identify opportunities, risks, and strategic moves in the EV industry.",
                tools=[self.tools["gemini"]],
                verbose=True,
                allow_delegation=True,
            ),
        }

    def _create_tasks(self) -> Dict[str, Task]:
        market_research_task = Task(
            description=f"""Conduct comprehensive market research on the EV industry focused on: {self.topic}.
            - Market size, growth projections
            - Key players, market share
            - Regional dynamics, consumer trends
            - Regulations, incentives, pricing, supply chain
            Provide structured report with data.""",
            expected_output="Detailed market research report",
            agent=self.agents["market_research"],
        )

        tech_analysis_task = Task(
            description=f"""Analyze EV tech trends related to: {self.topic}.
            - Battery roadmap, charging standards
            - Manufacturing innovations
            - Tech breakthroughs and performance metrics
            - Sustainability and future projections""",
            expected_output="Technical analysis report",
            agent=self.agents["tech_analysis"],
            context=[market_research_task],
        )

        strategy_task = Task(
            description=f"""Create strategic insights around {self.topic}:
            - Market opportunities, risks, differentiators
            - Regional growth suggestions
            - Partnerships, KPIs, investment timelines
            - Actionable recommendations""",
            expected_output="Strategic recommendations report",
            agent=self.agents["strategy"],
            context=[market_research_task, tech_analysis_task],
        )

        return {
            "market_research": market_research_task,
            "tech_analysis": tech_analysis_task,
            "strategy": strategy_task,
        }

    def _create_crew(self) -> Crew:
        return Crew(
            agents=list(self.agents.values()),
            tasks=list(self.tasks.values()),
            verbose=True,
            process=Process.sequential,
        )

    def run_research(self) -> Dict[str, str]:
        try:
            print("\n🚀 Starting EV Market Research Process...")
            self.crew.kickoff()
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return {
                "timestamp": timestamp,
                "market_research": self.tasks["market_research"].output.raw,
                "tech_analysis": self.tasks["tech_analysis"].output.raw,
                "strategy": self.tasks["strategy"].output.raw,
            }
        except Exception as e:
            raise RuntimeError(f"Error during research process: {e}")


# ✅ Gradio UI wrapper
def run_market_research_ui(topic: str):
    if not topic.strip():
        return "Please enter a valid topic.", "", ""

    research = EVMarketResearchCrew(topic=topic)
    results = research.run_research()
    return (
        results["market_research"],
        results["tech_analysis"],
        results["strategy"],
    )


# ✅ Launch UI
if __name__ == "__main__":
    print("🚀 Launching EV Research UI...")

    gr.Interface(
        fn=run_market_research_ui,
        inputs=gr.Textbox(
            label="🔍 EV Topic",
            placeholder="e.g. EV Growth in India, Tesla Battery Tech, EV Infrastructure 2030...",
            lines=2,
        ),
        outputs=[
            gr.Textbox(label="📊 Market Research Report"),
            gr.Textbox(label="🔬 Technology Analysis"),
            gr.Textbox(label="📈 Strategic Recommendations"),
        ],
        title="⚡ EV Market Research Assistant",
        description="🧠 Powered by CrewAI + Gemini 1.5 Pro. Enter a topic to get a multi-agent research report on the EV industry.",
        theme="default",
    ).launch()
