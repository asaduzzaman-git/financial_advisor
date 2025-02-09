from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo



financial_analyst=Agent(
    name="financial Agent",
    model=Groq(
        id="llama-3.3-70b-versatile"
    ),
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True
    )],
    show_tool_calls=True,
    markdown=True,
    instructions=["Always create tables for comparisons"],
)

web_researcher=Agent(
    name="Web Researcher",
    model=Groq(
        id="llama-3.3-70b-versatile"
    ),
    tools=[DuckDuckGo()],
    show_tool_calls=True,
    markdown=True,
    instructions=["Always include sources of the information that you gather"],  
)

agent_team=Agent(
    team=[financial_analyst, web_researcher],
    model=Groq(
        id="llama-3.3-70b-versatile"
    ),
    show_tool_calls=True,
    markdown=True,
    instructions=["Always include sources of the information that you gather", "Always create tables for comparisons"],
    debug_mode=True  
)


agent_team.print_response("summerise the analyst recommendation and share the latest information about Nvidia?")