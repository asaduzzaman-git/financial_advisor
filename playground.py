from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
from phi.playground import Playground, serve_playground_app
from phi.storage.agent.sqlite import SqlAgentStorage


finance_agent=Agent(
    name="Finance Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True, 
        company_info=True, 
        company_news=True
    )],
    #show_tool_calls=True,
    markdown=True,
    instructions=["Use tables to display data"],
    add_chat_history_to_messages=True,
    storage=SqlAgentStorage(table_name="finance_agent", db_file="agents.db"),
)

web_agent=Agent(
    name="Web Researcher",
    model=Groq(
        id="llama-3.3-70b-versatile"
    ),
    tools=[DuckDuckGo()],
    #show_tool_calls=True,
    markdown=True,
    add_history_to_messages=True,
    instructions=["Always include sources of the information that you gather"],
    storage=SqlAgentStorage(table_name="web_agent", db_file="agents.db"),  
)

app = Playground(agents=[finance_agent, web_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground:app", reload=True)