# Multi-Agent Travel Assistant with RAG, MCP and A2A Orchestration

This project is a multi-agent travel assistant demonstrating RAG, MCP tool integration and distributed agent orchestration across an A2A service boundary. A travel assistant is provided to answer questions related to travel and tourism information sourced from [Travel Wiki](https://en.wikivoyage.org/wiki/Main_Page) and from a proprietary SQL database created specifically for this project.
The travel assistant is based on the book [AI Agents and Applications](https://www.manning.com/books/ai-agents-and-applications) by Roberto Infante (Manning Publications), with several modifications to make the project more interesting and challenging.

The main changes are:
- The ability to choose between OpenAI embeddings and the locally hosted **BAAI/bge-base-en-v1.5** embedding model from [Hugging Face](https://huggingface.co/). The latter was chosen because it is free to run locally.
- One of the agents was deployed as a web service, requiring agent orchestration to cross an A2A boundary.

## **General Architecture**
The architecture of the system supporting the travel assistant is as follows:

```mermaid
graph TD;
    A{💂 Guardrail}==>|❌ Query out of scope| B[End]
    A ==> |✅ Query within scope| D[➡ Router]
    D -.->|A2A for travel info| E[Remote travel Info/Weather agent]
    D ==> |Accommodation info| I[Local accommodation agent]
    E ==> |Travel info| F[🛠 RAG tool]
    F ==> |Retrieve touristic info| L[👨‍🔧 RAG]
    L ==> |Docs of touristic info| M[📚 Docs Vectorial DB]
    E ==> |Weather forecast| G[MCP Client]
    E ==> |Search completed| B 
    G -.->|MCP| H[⛅ weather forecast service]
    I ==> |Search for accommodations| J[🛠 SQL tools]
    J -.->|In-process access| K[🛎 Accommodations SQL DB]
    I ==> |Search completed| B
```

A guardrail is used as a domain relevance prefilter that screens the user’s message before any agent reasoning or routing happens. If the question is deemed as out of scope- non–travel-related, e.g. sports, or not focused on UK tourist destinations Cornwall (England)-, the system intercepts it early and politely refuses to answer. In this travel assistant, the policy defines “in scope” as travel information such as destinations, attractions, lodging, prices/availability or weather in Cornwall/England.

With the query classified as in scope, a router decides whether the questions concerns
- Travel information including weather forecast or
- Accommodation booking.
Each of these branches has its own agent.

### **Travel information agent**

Travel information agent runs remotely on its own process. It is contacted via A2A protocol.
The travel information agent searches a RAG-backed vector store for tourist information, calls a weather tool/API to retrieve weather conditions and synthesizes a final response. This is accomplished by using two tools:
- A RAG information retriever to reply to user's query
- A client to remotely access a weather forecast service exposed as a MCP server

A RAG system stores document chunks and their embeddings in a vector store, ChromaDB.
Given a user question, an LLM generates N alternative versions of that question (multiple perspectives) to improve retrieval coverage and overcome limitations of pure distance-based similarity search. Each of the N queries is embedded and used to retrieve the top M chunks from the vector store, producing a total of N * M relevant document chunks.
The system then merges these ranked lists using Reciprocal Rank Fusion, scoring each chunk by its rank in each list (1/(rank + k)) and summing scores across lists, then reranking by the cumulative score.
Finally, only the top reranked chunks are passed as context to the LLM, together with the original user question, for final answer synthesis.

### **Weather Forecast Service**

A weather forecast service was mocked as a MCP server.

### **Accommodation booking agent**

The accommodation booking agent checks lodging availability and prices in Cornwall by calling SQL tools for retrieving data from a SQL SQLite database named cornwall_hotels.db. 
The agent doesn’t query the DB directly; instead it uses LangChain’s SQL Database toolkit (SQLDatabase + SQLDatabaseToolkit) to expose the database as a set of agent tools.

### Database choice

SQLAlchemy provides a database-agnostic abstraction layer for Python, similar in role to JDBC in Java. It delegates execution to DB-specific drivers such as sqlite3 or psycopg.
In this project, a local in-process SQLite database is used for simplicity

&emsp;&emsp;DB_URI = "sqlite:///./SQL_DB/cornwall_hotels.db"
&emsp;&emsp;hotel_db = SQLDatabase.from_uri(DB_URI)

but to use a production-level database, e.g. PostgreSQL, it would just be a matter of updating environment variable DB_URI to something as

&emsp;&emsp;DB_URI = "postgresql+psycopg://myuser:mypassword@db.example.com:5432/hotel_db?sslmode=require" 


## **Installation**
For installation run the following steps:
1. Install Python 3.11
2. Install [uv](https://docs.astral.sh/uv/) to manage environments
3. Git clone this repository
4. In the local root directory of cloned repository, execute
   cmd> uv venv 
   to create a new Python environment
5. Activate it (Windows version)
   cmd> .\.venv\Scripts\activate.bat
6. Install project dependencies
   cmd> uv pip install -r requirements.txt

## **Configure the Travel Assistant**
The configuration of the travel assistant is managed through an .env file at the project root. A template is provided in [`.env.example`](.env.example).
The `OPENAI_API_KEY` and `HUGGING_FACES_API_KEY` variables defined in this file can be overridden by environment variables with the same names.

## **Run Travel Assistant**
### **Run Weather Forecast MCP Server**
In a window with travelAgent Python environment activated, run

&emsp;&emsp;(travelAgent) cmd> cd src

&emsp;&emsp;(travelAgent) cmd> python -m mcp_weather.weatherForecast

### **Run Travel Info agent**
In a window with travelAgent Python environment activated, run

&emsp;&emsp;(travelAgent) cmd> cd src

&emsp;&emsp;(travelAgent) cmd> python -m travel_info_agent.agent a2a

### **Run Travel Assistant agent, the chatbot**
In a window with travelAgent Python environment activated, run

&emsp;&emsp;(travelAgent) cmd> cd src

&emsp;&emsp;(travelAgent) cmd> python -m orchestrator.agent

## **Run Tests**
To run automatic tests, open a window with travelAgent Python environment activated and proceed as follows:

&emsp;&emsp;(travelAgent) cmd> pytest tests -s
