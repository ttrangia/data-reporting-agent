from agent.state import AgentState

MAX_RETRIES = 2


def classify_intent(state: AgentState) -> dict:
    # Stub: always route to data path for now
    return {"intent": "data"}


def generate_sql(state: AgentState) -> dict:
    # Stub: pretend we generated SQL
    return {"sql": "SELECT 1 AS stub_result", "sql_error": None}


def validate_sql(state: AgentState) -> dict:
    # Stub: always passes for now
    return {"sql": state["sql"], "sql_error": None}


def execute_sql(state: AgentState) -> dict:
    # Stub: fake a result row
    return {"rows": [{"stub_result": 1}], "sql_error": None}


def summarize(state: AgentState) -> dict:
    rows = state.get("rows") or []
    return {"summary": f"Stub response. Got {len(rows)} row(s) from stub query."}


def handle_chat(state: AgentState) -> dict:
    return {"summary": "Stub chat response."}