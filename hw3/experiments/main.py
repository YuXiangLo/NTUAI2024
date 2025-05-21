from __future__ import annotations
from autogen import ConversableAgent, register_function
import os, sys, re, ast
from typing import Dict, List, get_type_hints

SCORE_KEYWORDS: dict[int, list[str]] = {
    1: ["awful", "horrible", "disgusting"],
    2: ["bad", "unpleasant", "offensive"],
    3: ["average", "uninspiring", "forgettable"],
    4: ["good", "enjoyable", "satisfying"],
    5: ["awesome", "incredible", "amazing"]
}

# ────────────────────────────────────────────────────────────────
# 0. OpenAI API key setup ── *Do **not** modify this block.*
# ────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    sys.exit("❗ Set the OPENAI_API_KEY environment variable first.")
LLM_CFG = {"config_list": [{"model": "gpt-4o-mini", "api_key": OPENAI_API_KEY}]}

# ────────────────────────────────────────────────────────────────
# 1. Utility data structures & helper functions
# ────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()

def fetch_restaurant_data(restaurant_name: str) -> dict[str, list[str]]:
    data = {}
    target = normalize(restaurant_name)
    print("Restaurant name: " + str(target))
    with open(DATA_PATH, encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            name, review = line.split('.', 1)
            if normalize(name) == target:
                data.setdefault(name.strip(), []).append(review.strip())
    return data

def calculate_overall_score(food_scores: List[int], customer_service_scores: List[int]):
    """Geometric-mean rating rounded to 3 dp."""
    n = len(food_scores)
    if n == 0 or n != len(customer_service_scores):
        raise ValueError("food_scores and customer_service_scores must be non-empty and same length")
    total = sum(((f**2 * s)**0.5) * (1 / (n * (125**0.5))) * 10 for f, s in zip(food_scores, customer_service_scores))
    return f"{total:.3f}"

# register functions
fetch_restaurant_data.__annotations__ = get_type_hints(fetch_restaurant_data)
calculate_overall_score.__annotations__ = get_type_hints(calculate_overall_score)

# ──────────────────────────────────────────────
# 2. Agent setup
# ──────────────────────────────────────────────

def build_agent(name, msg):
    return ConversableAgent(name=name, system_message=msg, llm_config=LLM_CFG)

SCORE_KEYWORDS_STR = str(SCORE_KEYWORDS)

# 'Return JSON {"call":"fetch_restaurant_data","args":{"restaurant_name":"<name>"}}'
DATA_FETCH = build_agent(
    "fetch_agent",
    '''
    You are the Data Fetch Agent responsible for retrieving reviews for a specific restaurant.
    You have to extract the restaurant name from the user query.
    Then, suggest the extracted restaurant name as argument to the function `fetch_restaurant_data`, that will be executed by the entrypoint agent.

    Steps:
    1. Extract the restaurant name from the user query.
    2. Call the function `fetch_restaurant_data(restaurant_name)` to fetch all the reviews of the restaurant.
    3. Return ALL the reviews fetched by the function `fetch_restaurant_data`, don't miss any review.

    Example:
    - User query: "How good is the food at Subway?"
    - Output: fetch_restaurant_data("Subway")
    '''
)

ANALYZER = build_agent(
    "review_analyzer_agent",
    '''
    You are the Analyzer Agent responsible for analyzing customer reviews. 
    Your task is to extract **one and only one** score pair for **each input review**, with NO EXCEPTIONS.

    ==== SCORING SYSTEM ====
    Each score pair MUST include:
    - a `food_score` (integer from 1-5)
    - a `customer_service_score` (integer from 1-5)

    **Keyword Reference:**
    Score 1: Awful, horrible, disgusting  
    Score 2: Bad, unpleasant, offensive  
    Score 3: Average, uninspiring, forgettable, meh
    Score 4: Good, enjoyable, satisfying  
    Score 5: Awesome, incredible, amazing

    ==== PROCESSING INSTRUCTIONS ====

    - You will receive input in format: {'Restaurant Name': ['Review 1', 'Review 2', ...]}
    - For each review in the list, you repeat it first, then tell the key described word of food and service respectively and return the food score and custom service score in such format: [food_score: X, customer_service_score: Y]

    **NOTICE** that meh, uninspiring, and forgettable are neutral descriptions.
    '''
)

ENTRY = build_agent("entry", 
    "You are the Entry Point Agent"
)

# register functions
register_function(
    fetch_restaurant_data,
    caller=DATA_FETCH,
    executor=ENTRY,
    name="fetch_restaurant_data",
    description="Fetch reviews from specified data file by name.",
)

# ────────────────────────────────────────────────────────────────
# 3. Conversation helpers
# ────────────────────────────────────────────────────────────────

def run_chat_sequence(entry: ConversableAgent, sequence: list[dict]) -> str:
    ctx = {**getattr(entry, "_initiate_chats_ctx", {})}
    for step in sequence:
        msg = step["message"].format(**ctx)
        chat = entry.initiate_chat(
            step["recipient"], message=msg,
            summary_method=step.get("summary_method", "last_msg"),
            max_turns=step.get("max_turns", 2),
        )
        out = chat.summary
        # Data fetch output
        if step["recipient"] is DATA_FETCH:
            for past in reversed(chat.chat_history):
                try:
                    data = ast.literal_eval(past["content"])
                    if isinstance(data, dict) and data and not ("call" in data):
                        ctx.update({"reviews_dict": data, "restaurant_name": next(iter(data))})
                        print("length of data: " + str(len(ctx["reviews_dict"][ctx["restaurant_name"]])))
                        break
                except:
                    continue
        # Analyzer output passed directly
        elif step["recipient"] is ANALYZER:
            ctx["analyzer_output"] = out
    return out

ConversableAgent.initiate_chats = lambda self, seq: run_chat_sequence(self, seq)

# ──────────────────────────────────────────────
# 4. Main entry
# ──────────────────────────────────────────────

def main(user_query: str, data_path: str = "restaurant-data.txt"):
    global DATA_PATH
    DATA_PATH = data_path
    agents = {"data_fetch": DATA_FETCH, "analyzer": ANALYZER}
    chat_sequence = [
        {"recipient": agents["data_fetch"], 
         "message": "Find reviews for this query: {user_query}", 
         "summary_method": "last_msg", 
         "max_turns": 2},

        {"recipient": agents["analyzer"], 
         "message": "Here are the reviews from the data fetch agent:\n{reviews_dict}\n\nExtract food and service scores for each review.", 
         "summary_method": "last_msg", 
         "max_turns": 1},
    ]
    ENTRY._initiate_chats_ctx = {"user_query": user_query}
    result = ENTRY.initiate_chats(chat_sequence)

    # print(result, file=sys.stderr)
    pattern = r'\[food_score:\s*(\d+),\s*customer_service_score:\s*(\d+)\]'
    matches = re.findall(pattern, result)

    food_scores = [int(food) for food, _ in matches]
    customer_service_scores = [int(service) for _, service in matches]

    return calculate_overall_score(food_scores, customer_service_scores)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python main.py path/to/data.txt "How good is Subway?" ')
        sys.exit(1)

    path = sys.argv[1]
    query = sys.argv[2]
    main(query, path)
