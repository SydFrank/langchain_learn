from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from openai import OpenAI
from langchain.tools import tool
from langchain.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage
from langsmith import traceable
import json

MAX_ITERATIONS = 10
MODEL = "gpt-4o-mini"

# --- Tools (Langchain @tool decorator) ---

@tool
def get_product_price(product: str) -> float:
  """
  Look up the price of a product in the catalog.
  """
  print(f"     >> Executing get_product_price(product='{product}')")
  prices = {'laptop': 1299.99, "headphones": 149.94, "keyboard": 79.99}

  return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
  """
  APPLY a discounttier to a price and return the final price.
  Available tiers: bronze, silver, gold.
  """
  print(f"     >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
  discount_percentages = {'bronze': 0.1, 'silver': 0.2, 'gold': 0.3}
  discount = discount_percentages.get(discount_tier, 0)
  return round(price * (1 - discount), 2)


# ---Agent Loop ---
@traceable(name='Langchain Agent Loop')
def run_agent(question: str):
  tools = [get_product_price, apply_discount]
  tools_dict = {t.name: t for t in tools}
  llm = init_chat_model(model=MODEL, temperature=0)
  llm_with_tools = llm.bind_tools(tools)

  print(f"     >> Question: {question}")
  print("-" * 40)

  messages = [SystemMessage(
    content=
    "You are a helpful assistant that can use tools to answer questions."
    "Strict rules - you must follow these exactly:"
    "1.Never guess or assume any product price.\n"
    "You must call get_product_price first to get the real price.\n"
    "2.Only call apply_discount after you have reveived a price from get_product_price. Pass the exact price returned by get_product_price - do NOT pass a made-up number.\n"
    ),
    HumanMessage(content=question)
  ]

  for i in range(1,MAX_ITERATIONS + 1):
    print(f"     >> Iteration {i} of {MAX_ITERATIONS}")

    ai_message = llm_with_tools.invoke(messages)
    # print(json.dumps(ai_message.dict(), indent=2, ensure_ascii=False))
    tool_calls = ai_message.tool_calls

    # # if no tool calls, break
    if not tool_calls:
      print(f"\n Final Answer: {ai_message.content}")
      return ai_message.content

    # # process only the first tool call - force one tool per iteration
    tool_call = tool_calls[0]
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args", {})
    tool_call_id = tool_call.get("id")

    print(f"[Tool Call] {tool_name} with args: {tool_args}")

    tool_to_use = tools_dict.get(tool_name)

    print(f"{tool_to_use}")
    if not tool_to_use:
      raise ValueError(f"Tool {tool_name} not found in tools_dict")

    observation = tool_to_use.invoke(tool_args)

    print(f"[Observation] {observation}")


    messages.append(ai_message)
    messages.append(ToolMessage(content=str(observation), tool_call_id=tool_call_id))
    
  print("Error: Maximum number of iterations reached without finding a solution")
  return None
    



if __name__ == "__main__":
  print("Hello Langchain Agent (.bind_tools)!")
  # print("--------------------------------")
  result = run_agent("What is the price of a laptop after applying a gold discount?")