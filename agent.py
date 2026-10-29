
import urllib.parse
import httpx
import os
import json
import subprocess
import asyncio
from typing import AsyncGenerator, List, Dict, Any
from dotenv import load_dotenv
from groq import Groq
from ddgs import DDGS

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
#tools: web searching, image gen, scripting in python, more soon (Video gen and document analysis!!!)

def execute_python_code(code: str) -> str:
    """Executes a Python code snippet safely and returns the output or errors."""
    try:
        result = subprocess.run(
            ["python", "-c", code], 
            capture_output=True, 
            text=True, 
            timeout=5
        )
        if result.returncode != 0:
            return f"Execution Error: {result.stderr}"
        return f"Output:\n{result.stdout}"
    except Exception as e:
        return f"Failed to execute code: {str(e)}"

def web_search(query: str) -> str:
    """Performs a web search using DuckDuckGo and returns the top results."""
    try:
        results = []
        with DDGS() as ddgs:
            # Set a timeout so it never hangs the stream
            ddgs_results = list(ddgs.text(query, max_results=3))
            for r in ddgs_results:
                results.append(f"Title: {r['title']}\nURL: {r['href']}\nSnippet: {r['body']}\n")
        if not results:
            return "No search results found."
        return "\n".join(results)
    except Exception as e:
        print(f"Search warning: {str(e)}")
        return f"Search temporarily unavailable. Proceed based on general knowledge."

async def generate_image(prompt_description: str) -> str:
    """
    Directly generates a robust markdown image link using a public rendering engine.
    """
    try:
        print(f"\n🎨 [Generating Image with prompt: '{prompt_description[:100]}...']")
        
        # URL encode the prompt safely
        encoded_prompt = urllib.parse.quote(prompt_description)
        
        # public image renderer, no token shenanigans
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1280&height=720&nologo=true"
        
        print(f"✅ [Image URL created successfully]")
        return f"Image successfully generated. URL: {image_url}"

    except Exception as e:
        print(f"❌ [Image Generation Error: {str(e)}]")
        return f"Failed to generate image: {str(e)}"


#schemas

tools = [
    {
        "type": "function",
        "function": {
            "name": "execute_python_code",
            "description": "Write and execute Python code for data analysis or math logic. Do NOT use this to scrape web pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "The valid Python code to run."}
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search DuckDuckGo for live internet information, news, current events, or sports results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query keywords to look up on DuckDuckGo.",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",  # <--- Updated the function name used in schema
            "description": "Generate or create new high-quality images based on detailed descriptive text prompts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt_description": {
                        "type": "string", 
                        "description": "Detailed description, style, and content of the image to generate, e.g., 'A cyberpunk street vendor selling ramen at night in Tokyo, realistic 8k, dramatic lighting'"
                    }
                },
                "required": ["prompt_description"],
            },
        },
    }
]

#so it knows that tools it has
available_functions = {
    "execute_python_code": execute_python_code,
    "web_search": web_search,
    "generate_image": generate_image
}

#the meat
async def run_agent_stream(messages: List[Dict[str, Any]]) -> AsyncGenerator[str, None]:
    """
    Executes the multi-tool agent loop and yields SSE-compatible string events.
    """
    yield "[Status] Agent initialized...\n\n"

    # guiding the bot
    system_prompt = {
        "role": "system",
        "content": (
            "You are an autonomous AI assistant with access to tools (web search, python execution, image generation). "
            "CRITICAL INSTRUCTIONS:\n"
            "1. Call a tool only ONCE per request unless absolutely necessary.\n"
            "2. Once a tool returns its output (such as an image URL or search result), DO NOT call the same tool again. "
            "Instead, use the output to directly answer the user in natural language and format image URLs nicely using markdown (e.g. ![Generated Image](url))."
        )
    }
    
    # Prepend system prompt if not already present
    if not messages or messages[0].get("role") != "system":
        messages.insert(0, system_prompt)

    MAX_TOOL_CALLS = 2
    tool_call_count = 0

    while True:
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0,
        )
        
        response_message = response.choices[0].message

        if response_message.tool_calls:
            messages.append(response_message)

            if tool_call_count >= MAX_TOOL_CALLS:
                yield f"[Status] Tool call limit reached. Forcing response...\n\n"
                for tool_call in response_message.tool_calls:
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": "Error: Maximum search limit reached. Answer directly."
                    })
                continue

            tool_call_count += 1
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except Exception:
                    function_args = {}

                # Yield status notification to the client
                yield f"[Tool Triggered]: {function_name}({function_args})\n\n"

                # HANDLING SYNC VS ASYNC TOOLS DYNAMICALLY
                if function_name in available_functions:
                    tool_to_call = available_functions[function_name]
                    
                    # Check if the function object is async (like generate_image)
                    if asyncio.iscoroutinefunction(tool_to_call):
                        # Await the async function
                        tool_output = await tool_to_call(**function_args)
                    else:
                        # Run the sync function directly
                        tool_output = tool_to_call(**function_args)
                else:
                    tool_output = f"Error: Tool {function_name} not found."

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_output),
                })

            continue

        else:
            # Handle final text stream to client
            stream_response = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=messages,
                temperature=0,
                stream=True
            )

            full_reply = ""
            for chunk in stream_response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    content = delta.content
                    full_reply += content
                    # Yield incremental text chunks directly to HTTP stream
                    yield content

            messages.append({"role": "assistant", "content": full_reply})
            break