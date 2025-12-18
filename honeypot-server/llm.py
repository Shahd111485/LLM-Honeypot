import os
import json
import time
import yaml
import asyncio
import re
import logging
from collections import OrderedDict
from openai import AsyncOpenAI

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("LLM_Honeypot")

# =============================
#    Resilience Components
# =============================

class CircuitBreaker:
    def __init__(self, fail_threshold=3, reset_time=30):
        self.fail_threshold = fail_threshold
        self.reset_time = reset_time
        self.fail_count = 0
        self.last_fail_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    def allow_request(self):
        if self.state == "OPEN":
            if time.time() - self.last_fail_time > self.reset_time:
                self.state = "HALF_OPEN"
                logger.info("Circuit Breaker entering HALF_OPEN state.")
                return True
            return False
        return True

    def record_success(self):
        if self.state != "CLOSED":
            logger.info("Circuit Breaker recovered. Resetting to CLOSED.")
        self.fail_count = 0
        self.state = "CLOSED"

    def record_failure(self):
        self.fail_count += 1
        self.last_fail_time = time.time()
        logger.warning(f"Failure detected. Count: {self.fail_count}/{self.fail_threshold}")
        
        if self.fail_count >= self.fail_threshold:
            self.state = "OPEN"
            logger.error(f"Circuit Breaker TRIPPED. Pausing for {self.reset_time}s.")

class TTLCache:
    def __init__(self, ttl_seconds=60, max_size=100):
        self.cache = OrderedDict()
        self.ttl = ttl_seconds
        self.max_size = max_size

    def get(self, key):
        if key not in self.cache:
            return None
        
        value, timestamp = self.cache[key]
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            return None
        
        self.cache.move_to_end(key)  # Mark as recently used
        return value

    def set(self, key, value):
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)  # Remove oldest
        self.cache[key] = (value, time.time())

# =============================
#     Helper / Loading Code
# =============================

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_system_prompt():
    path = os.path.join(MODULE_DIR, 'personalitySSH.yml')
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                return data.get('personality', {}).get('prompt', "You are a Linux server.")
    except Exception as e:
        logger.error(f"Failed to load personality: {e}")
    return "You are a Linux server. Reply only with terminal output."

def load_default_examples():
    path = os.path.join(MODULE_DIR, "fewshots.json")
    fallback = [{"command": "whoami", "response": "root"}]
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return fallback

DEFAULT_FEW_SHOT_EXAMPLES = load_default_examples()

def build_few_shot_prompt(system_prompt, examples, user_input):
    parts = [system_prompt.strip(), ""]
    for ex in examples:
        parts.append(f"Input: {ex.get('command')}\nOutput: {ex.get('response')}")
    parts.append("### Task")
    parts.append(f"Input: {user_input.strip()}")
    return "\n".join(parts)

def select_dynamic_examples(command, examples, k=5):
    """
    Select k examples closest to the input command using simple token overlap.
    Fast, no embeddings, perfect for SSH commands.
    """
    command_tokens = set(command.lower().split())
    scored = []

    for ex in examples:
        ex_cmd = ex.get("command", "").lower()
        ex_tokens = set(ex_cmd.split())


        score = len(command_tokens & ex_tokens)
        if score > 0:
            scored.append((score, ex))

    
    scored.sort(key=lambda x: x[0], reverse=True)

    return [ex for _, ex in scored[:k]]


# =============================
#           LLM Class
# =============================

class LLM:
    def __init__(self, api_key=None, api_model=None, max_examples=None, max_retries=3):
        self.api_model = api_model or os.getenv("MODEL_NAME") or "gemini-2.0-flash"
        self.examples = DEFAULT_FEW_SHOT_EXAMPLES[:max_examples] if max_examples else DEFAULT_FEW_SHOT_EXAMPLES
        self.max_retries = max_retries
        self.system_prompt = load_system_prompt()

        key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
        if not key:
            raise ValueError("API Key missing. Set GEMINI_API_KEY in .env")

        # Initialize Async Client
        self.client = AsyncOpenAI(
            api_key=key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )

        # Resilience Layers
        self.circuit = CircuitBreaker(fail_threshold=3, reset_time=20)
        self.cache = TTLCache(ttl_seconds=300, max_size=200)

        logger.info(f"LLM initialized. Model: {self.api_model}")

    def _sanitize(self, text: str) -> str:
        """
        Removes markdown code blocks, bolding, and keeps output looking like raw terminal text.
        """
        if not text: return ""
        text = str(text)
        
        # 1. Remove code block markers (```bash, ```, etc)
        text = re.sub(r'```[a-zA-Z]*', '', text)
        
        # 2. Remove inline code backticks
        text = text.replace('`', '')
        
        # 3. Remove Markdown bold/italic
        text = re.sub(r'\*\*|__|\*|_', '', text)
        
        return text.strip()

    async def answer(self, query, log_history=None):
        if log_history is None: 
            log_history = []
        
        # 1. Cache Check
        cache_key = f"{query}::{len(log_history)}"
        cached_resp = self.cache.get(cache_key)
        if cached_resp:
            return cached_resp

        # 2. Circuit Breaker Check
        if not self.circuit.allow_request():
            logger.warning("Circuit breaker OPEN → degraded AI response")
            return await self._degraded_ai_response(query)
            

        # 3. Construct Payload
        # Dynamic Few-Shot Selection
        dynamic_examples = select_dynamic_examples (
            query,
            self.examples,
            k=5
        )
        prompt_content = build_few_shot_prompt (
            self.system_prompt,
            dynamic_examples,
            query
        )
        messages = [{"role": "user", "content": prompt_content}]
        

        # 4. Execute with Retries
        for attempt in range(1, self.max_retries + 1):
            try:
                completion = await self.client.chat.completions.create(
                    model=self.api_model,
                    messages=messages,
                    max_tokens=512,
                    temperature=0.0, # Low temp for consistent terminal output
                )
                
                raw_text = completion.choices[0].message.content
                clean_text = self._sanitize(raw_text)
                
                # SSH safety: never return empty output
                if not clean_text.strip():
                    clean_text = "\n"
                
                # Success: Update State
                self.circuit.record_success()
                self.cache.set(cache_key, clean_text)
                
                return clean_text

            except Exception as e:
                logger.error(f"Attempt {attempt} failed: {e}")
                self.circuit.record_failure()
                await asyncio.sleep(0.5 * attempt) # Exponential backoff

        return await self._degraded_ai_response(query)
    async def _degraded_ai_response(self, query: str) -> str:
        try:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a Linux shell in a degraded state."
                        "Some services are unavailable."
                        "Respond briefly like a real Linux server."
                    )
                },
                {"role": "user", "content": query}
            ]
            completion = await self.client.chat.completions.create(
                model=self.api_model,
                messages=messages,
                max_tokens=64,
                temperature=0.2
            )
            text = self._sanitize(completion.choices[0].message.content)
            return text if text.strip() else "\n"
        
        except Exception as e:
            logger.error(f"Degraded AI failed: {e}")
            return "\n"