from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass
from enum import Enum
import os
import threading
import queue
import random
import threading

import omegaconf

from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from nexusvoice.utils.logging import get_logger
logger = get_logger(__name__)

from nexusvoice.ai.InferenceEngine import InferenceEnginePool

class Agent(threading.Thread):
    """AI Agent that maintains conversation history and queues inference requests."""

    class AgentCommand(Enum):
        SHUTDOWN = "shutdown"
    
    def __init__(self, config, agent_id, resource_pool):
        super().__init__(daemon=True)
        self.config = config
        self.agent_id = agent_id
        self.resource_pool = resource_pool
        self.inference_engine = None
        self.request_queue = queue.Queue()  # Queue for incoming tasks
        self.history = []  # Conversation history
        self.running = False
        self.is_ready_event = threading.Event()

        # Set TOKENIZERS_PARALLELISM to True
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    def stop(self):
        """Stops the agent gracefully."""
        self.running = False
        self.request_queue.put((Agent.AgentCommand.SHUTDOWN, None))  # Wake up the thread to exit

    def process_request(self, prompt) -> Future:
        """Queues a request for processing and returns a Future immediately."""
        future = Future()
        logger.debug(f"[Agent {self.agent_id}]: Received request '{prompt}'")
        self.request_queue.put((prompt, future))
        return future  # Caller retrieves the result later

    def _init_conversation(self):
        """Initializes the conversation history."""
        self.history = []

        # Set System Prompt
        system_prompt = (
            "You are a helpful assistant. In addition to providing information, you can also help with tasks. "
            "Some of those tasks involve the control of a home automation system and include turning on or off lights, "
            "raising or lowering shades, turning on or off fans.  You must determine if the users request is asking "
            "for information or asking for a task to be performed.  If the user is asking for information, you should "
            "provide the information in a means which can be conveyed audibly.  If the user is asking for a task to be "
            "performed, you return a JSON object expressing the task intent and the parameters needed to perform the task."
            "Example: {\"intent\": \"turn_on\", \"device\": \"light\", \"room\": \"kitchen\"}"
            "Example: {\"intent\": \"turn_off\", \"device\": \"fan\", \"room\": \"living room\"}"
            "Example: {\"intent\": \"raise\", \"device\": \"shade\", \"room\": \"bedroom\"}"
            "Notes: - The system should be able to handle multiple devices and rooms in a single request."
            " - Be short and to the point in your responses.  Explaining your reasoning is not necessary."
        )

        system_message = { "role": "system", "content": system_prompt }
        self.history.append(system_message)
        
    def wait_until_ready(self):
        self.is_ready_event.wait()

    def run(self):
        """Agent's event loop - listens for requests and processes them asynchronously."""
        logger.info(f"Agent {self.agent_id} Started.")
        self._init_conversation()

        self.inference_engine = self.resource_pool.getResource(
            self.config.llm.name,
            model_id=self.config.llm.model)

        self.running = True        
        self.is_ready_event.set()


        while self.running:
            try:
                prompt, future = self.request_queue.get()
                if prompt == Agent.AgentCommand.SHUTDOWN or not self.running:
                    break  # Exit thread gracefully
                
                logger.debug(f"[Agent {self.agent_id}]: Processing request '{prompt}'...")
                self._process_request(prompt, future)
            except queue.Empty:
                continue  # No requests, keep running

    def _process_request(self, prompt, future):
        """Handles request processing and delegates inference."""
        self.history.append({"role": "user", "content": prompt})

        # Submit inference task to the inference executor
        result = self._infer(prompt)

        self.history.append({"role": "assistant", "content": result})

        future.set_result(result)  # Complete the future

    def _infer(self, prompt):
        """Simulates AI inference (Replace with actual AI model call)."""
        logger.debug(f"[Agent {self.agent_id}] Running inference '{prompt}'...")

        tokenizer = self.inference_engine.getTokenizer()
        tokenizer.pad_token = tokenizer.eos_token
        
        inference_engine = self.resource_pool.getResource(
            "Llama-3.2",
            model_id="meta-llama/Llama-3.2-3B-Instruct")
        
        chat_history = tokenizer.apply_chat_template(
            self.history,
            tokenize=False,
            add_generation_prompt=True
        )

        with self.resource_pool.getResourceLock("Llama-3.2"):
            inference_engine.loadResource()
            # Perform inference
            print(tokenizer.pad_token)
            inputs = tokenizer.encode(
                chat_history,
                add_special_tokens=False,
                return_tensors="pt",
                padding=True,
                truncation=True)
            input_length = inputs.shape[1]
            
            attention_mask = (inputs != tokenizer.pad_token_id).long().to(self.inference_engine.device)

            outputs = inference_engine.generate(
                inputs,
                attention_mask=attention_mask,
                max_length=4096)
            
            new_outputs = outputs[0, input_length:]
            # logger.debug("="*50)
            # logger.debug(f"Input Length: {input_length}")
            # logger.debug(new_outputs.shape)
            # logger.debug(f"Output Length: {len(outputs[0])}")
            # logger.debug("="*50)
            # logger.debug(f"Outputs: {new_outputs}")
            # logger.debug("="*50)
            # logger.debug("-".join([tokenizer.decode([tok], skip_special_tokens=False) for tok in outputs[0]]))
            # logger.debug("="*50)

            result = tokenizer.decode(new_outputs, skip_special_tokens=True)
            self.history.append({"role": "assistant", "content": result})

            return result
        # Create inputs from conversation history and prompt
        

        # time.sleep(random.randint(1,6))  # Simulated delay
        # return f"[Agent {self.agent_id}] AI Response to '{prompt}'"

class AgentManager:
    """Manages AI agents and a controlled thread pool for execution."""

    def __init__(self, config, max_agents=2):
        self.config = config 
        self.agent_executor = ThreadPoolExecutor(max_workers=max_agents)  # Limits running agents
        self.agents = {}
        self.lock = threading.Lock()

        self.resource_pool = InferenceEnginePool()

    def get_agent(self, agent_id) -> Agent:
        """Returns an agent instance and starts it if necessary."""
        if agent_id not in self.agents:
            agent = Agent(self.config, agent_id, self.resource_pool)
            agent.start()  # Runs as a thread
            self.agents[agent_id] = agent
        return self.agents[agent_id]

    def shutdown(self):
        """Stops all agents and shuts down executors."""
        with self.lock:
            for agent in self.agents.values():
                agent.stop()
            self.agents.clear()
        self.agent_executor.shutdown(wait=True)


# Example Usage
if __name__ == "__main__":
    config = omegaconf.OmegaConf.load("config.yml")
    manager = AgentManager(config, max_agents=3)

    # Create a number of agents
    num_agents = 5
    for i in range(num_agents):
        agent = manager.get_agent(f"agent::{i:02d}")
        
    # Submit inference requests (non-blocking)
    # 20 example requests
    requests = [
        "What's the weather like?",
        "Tell me a joke!",
        "Who are you?",
        "What's the time now?",
        "How are you doing?",
        "What's 2+2?",
        "What's the capital of France?",
        "Turn on the lights!",
        "What's the latest news?",
        "What's the temperature?",
        "What's your name?",
        "What's the meaning of life?",
        "What's the best movie?",
    ]

    import random
    tasks = []
    for i in range(num_agents):
        agent = manager.get_agent(f"agent::{i:02d}")
        request = random.choice(requests)
        future = agent.process_request(request)
        tasks.append(future)
    logger.debug("Requests sent, continuing execution...")

    # Retrieve results asynchronously
    for response in as_completed(tasks):
        logger.debug(f"    Response: {response.result()}")

    # Shutdown manager when done
    manager.shutdown()