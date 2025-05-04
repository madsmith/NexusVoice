from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from enum import Enum
import os
import threading
import queue
import random
import threading
from typing import List

import omegaconf

from nexusvoice.protocol.mcp import MCPMessage, ModelMessage, ToolResult, UserMessage
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
        self._inference_engine = None
        self.request_queue = queue.Queue()  # Queue for incoming tasks
        self.history = []  # Conversation history
        self.running = False
        self.is_ready_event = threading.Event()

        # Set TOKENIZERS_PARALLELISM to True
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

    @property
    def inference_engine(self):
        assert self._inference_engine is not None, "Inference engine not initialized"
        return self._inference_engine
    
    def stop(self):
        """Stops the agent gracefully."""
        self.running = False
        self.request_queue.put((Agent.AgentCommand.SHUTDOWN, None))  # Wake up the thread to exit

    def process_request(self, prompt) -> Future[str]:
        """Queues a request for processing and returns a Future immediately."""
        future = Future()
        logger.debug(f"[Agent {self.agent_id}]: Received request '{prompt}'")
        self.request_queue.put((prompt, future))
        return future  # Caller retrieves the result later

    def process_tool_response(self, tool_responses: List[ToolResult]) -> Future[ModelMessage]:
        """Queues a tool response for processing and returns a Future immediately."""
        future: Future[ModelMessage] = Future()
        logger.debug(f"[Agent {self.agent_id}]: Received tool response '{tool_responses}'")
        self.request_queue.put((tool_responses, future))
        return future
    
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
        system_prompt = self.config.llm.system_prompt or system_prompt

        system_message = { "role": "system", "content": system_prompt }
        self.history.append(system_message)
        
    def wait_until_ready(self):
        self.is_ready_event.wait()

    def run(self):
        """Agent's event loop - listens for requests and processes them asynchronously."""
        logger.info(f"Agent {self.agent_id} Started.")
        self._init_conversation()

        self._inference_engine = self.resource_pool.getResource(
            self.config.llm.name,
            model_id=self.config.llm.model)

        self.running = True        
        self.is_ready_event.set()

        while self.running:
            try:
                request, future = self.request_queue.get()
                if request == Agent.AgentCommand.SHUTDOWN or not self.running:
                    break  # Exit thread gracefully
                
                # Handle MCP style requests
                is_MCP_list = isinstance(request, List) and all(isinstance(item, MCPMessage) for item in request)
                is_MCP = isinstance(request, MCPMessage) or is_MCP_list
                if is_MCP:
                    logger.debug(f"[Agent {self.agent_id}]: Processing MCP request '{request}'...")
                    self._process_mcp_request(request, future)
                else:
                    # Handle regular requests (presumptively strings) 
                    logger.debug(f"[Agent {self.agent_id}]: Processing request '{request}'...")
                    self._process_request(request, future)
            except queue.Empty:
                continue  # No requests, keep running

    def _process_mcp_request(self, request, future):
        """ Handle the MCP Message and infer a new response, fullfilling the future. """
        if isinstance(request, UserMessage):
            self.history.append({"role": "user", "content": request.text})
            
            result_text = self._run_inference()

            self.history.append({"role": "assistant", "content": result_text})

            # Wrap the result in a ModelMessage
            try:
                message = ModelMessage.model_validate_json(result_text)
            except Exception:
                message = ModelMessage(text=result_text)

            future.set_result(message)

        elif isinstance(request, List) and all(isinstance(item, ToolResult) for item in request):
            for tool_result in request:
                # ID is currently not supported by the chat template
                self.history.append({
                    "role": "tool",
                    id: tool_result.id,
                    "content": tool_result.output
                })

            result_text = self._run_inference()

            self.history.append({"role": "assistant", "content": result_text})

            # Wrap the result in a ModelMessage
            try:
                message = ModelMessage.model_validate_json(result_text)
            except Exception:
                message = ModelMessage(text=result_text)

            future.set_result(message)
    

    def _process_request(self, prompt, future):
        """Handles request processing and delegates inference."""
        self.history.append({"role": "user", "content": prompt})

        # Submit inference task to the inference executor
        result = self._run_inference()

        self.history.append({"role": "assistant", "content": result})

        future.set_result(result)  # Complete the future

    def _run_inference(self):
        """Simulates AI inference (Replace with actual AI model call)."""
        logger.debug(f"[Agent {self.agent_id}] Running inference...")

        tokenizer = self.inference_engine.tokenizer
        
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
            logger.trace(f"[Agent {self.agent_id}] Chat History: {chat_history}")
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
                pad_token_id=tokenizer.pad_token_id,
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