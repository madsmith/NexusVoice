import time
import argparse
from nexusvoice.ai.pydantic_agent import HomeAutomationAgent, NexusSupportDependencies
from nexusvoice.core.config import load_config
from pydantic_ai.providers.openai import OpenAIProvider

# Example prompt and number of samples
test_prompt = "Turn on the living room lights"

# Define different agent variants for benchmarking
def agent_variants():
    config = load_config()

    # Default agent (no custom provider)
    openai_models = [
        ("o4-mini", "OpenAI - O4-mini"),
        ("gpt-4o-mini-2024-07-18", "OpenAI - GPT-4o-mini"),
        ("o3-mini-2025-01-31", "OpenAI - O3-mini"),
        ("gpt-4.1-nano-2025-04-14", "OpenAI - GPT-4.1-nano"),
        ("gpt-4.1-mini-2025-04-14", "OpenAI - GPT-4.1-mini"),
        ("gpt-3.5-turbo-1106", "OpenAI - GPT-3.5-turbo"),
        ("gpt-4o-2024-08-06", "OpenAI - GPT-4o"),
        ("gpt-4.1-2025-04-14", "OpenAI - GPT-4.1")
    ]

    for model, name in openai_models:
        config.set("agents.home_automation.model", model)
        yield name, lambda: HomeAutomationAgent(NexusSupportDependencies(config=config))
    
    # Custom provider agent
    config.set("agents.home_automation.model", "llama-3.2-8x3b-moe-dark-champion-instruct-uncensored-abliterated-18.4b")
    custom_provider = OpenAIProvider(
        api_key=config.get("openai.api_key", ""),
        base_url="http://localhost:1234/v1/"
    )
    yield "LM - LLama 3.2 8x3B Moe Instruct Uncensored", lambda: HomeAutomationAgent(NexusSupportDependencies(config=config), provider=custom_provider)

    config.set("agents.home_automation.model", "llama-3.2-3b-instruct")
    custom_provider = OpenAIProvider(
        api_key=config.get("openai.api_key", ""),
        base_url="http://localhost:1234/v1/"
    )
    yield "LM - LLama 3.2 3B Instruct", lambda: HomeAutomationAgent(NexusSupportDependencies(config=config), provider=custom_provider)

    config.set("agents.home_automation.model", "hermes-3-llama-3.2-3b")
    custom_provider = OpenAIProvider(
        api_key=config.get("openai.api_key", ""),
        base_url="http://localhost:1234/v1/"
    )
    yield "LM - Hermes 3 LLama 3.2 3B", lambda: HomeAutomationAgent(NexusSupportDependencies(config=config), provider=custom_provider)

    # config.set("agents.home_automation.model", "llama-3.3-70b-instruct")
    # custom_provider = OpenAIProvider(s
    #     api_key=config.get("openai.api_key", ""),
    #     base_url="http://localhost:1234/v1/"
    # )
    # yield "LM - LLama 3.3 70B Instruct", lambda: HomeAutomationAgent(NexusSupportDependencies(config=config), provider=custom_provider)

    # Add more variants here as needed

# Dummy tool for registration
# Now as a factory to track per-sample calls
def home_control_factory():
    calls = []
    def home_control(ctx, intent: str, device: str, room: str) -> list[dict[str, str]]:
        calls.append({"intent": intent, "device": device, "room": room})
        status = [
            {
                "room": "living room",
                "state": "on"
            }
        ]
        return status
    return home_control, calls


def run_timing(num_samples=5, name_filter=None):
    results = {}
    accuracy_results = {}
    for name, agent_factory in agent_variants():
        if name_filter and name_filter.lower() not in name.lower():
            continue

        # Setup agent and tool
        home_control, calls = home_control_factory()
        agent = agent_factory()
        agent.register_tool(home_control)
        
        # Run timing
        durations = []
        valid_count = 0
        for i in range(num_samples):
            # Time the call
            start = time.time()
            try:
                result = agent.run_sync(test_prompt)
            except Exception as e:
                print(f"[{name}] Sample {i+1}: Exception: {str(e)}")
                continue
            end = time.time()
            duration = end - start
            durations.append(duration)
            call = calls[-1] if calls else {}
            valid = (
                "living" in call.get("room", "") and
                "on" in call.get("intent", "")
            )
            if valid:
                valid_count += 1
            status = "VALID" if valid else "INVALID"
            print(f"[{name}] Sample {i+1}: {duration:.3f}s | {status}")
            print(f"       Args: {call}")
            print(f"       Response: {getattr(result.data, 'text', result.data)}")

        if len(durations) == 0:
            avg = float('nan')
        else:
            avg = sum(durations) / len(durations)
        if num_samples == 0:
            accuracy = float('nan')
        else:
            accuracy = valid_count / num_samples
        results[name] = avg
        accuracy_results[name] = accuracy
        print(f"[{name}] Average duration over {num_samples} samples: {avg:.3f}s | Accuracy: {accuracy:.2%}\n")

    # Print final summary
    print("Summary of results:")
    print(f"{'Agent Name':<45} | {'Avg Duration':<12} | {'Accuracy':<8}")
    print(f"{'-'*45}-+-{'-'*12}-+-{'-'*8}")
    for name in results:
        # Truncate name if too long
        if len(name) > 45:
            display_name = name[:42] + "..."
        else:
            display_name = name
        print(f"{display_name:<45} | {results[name]:<12.3f} | {accuracy_results[name]:<8.2%}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark HomeAutomationAgent variants.")
    parser.add_argument('-n', type=int, default=5, help='Number of runs per agent')
    parser.add_argument('name', nargs='?', default=None, help='Optional substring to filter agent names')
    args = parser.parse_args()

    run_timing(num_samples=args.n, name_filter=args.name)

if __name__ == "__main__":
    main()
