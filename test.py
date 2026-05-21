import requests
import json
import time

URL = "http://localhost:8000/chat"

print("==============================================")
print("  SHL Recommender - Local Evaluation Harness  ")
print("==============================================\n")

def send_message(history):
    print(f"--- Turn {len(history) // 2 + 1} ---")
    print(f"User: {history[-1]['content']}")
    
    start = time.time()
    try:
        response = requests.post(URL, json={"messages": history})
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Server unreachable or failed. Is uvicorn running?\nDetails: {e}")
        return None
        
    duration = time.time() - start
    
    data = response.json()
    print(f"\nAssistant ({duration:.2f}s): {data['reply']}")
    if data['recommendations']:
        print("\nRecommendations:")
        for r in data['recommendations']:
            print(f"  - [{r['test_type']}] {r['name']}\n    URL: {r['url']}")
    print(f"\nEnd of Conversation: {data['end_of_conversation']}\n")
    return data

# --- Simulated Conversation ---

history = []

# Turn 1: Vague Intent (Should Clarify)
history.append({"role": "user", "content": "I need an assessment for a developer."})
response = send_message(history)

if response:
    history.append({"role": "assistant", "content": response["reply"]})
    
    # Turn 2: Give Context (Should Recommend)
    history.append({"role": "user", "content": "A mid-level Java developer."})
    response = send_message(history)

if response:
    history.append({"role": "assistant", "content": response["reply"]})
    
    # Turn 3: Refine Criteria (Should Refine)
    history.append({"role": "user", "content": "Can you also add a personality test?"})
    response = send_message(history)

print("Test complete. Compare the output above with the assignment schemas to verify!")
