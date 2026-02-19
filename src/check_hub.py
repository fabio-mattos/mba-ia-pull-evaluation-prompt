import inspect
from langchain import hub

try:
    sig = inspect.signature(hub.push)
    print(f"Signature: {sig}")
except Exception as e:
    print(f"Error inspecting signature: {e}")
    # Fallback: dir(hub)
    print(f"Dir hub: {dir(hub)}")
