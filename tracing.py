"""
CS 5374 - LangSmith Tracing Configuration
Enables LangSmith tracing for all LangChain/LangGraph operations.

Usage:
    from tracing import enable_tracing
    enable_tracing("bitcoin-wallet-test")  # sets project name

Setup:
    1. Get your API key from https://smith.langchain.com/settings
    2. Create a .env file in the project root:
         LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxx
    3. Or set the environment variable directly:
         export LANGCHAIN_API_KEY=lsv2_pt_xxxxxxxxxxxxxxxx  (Linux/Mac)
         $env:LANGCHAIN_API_KEY="lsv2_pt_xxxxxxxxxxxxxxxx"  (PowerShell)

Once enabled, all ChatOllama calls and LangGraph executions are
automatically traced and visible at https://smith.langchain.com
"""

import os


def enable_tracing(project_name: str = "cs5374-adversarial-test", verbose: bool = True) -> bool:
    """
    Enable LangSmith tracing for all LangChain/LangGraph operations.

    Args:
        project_name: Project name shown in the LangSmith dashboard.
                      Use different names per test type for easy filtering.
        verbose: Print status messages about tracing setup.

    Returns:
        True if tracing was enabled, False if API key is missing.
    """
    # Check for API key in environment or .env file
    api_key = os.environ.get("LANGCHAIN_API_KEY", "")

    if not api_key:
        # Try loading from .env file
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if os.path.exists(env_path):
            with open(env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key == "LANGCHAIN_API_KEY":
                            api_key = value
                            break

    if not api_key:
        if verbose:
            print("[TRACING] No LANGCHAIN_API_KEY found. Tracing disabled.")
            print("[TRACING] To enable: create .env with LANGCHAIN_API_KEY=lsv2_pt_...")
            print("[TRACING] Get your key at https://smith.langchain.com/settings")
        return False

    # Set all required environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_PROJECT"] = project_name
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

    if verbose:
        print(f"[TRACING] LangSmith enabled for project: {project_name}")
        print(f"[TRACING] View traces at: https://smith.langchain.com/o/default/projects")

    return True


def disable_tracing():
    """Disable LangSmith tracing."""
    os.environ.pop("LANGCHAIN_TRACING_V2", None)
    os.environ.pop("LANGCHAIN_PROJECT", None)
