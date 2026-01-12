import subprocess

def generate_with_ollama(prompt: str, model: str = "mistral") -> str:

    """
    Call Ollama locally to generate text
    """

    try:
        proc = subprocess.run(
            ["ollama", "run", model],
            input = prompt,
            text = True,
            capture_output = True,
            timeout = 60
        )
        return proc.stdout.strip()
    except Exception as e:
        return f"Ollama error: {e}"