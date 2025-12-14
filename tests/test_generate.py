from app.generate import generate_text

if __name__ == "__main__":
    system = "You are a helpful assistant."
    prompt = "Say hello in one sentence."

    out = generate_text(system, prompt)
    print(out)
