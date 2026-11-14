import os
import sys
import time
from dotenv import load_dotenv
from google import genai
from google.genai import errors

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


MODELS = ("gemini-3.5-flash", "gemini-3.6-flash")
RETRYABLE_STATUS_CODES = (429, 500, 502, 503, 504)


def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")

    if not api_key:
        print(
            "API key not found. Set GEMINI_API_KEY or GENAI_API_KEY in your .env file.",
            file=sys.stderr,
        )
        return 1

    question = input("What would you like to ask? ").strip()
    if not question:
        print("Please enter a question and try again.", file=sys.stderr)
        return 1

    client = genai.Client(api_key=api_key)
    for model in MODELS:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=question,
                )
                print("\nAI Assistant")
                print(response.text or "(The model returned an empty response.)")
                return 0
            except errors.APIError as exc:
                if exc.code == 404:
                    print(f"{model} is unavailable; trying the backup model.", file=sys.stderr)
                    break

                if exc.code not in RETRYABLE_STATUS_CODES:
                    print(
                        f"Gemini request failed: {exc.message or 'Please check your API key and settings.'}",
                        file=sys.stderr,
                    )
                    return 1

                if attempt < 2:
                    delay = 2 ** (attempt + 1)
                    print(
                        f"Gemini is temporarily unavailable. Retrying in {delay} seconds "
                        f"(attempt {attempt + 1}/3)...",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                else:
                    print(f"{model} is still unavailable; trying the backup model.", file=sys.stderr)

    print(
        "Sorry, Gemini is unavailable right now. Please try again later.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
