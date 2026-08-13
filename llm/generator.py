import os

from dotenv import load_dotenv
from google import genai


load_dotenv()


MODEL_NAME = "gemini-2.5-flash"


class FinancialLLM:

    def __init__(self):

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY was not found in .env"
            )

        print("Initializing Gemini...")

        self.client = genai.Client(
            api_key=api_key
        )

        print(
            f"Gemini model: {MODEL_NAME}"
        )

    def generate(
        self,
        question: str,
        context: str,
    ):

        prompt = f"""
You are FinRAG AI, a financial research assistant.

Answer the user's question using ONLY the
financial information provided in the context.

Do not invent financial figures.

If the answer cannot be determined from the
provided context, clearly say that the information
is not available in the retrieved documents.

Always provide a concise explanation and identify
the company and fiscal year when relevant.

Use clear, professional formatting with normal
spacing between words, numbers, and currency symbols.
Do not remove spaces between words or between
currency symbols and amounts.

User question:
{question}

Financial context:
{context}

Answer:
"""

        response = self.client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        return response.text