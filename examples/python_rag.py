"""Plug the selector between your existing retrieve() and generate() functions."""

import asyncio
import os

from dotenv import load_dotenv

from rag_jev import ContextSelector, Jev
from rag_jev.demo import CASES


async def main() -> None:
    load_dotenv()
    case = CASES[0]  # Replace with your retriever's query and documents.
    async with Jev(api_key=os.environ["TYPESAFE_API_KEY"]) as provider:
        selector = ContextSelector(provider, on_error="raise")
        result = await selector.select(
            query=case.query,
            documents=case.documents,
            min_relevance=0.2,
            shadow=False,
        )
    print(result.model_dump_json(indent=2))
    # Supply result.documents to your existing answering model, including source metadata.


if __name__ == "__main__":
    asyncio.run(main())
