"""Thin adapters for the actual upstream FlashRAG SequentialPipeline."""

import asyncio

from rag_jev.models import Document


class JevFlashRAGRetriever:
    """Wrap a FlashRAG retriever without changing its corpus or document metadata.

    FlashRAG's SequentialPipeline uses its synchronous batch_search interface.
    In async applications call abatch_search instead. Selection is applied to
    each complete query context, never mixed across queries.
    """

    def __init__(self, retriever, selector, **policy):
        self.retriever, self.selector, self.policy = retriever, selector, policy
        self.traces = []

    async def abatch_search(self, queries):
        retrieved = self.retriever.batch_search(queries)
        if len(retrieved) != len(queries):
            raise ValueError("Retriever batch length mismatch")
        output = []
        self.traces = []
        for query, rows in zip(queries, retrieved, strict=True):
            result = await self.selector.select(
                query=query,
                documents=[Document(id=str(i), text=r["contents"]) for i, r in enumerate(rows)],
                **self.policy,
            )
            output.append([rows[int(i)] for i in result.selected_ids])
            self.traces.append(result)
        return output

    def batch_search(self, queries):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.abatch_search(queries))
        raise RuntimeError("Use await abatch_search inside an event loop")

    def _save_cache(self):
        return self.retriever._save_cache()


class EvidencePrompt:
    """Prompt object passed intact by upstream to our configurable Generator bridge."""

    def get_string(self, question, retrieval_result):
        return {
            "query": question,
            "documents": [{"id": r["id"], "text": r["contents"]} for r in retrieval_result],
        }


def config():
    return {
        "device": "cpu",
        "save_dir": ".",
        "save_metric_score": False,
        "save_intermediate_data": False,
        "metrics": [],
        "save_retrieval_cache": False,
        "use_fid": False,
        "refiner_name": None,
        "dataset_name": "crag_task1_pilot",
    }


def run_native(cases, retriever, generator):
    from flashrag.dataset import Dataset
    from flashrag.pipeline import SequentialPipeline

    data = Dataset(
        config=config(),
        data=[
            {"id": c["id"], "question": c["query"], "golden_answers": c.get("references", [])}
            for c in cases
        ],
    )
    pipeline = SequentialPipeline(
        config(), prompt_template=EvidencePrompt(), retriever=retriever, generator=generator
    )
    return pipeline.run(data, do_eval=False)
