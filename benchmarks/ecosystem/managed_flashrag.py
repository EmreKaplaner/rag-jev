"""Reusable synchronous FlashRAG adapter with an explicit async-client lifetime."""

import asyncio

from benchmarks.ecosystem.flashrag import JevFlashRAGRetriever


class ManagedJevFlashRAGRetriever(JevFlashRAGRetriever):
    """Use one event loop across repeated native pipeline.run calls.

    Async HTTP pools and semaphore contention bind to an event loop. Creating a
    new loop for every synchronous batch can fail on a later batch. A Runner
    keeps their loop alive. Set close_provider=True only when this adapter owns
    the supplied provider; externally owned clients remain the caller's duty.
    """

    def __init__(self, retriever, selector, *, close_provider=False, **policy):
        super().__init__(retriever, selector, **policy)
        self.close_provider = close_provider
        self._runner = None
        self._closed = False

    def batch_search(self, queries):
        if self._closed:
            raise RuntimeError("Retriever is closed")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            if self._runner is None:
                self._runner = asyncio.Runner()
            return self._runner.run(self.abatch_search(queries))
        raise RuntimeError("Use await abatch_search inside an event loop")

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._runner is not None:
            try:
                close = getattr(self.selector.provider, "aclose", None)
                if self.close_provider and close is not None:
                    self._runner.run(close())
            finally:
                self._runner.close()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Retriever is closed")
        return self

    def __exit__(self, *args):
        self.close()
