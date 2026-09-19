"""Run with the plugin environment; real SDK discovery plus real HTTP invocation."""

import os
import sys
from pathlib import Path

from dify_plugin import DifyPluginEnv
from dify_plugin.core.plugin_registration import PluginRegistration
from dify_plugin.entities.model import ModelType
from dify_plugin.errors.model import InvokeError

os.chdir(Path(__file__).resolve().parents[1] / "integrations" / "dify")
sys.path.insert(0, os.getcwd())
registration = PluginRegistration(DifyPluginEnv())
provider = registration.models_mapping["rag_jev"][1]
model = provider.get_model_instance(ModelType.RERANK)
credentials = {
    "base_url": os.environ["RAG_JEV_TEST_URL"],
    "api_token": os.environ.get("RAG_JEV_TEST_TOKEN", ""),
}
docs = [
    "Orders ship within 5 business days.",
    "You may request a refund within 30 days of purchase.",
    "Keep your receipt; it is required for refund requests.",
]
result = model.invoke(
    "jev", credentials, "What is the refund deadline?", docs, score_threshold=0.2, top_n=2
)
assert [d.index for d in result.docs] == [1, 2], result
assert all(d.text == docs[d.index] for d in result.docs)
contextual = model.invoke(
    "jev",
    {**credentials, "scoring_strategy": "contextual"},
    "What is the refund deadline?",
    docs,
    score_threshold=0.2,
    top_n=2,
)
assert [d.index for d in contextual.docs] == [1, 2], contextual
empty = model.invoke("jev", credentials, "What is the refund deadline?", docs, score_threshold=1)
assert empty.docs == []
try:
    model.invoke("jev", {**credentials, "api_token": "wrong"}, "question", docs)
except InvokeError:
    pass
else:
    raise AssertionError("invalid service credentials accepted")
print("Dify SDK discovery, both scoring strategies, threshold, indices, and auth verified")
