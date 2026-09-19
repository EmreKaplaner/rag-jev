from dify_plugin import ModelProvider
from dify_plugin.entities.model import ModelType


class RagJevProvider(ModelProvider):
    def validate_provider_credentials(self, credentials: dict) -> None:
        self.get_model_instance(ModelType.RERANK).validate_credentials("jev", credentials)
