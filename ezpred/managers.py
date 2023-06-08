from magnet import Manager as _Manager
from torchmanager_core.typing import Any, Module


class Manager(_Manager[Module]):
    def unpack_data(self, data: dict[str, Any]) -> tuple[Any, Any]:
        image, label = data["image"], data["label"]
        return image, label
