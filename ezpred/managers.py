from magnet import Manager as _Manager
from torchmanager_core import deprecated
from torchmanager_core.typing import Module


@deprecated('v0.2', 'v1.0')
class Manager(_Manager[Module]):
    pass
