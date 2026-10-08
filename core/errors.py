"""错误码（E100~E500）。作者：晨星"""

from __future__ import annotations


class ForgeError(Exception):
    code = "E000"

    def __init__(self, msg: str = ""):
        super().__init__(f"[{self.code}] {msg}")
        self.msg = msg


class EnvError(ForgeError):
    code = "E100"


class PolicyError(ForgeError):
    code = "E200"


class DataError(ForgeError):
    code = "E300"


class ConfigError(ForgeError):
    code = "E400"


class MethodError(ForgeError):
    code = "E500"
