from enum import Enum


class SpecMode(Enum):
    NONE = "none"
    MTP = "mtp"
    DFLASH_B4 = "dflash_b4"
    DFLASH_B8 = "dflash_b8"
    DFLASH_B16 = "dflash_b16"
    DFLASH_B32 = "dflash_b32"


class Strategy:
    def __init__(
        self,
        spec_mode: SpecMode = SpecMode.NONE,
        prefetch_source: str = "router_lookahead",
        cache_policy: str = "lru",
        block_size: int = 0,
    ):
        self.spec_mode = spec_mode
        self.prefetch_source = prefetch_source
        self.cache_policy = cache_policy
        self.block_size = block_size or self._default_block_size()

    def _default_block_size(self) -> int:
        mapping = {
            SpecMode.NONE: 0,
            SpecMode.MTP: 3,
            SpecMode.DFLASH_B4: 4,
            SpecMode.DFLASH_B8: 8,
            SpecMode.DFLASH_B16: 16,
            SpecMode.DFLASH_B32: 32,
        }
        return mapping.get(self.spec_mode, 0)

    def label(self) -> str:
        return f"{self.spec_mode.value}_prefetch={self.prefetch_source}"


STRATEGY_NONE = Strategy(SpecMode.NONE)
STRATEGY_MTP = Strategy(SpecMode.MTP, prefetch_source="router_lookahead")
STRATEGY_DFLASH_B8 = Strategy(SpecMode.DFLASH_B8, prefetch_source="draft_guided")
STRATEGY_DFLASH_B16 = Strategy(SpecMode.DFLASH_B16, prefetch_source="draft_guided")
STRATEGY_DFLASH_B32 = Strategy(SpecMode.DFLASH_B32, prefetch_source="draft_guided")
