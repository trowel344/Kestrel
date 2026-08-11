"""Secure node inventory and deterministic llama.cpp RPC placement planning.

TCP reachability remains distinct from a bounded protocol/device probe. Neither
authenticates the worker, so callers still need a trusted transport (normally
an authenticated SSH tunnel to a loopback-bound RPC server).
"""

from __future__ import annotations

import ipaddress
import json
import math
import re
import socket
import struct
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .config import config_path as _config_path
from .errors import KestrelError
from .util import write_atomic

SCHEMA_VERSION = 1
DEFAULT_RPC_PORT = 50052
MAX_PROBE_TIMEOUT = 10.0
MAX_RPC_DEVICES = 64
RPC_PROTO_MAJOR_VERSION = 4
RPC_PROTO_MINOR_VERSION = 0
RPC_PROTO_PATCH_VERSION = 0
RPC_CONN_CAPS_SIZE = 24
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_HOST_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_GIT_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{7,64}$")


class NodeError(KestrelError):
    """Base class for node inventory and placement failures."""

    code = "node_error"


class NodeValidationError(NodeError):
    code = "node_invalid"


class NodeStateError(NodeError):
    code = "node_state_error"


class NodeSecurityError(NodeError):
    code = "node_security_error"


class NodePlanningError(NodeError):
    code = "node_planning_error"


@dataclass(frozen=True)
class Endpoint:
    """A parsed host/port pair suitable for a TCP connection."""

    host: str
    port: int

    @property
    def address(self) -> str:
        return format_endpoint(self.host, self.port)


def parse_endpoint(value: str, *, default_port: int | None = None) -> Endpoint:
    """Parse an RPC endpoint.

    Supported forms are ``hostname:port``, ``IPv4:port`` and
    ``[IPv6]:port``.  Bare hosts may use ``default_port``; unbracketed IPv6
    with a port is rejected because it is ambiguous.
    """

    if not isinstance(value, str) or not value or value.strip() != value:
        raise NodeValidationError("RPC endpoint must be a non-empty string without surrounding whitespace")
    if any(char.isspace() for char in value):
        raise NodeValidationError("RPC endpoint must not contain whitespace")
    host: str
    port_text: str | None
    if value.startswith("["):
        closing = value.find("]")
        if closing < 0:
            raise NodeValidationError(f"invalid bracketed RPC endpoint: {value!r}")
        host = value[1:closing]
        suffix = value[closing + 1 :]
        if suffix:
            if not suffix.startswith(":") or not suffix[1:]:
                raise NodeValidationError(f"invalid RPC endpoint port: {value!r}")
            port_text = suffix[1:]
        else:
            port_text = str(default_port) if default_port is not None else None
    else:
        colon_count = value.count(":")
        if colon_count > 1:
            # A raw IPv6 literal is useful with a default port, but an
            # unbracketed literal carrying a port cannot be interpreted safely.
            try:
                ipaddress.IPv6Address(value)
            except ValueError as exc:
                raise NodeValidationError(f"IPv6 RPC endpoints must use [address]:port: {value!r}") from exc
            host, port_text = value, str(default_port) if default_port is not None else None
        elif colon_count == 1:
            host, port_text = value.rsplit(":", 1)
        else:
            host, port_text = value, str(default_port) if default_port is not None else None
    if not host:
        raise NodeValidationError(f"RPC endpoint has no host: {value!r}")
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None
    if ip is None and not _HOST_RE.fullmatch(host):
        raise NodeValidationError(f"invalid RPC endpoint host: {host!r}")
    if port_text is None:
        raise NodeValidationError(f"RPC endpoint must include a port: {value!r}")
    if not port_text.isdigit():
        raise NodeValidationError(f"RPC endpoint port must be numeric: {port_text!r}")
    port = int(port_text)
    if not 1 <= port <= 65535:
        raise NodeValidationError(f"RPC endpoint port out of range: {port}")
    return Endpoint(host, port)


def format_endpoint(host: str, port: int) -> str:
    """Format an endpoint, adding brackets when the host is IPv6."""

    return f"[{host}]:{port}" if ":" in host else f"{host}:{port}"


def is_loopback_host(host: str) -> bool:
    """Return whether *host* is an explicitly loopback address/name."""

    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _check_endpoint_security(endpoint: Endpoint, *, allow_insecure_direct_rpc: bool) -> None:
    if not allow_insecure_direct_rpc and not is_loopback_host(endpoint.host):
        raise NodeSecurityError(
            f"direct RPC endpoint {endpoint.address} is not loopback-only",
            hint="Use an SSH tunnel or explicitly set allow_insecure_direct_rpc=True on a trusted network.",
        )


def node_store_path(config_file: str | Path | None = None) -> Path:
    """Return the node inventory beside Kestrel's TOML configuration."""

    config = Path(config_file) if config_file is not None else _config_path()
    return config.expanduser().parent / "nodes.json"


@dataclass(frozen=True)
class Node:
    name: str
    endpoint: str
    accelerator_memory_mib: int
    ram_mib: int | None = None
    enabled: bool = True
    engine_version: str | None = None
    engine_commit: str | None = None
    model_cache_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_RE.fullmatch(self.name):
            raise NodeValidationError("node name must be 1-128 characters of letters, digits, '.', '-' or '_'")
        parse_endpoint(self.endpoint)
        for field_name in ("accelerator_memory_mib", "ram_mib"):
            value = getattr(self, field_name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0
            ):
                raise NodeValidationError(f"{field_name} must be a finite non-negative number or null")
        if not isinstance(self.enabled, bool):
            raise NodeValidationError("enabled must be a boolean")
        for field_name in ("engine_version", "engine_commit"):
            value = getattr(self, field_name)
            if value is not None and (not isinstance(value, str) or not value or value.strip() != value):
                raise NodeValidationError(f"{field_name} must be a non-empty string or null")
        if self.engine_commit is not None and not _GIT_COMMIT_RE.fullmatch(self.engine_commit):
            raise NodeValidationError("engine_commit must be a 7-64 character hexadecimal git commit")
        if not isinstance(self.model_cache_hashes, (tuple, list)):
            raise NodeValidationError("model_cache_hashes must be a sequence of SHA-256 strings")
        hashes = tuple(self.model_cache_hashes)
        if len(set(hashes)) != len(hashes) or any(
            not isinstance(item, str) or not _SHA256_RE.fullmatch(item) for item in hashes
        ):
            raise NodeValidationError("model_cache_hashes must contain unique 64-character hexadecimal SHA-256 hashes")
        object.__setattr__(self, "model_cache_hashes", hashes)

    @property
    def parsed_endpoint(self) -> Endpoint:
        return parse_endpoint(self.endpoint)

    @property
    def rpc_endpoint(self) -> str:
        return self.parsed_endpoint.address

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "endpoint": self.rpc_endpoint,
            "accelerator_memory_mib": self.accelerator_memory_mib,
            "ram_mib": self.ram_mib,
            "enabled": self.enabled,
            "engine_version": self.engine_version,
            "engine_commit": self.engine_commit,
            "model_cache_hashes": list(self.model_cache_hashes),
        }


_NODE_KEYS = {
    "name",
    "endpoint",
    "accelerator_memory_mib",
    "ram_mib",
    "enabled",
    "engine_version",
    "engine_commit",
    "model_cache_hashes",
}


def _node_from_dict(raw: Any) -> Node:
    if not isinstance(raw, dict) or set(raw) != _NODE_KEYS:
        raise NodeStateError("node inventory contains an object with missing or unknown fields")
    try:
        return Node(**raw)
    except (NodeValidationError, TypeError) as exc:
        if isinstance(exc, NodeValidationError):
            raise NodeStateError(f"invalid node inventory entry: {exc.message}") from exc
        raise NodeStateError("invalid node inventory entry types") from exc


class NodeStore:
    """Crash-safe named-node inventory backed by one strict JSON document."""

    def __init__(self, path: str | Path | None = None, *, allow_insecure_direct_rpc: bool = False) -> None:
        self.path = Path(path) if path is not None else node_store_path()
        self.allow_insecure_direct_rpc = allow_insecure_direct_rpc

    def load(self) -> tuple[Node, ...]:
        if not self.path.exists() and not self.path.is_symlink():
            return ()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise NodeStateError(f"unable to read node inventory {self.path}: {exc}") from exc
        if not isinstance(raw, dict) or set(raw) != {"schema_version", "nodes"}:
            raise NodeStateError("node inventory has an unknown or missing schema")
        if type(raw["schema_version"]) is not int or raw["schema_version"] != SCHEMA_VERSION:
            raise NodeStateError(f"unsupported node inventory schema version: {raw.get('schema_version')!r}")
        if not isinstance(raw["nodes"], list):
            raise NodeStateError("node inventory 'nodes' must be an array")
        nodes = tuple(_node_from_dict(item) for item in raw["nodes"])
        if len({node.name for node in nodes}) != len(nodes):
            raise NodeStateError("node inventory contains duplicate names")
        if len({node.rpc_endpoint for node in nodes}) != len(nodes):
            raise NodeStateError("node inventory contains duplicate endpoints")
        for node in nodes:
            _check_endpoint_security(node.parsed_endpoint, allow_insecure_direct_rpc=self.allow_insecure_direct_rpc)
        return tuple(sorted(nodes, key=lambda node: node.name))

    def save(self, nodes: Iterable[Node]) -> Path:
        entries = tuple(nodes)
        if any(not isinstance(node, Node) for node in entries):
            raise NodeValidationError("node inventory can only persist Node objects")
        if len({node.name for node in entries}) != len(entries):
            raise NodeValidationError("node inventory contains duplicate names")
        if len({node.rpc_endpoint for node in entries}) != len(entries):
            raise NodeValidationError("node inventory contains duplicate endpoints")
        for node in entries:
            _check_endpoint_security(node.parsed_endpoint, allow_insecure_direct_rpc=self.allow_insecure_direct_rpc)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "nodes": [node.as_dict() for node in sorted(entries, key=lambda n: n.name)],
        }
        try:
            write_atomic(self.path, json.dumps(payload, indent=2, sort_keys=True) + "\n")
        except OSError as exc:
            raise NodeStateError(f"unable to write node inventory {self.path}: {exc}") from exc
        return self.path

    def upsert(self, node: Node) -> Node:
        current = {item.name: item for item in self.load()}
        current[node.name] = node
        self.save(current.values())
        return node

    add = upsert

    def remove(self, name: str) -> None:
        current = {item.name: item for item in self.load()}
        if name not in current:
            raise NodeValidationError(f"unknown node: {name}")
        del current[name]
        self.save(current.values())

    def get(self, name: str) -> Node:
        for node in self.load():
            if node.name == name:
                return node
        raise NodeValidationError(f"unknown node: {name}")

    def list(self) -> tuple[Node, ...]:
        """Return the sorted inventory (an alias convenient for CLI callers)."""

        return self.load()


def probe_reachability(
    node_or_endpoint: Node | Endpoint | str,
    *,
    timeout: float = 1.0,
    allow_insecure_direct_rpc: bool = False,
) -> bool:
    """Probe only TCP reachability; no RPC bytes are sent or protocol inferred."""

    if isinstance(node_or_endpoint, Node):
        endpoint = node_or_endpoint.parsed_endpoint
    elif isinstance(node_or_endpoint, Endpoint):
        endpoint = node_or_endpoint
    else:
        endpoint = parse_endpoint(node_or_endpoint)
    _check_endpoint_security(endpoint, allow_insecure_direct_rpc=allow_insecure_direct_rpc)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise NodeValidationError("probe timeout must be positive")
    bounded_timeout = min(float(timeout), MAX_PROBE_TIMEOUT)
    sock: Any = None
    try:
        sock = socket.create_connection((endpoint.host, endpoint.port), timeout=bounded_timeout)
        return True
    except OSError:
        return False
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


probe_node = probe_reachability


@dataclass(frozen=True)
class RpcProbeResult:
    """Evidence returned by :func:`probe_rpc`.

    ``tcp_reachable`` is intentionally separate from ``rpc_protocol``: a
    successful TCP connect never implies that the peer speaks llama.cpp RPC.
    """

    tcp_reachable: bool
    rpc_protocol: bool
    version: tuple[int, int, int] | None = None
    device_count: int | None = None
    error: str | None = None
    # Exact free/total byte readings returned by GET_DEVICE_MEMORY, in device
    # order. These are live session evidence, unlike Node's advertised MiB.
    device_memory: tuple[tuple[int, int], ...] = ()

    @property
    def protocol_compatible(self) -> bool:
        return (
            self.rpc_protocol
            and self.version is not None
            and self.version[0] == RPC_PROTO_MAJOR_VERSION
            and self.version[1] <= RPC_PROTO_MINOR_VERSION
        )

    @property
    def usable(self) -> bool:
        """Whether the peer passed protocol/version/device preflight."""

        return (
            self.tcp_reachable
            and self.protocol_compatible
            and bool(self.device_count)
            and len(self.device_memory) == self.device_count
            and all(free >= 0 and total > 0 and free <= total for free, total in self.device_memory)
        )


def _recv_exact(sock: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise OSError("peer closed the RPC connection")
        if len(chunk) > remaining:
            raise ValueError("peer returned more bytes than requested")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def probe_rpc(
    node_or_endpoint: Node | Endpoint | str,
    *,
    timeout: float = 1.0,
    allow_insecure_direct_rpc: bool = False,
) -> RpcProbeResult:
    """Perform a bounded ggml-rpc HELLO and DEVICE_COUNT probe.

    This sends only the protocol's fixed-size discovery messages.  It does
    not authenticate, launch, or claim that a peer is safe for production.
    """

    if isinstance(node_or_endpoint, Node):
        endpoint = node_or_endpoint.parsed_endpoint
    elif isinstance(node_or_endpoint, Endpoint):
        endpoint = node_or_endpoint
    else:
        endpoint = parse_endpoint(node_or_endpoint)
    _check_endpoint_security(endpoint, allow_insecure_direct_rpc=allow_insecure_direct_rpc)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise NodeValidationError("probe timeout must be positive")
    bounded_timeout = min(float(timeout), MAX_PROBE_TIMEOUT)
    sock: Any = None
    try:
        sock = socket.create_connection((endpoint.host, endpoint.port), timeout=bounded_timeout)
        # ggml-rpc v4.0.0: command, little-endian uint64 payload length,
        # followed by 24 capability bytes. Zero capabilities avoid transport
        # negotiation and keep this probe limited to ordinary TCP.
        sock.sendall(bytes((14,)) + struct.pack("<Q", RPC_CONN_CAPS_SIZE) + bytes(RPC_CONN_CAPS_SIZE))
        hello_size = struct.unpack("<Q", _recv_exact(sock, 8))[0]
        if hello_size != 28:
            raise ValueError(f"unexpected HELLO response size {hello_size}")
        hello = _recv_exact(sock, 4 + RPC_CONN_CAPS_SIZE)
        version = (hello[0], hello[1], hello[2])
        sock.sendall(bytes((15,)) + struct.pack("<Q", 0))
        device_size = struct.unpack("<Q", _recv_exact(sock, 8))[0]
        if device_size != 4:
            raise ValueError(f"unexpected DEVICE_COUNT response size {device_size}")
        device_count = struct.unpack("<I", _recv_exact(sock, 4))[0]
        if device_count > MAX_RPC_DEVICES:
            raise ValueError(f"RPC worker reports too many devices ({device_count}; maximum {MAX_RPC_DEVICES})")
        device_memory: list[tuple[int, int]] = []
        for device in range(device_count):
            # GET_DEVICE_MEMORY is command 11 in the pinned ggml-rpc v4
            # command table (command 10 is GRAPH_COMPUTE).
            sock.sendall(bytes((11,)) + struct.pack("<Q", 4) + struct.pack("<I", device))
            memory_size = struct.unpack("<Q", _recv_exact(sock, 8))[0]
            if memory_size != 16:
                raise ValueError(f"unexpected GET_DEVICE_MEMORY response size {memory_size}")
            device_memory.append(struct.unpack("<QQ", _recv_exact(sock, 16)))
        return RpcProbeResult(True, True, version, device_count, device_memory=tuple(device_memory))
    except (OSError, ValueError, struct.error) as exc:
        if sock is None:
            return RpcProbeResult(False, False, error=str(exc))
        return RpcProbeResult(True, False, error=str(exc))
    finally:
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass


@dataclass(frozen=True)
class EngineProvenance:
    version: str | None = None
    commit: str | None = None


@dataclass(frozen=True)
class PlacementPlan:
    rpc_endpoints: tuple[str, ...]
    tensor_split: tuple[float, ...]
    capacities_mib: tuple[Real, ...]
    total_capacity_mib: Real
    device_names: tuple[str, ...]
    reachable_nodes: tuple[str, ...]

    @property
    def tensor_split_arg(self) -> str:
        return ",".join(format(value, ".6g") for value in self.tensor_split)


def _expected_provenance(
    value: EngineProvenance | Mapping[str, str | None] | tuple[str, str] | str | None,
    *,
    version: str | None,
    commit: str | None,
) -> EngineProvenance | None:
    if value is not None:
        if isinstance(value, EngineProvenance):
            version, commit = value.version, value.commit
        elif isinstance(value, Mapping):
            version, commit = value.get("version"), value.get("commit")
        elif isinstance(value, tuple) and len(value) == 2:
            version, commit = value
        elif isinstance(value, str):
            commit = value
            version = None
        else:
            raise NodeValidationError(
                "expected_engine must be EngineProvenance, mapping, (version, commit), or commit string"
            )
    if version is None and commit is None:
        return None
    return EngineProvenance(version, commit)


def _capacity_sequence(value: Real | Iterable[Real], *, field_name: str) -> tuple[Real, ...]:
    if isinstance(value, Real) and not isinstance(value, bool):
        values = (value,)
    else:
        if isinstance(value, (str, bytes)):
            raise NodeValidationError(f"{field_name} must be a finite non-negative number or sequence")
        try:
            values = tuple(value)
        except TypeError as exc:
            raise NodeValidationError(f"{field_name} must be a finite non-negative number or sequence") from exc
    if any(
        isinstance(item, bool) or not isinstance(item, Real) or not math.isfinite(item) or item < 0 for item in values
    ):
        raise NodeValidationError(f"{field_name} must contain finite non-negative numbers")
    return values


def plan_placement(
    local_free_vram_mib: Real | Iterable[Real],
    nodes: Iterable[Node] = (),
    *,
    reachable: Mapping[str, bool] | None = None,
    probe: Callable[[Node], bool] = probe_node,
    timeout: float = 1.0,
    allow_insecure_direct_rpc: bool = False,
    expected_engine: EngineProvenance | Mapping[str, str | None] | tuple[str, str] | str | None = None,
    expected_engine_version: str | None = None,
    expected_engine_commit: str | None = None,
    local_engine_version: str | None = None,
    local_engine_commit: str | None = None,
    rpc_probe_results: Mapping[str, RpcProbeResult] | None = None,
) -> PlacementPlan:
    """Build a deterministic local-plus-remote capacity placement.

    Local devices come first in probe order; subsequent entries correspond to
    every device on sorted, enabled, reachable remote nodes. ``rpc_endpoints``
    remains one entry per server. ``reachable`` is an optional name/endpoint
    keyed result map for callers that already performed probing.
    """

    local_capacities = _capacity_sequence(local_free_vram_mib, field_name="local_free_vram_mib")
    expected = _expected_provenance(expected_engine, version=expected_engine_version, commit=expected_engine_commit)
    candidates = tuple(nodes)
    if any(not isinstance(node, Node) for node in candidates):
        raise NodeValidationError("placement nodes must be Node objects")
    if len({node.name for node in candidates}) != len(candidates):
        raise NodeValidationError("placement nodes contain duplicate names")
    if len({node.rpc_endpoint for node in candidates}) != len(candidates):
        raise NodeValidationError("placement nodes contain duplicate endpoints")
    selected: list[Node] = []
    live_memory: dict[str, tuple[Real, ...]] = {}
    for node in sorted(candidates, key=lambda item: item.name):
        _check_endpoint_security(node.parsed_endpoint, allow_insecure_direct_rpc=allow_insecure_direct_rpc)
        if not node.enabled:
            continue
        live = (
            rpc_probe_results.get(node.name) or rpc_probe_results.get(node.rpc_endpoint) if rpc_probe_results else None
        )
        if live is not None:
            is_reachable = live.tcp_reachable
        elif reachable is not None:
            is_reachable = reachable.get(node.name, reachable.get(node.rpc_endpoint, False))
        elif probe is probe_node:
            is_reachable = probe(node, timeout=timeout, allow_insecure_direct_rpc=allow_insecure_direct_rpc)
        else:
            is_reachable = probe(node)
        if live is not None:
            if not live.usable:
                continue
            # Preserve one capacity entry per enumerated RPC device, including
            # a temporarily full device with zero free bytes. Removing it
            # would shift subsequent tensor-split ratios onto the wrong device.
            live_values = tuple(free / 1024**2 for free, _total in live.device_memory)
            live_memory[node.name] = live_values
        if not is_reachable:
            continue
        if live is None and node.accelerator_memory_mib <= 0:
            continue
        if expected is not None and (
            (expected.version is not None and node.engine_version != expected.version)
            or (expected.commit is not None and node.engine_commit != expected.commit)
        ):
            raise NodePlanningError(
                f"node {node.name!r} has incompatible engine provenance",
                hint="Use nodes built from the same pinned llama.cpp version and commit.",
            )
        selected.append(node)
    if (
        expected is not None
        and sum(local_capacities) > 0
        and (
            (expected.version is not None and local_engine_version != expected.version)
            or (expected.commit is not None and local_engine_commit != expected.commit)
        )
    ):
        raise NodePlanningError("local engine provenance is incompatible with the expected engine")
    # Keep zero-free devices in the split with a zero ratio. llama.cpp expects
    # one entry for every enumerated local and remote device; dropping a full
    # device shifts every following ratio onto the wrong backend.
    capacities: list[Real] = list(local_capacities)
    device_names: list[str] = [f"local:{index}" for index in range(len(local_capacities))]
    for node in selected:
        node_capacities = live_memory.get(node.name, (node.accelerator_memory_mib,))
        capacities.extend(node_capacities)
        device_names.extend(f"{node.name}:{index}" for index in range(len(node_capacities)))
    if not capacities or sum(capacities) <= 0:
        raise NodePlanningError("placement has zero accelerator capacity")
    total = sum(capacities)
    ratios = [round(capacity / total, 6) for capacity in capacities]
    if len(ratios) > 1:
        ratios[-1] = round(1.0 - sum(ratios[:-1]), 6)
    return PlacementPlan(
        rpc_endpoints=tuple(node.rpc_endpoint for node in selected),
        tensor_split=tuple(ratios),
        capacities_mib=tuple(capacities),
        total_capacity_mib=total,
        device_names=tuple(device_names),
        reachable_nodes=tuple(node.name for node in selected),
    )


def resolve_node_plan(
    *,
    names: Iterable[str] = (),
    selector: str | None = None,
    allow_insecure_rpc: bool = False,
    store: NodeStore | None = None,
    local_free_vram_mib: Real | Iterable[Real] = 0,
    local_engine_version: str | None = None,
    local_engine_commit: str | None = None,
    expected_engine: EngineProvenance | Mapping[str, str | None] | tuple[str, str] | str | None = None,
    timeout: float = 1.0,
) -> dict[str, Any]:
    """Resolve selected inventory entries using RPC protocol preflight.

    Unlike :func:`plan_placement`'s intentionally transport-only default,
    this CLI-facing resolver verifies the ggml-rpc handshake and every remote
    device's live free memory before returning an actionable plan.
    """

    inventory_store = store or NodeStore(allow_insecure_direct_rpc=allow_insecure_rpc)
    inventory = inventory_store.load()
    requested = [name for name in names if name]
    if selector and selector != "all":
        requested.extend(item.strip() for item in selector.split(",") if item.strip())
    if selector == "all":
        selected = list(inventory)
    else:
        wanted = set(requested)
        selected = [item for item in inventory if item.name in wanted]
        missing = [name for name in requested if name not in {item.name for item in selected}]
        if missing:
            raise NodePlanningError(f"unknown node(s): {', '.join(missing)}")
    if not selected:
        raise NodePlanningError("no RPC nodes were selected", hint="Add an enabled node with `kestrel nodes add`.")
    if not any(item.enabled for item in selected):
        raise NodePlanningError("selected inventory contains no enabled RPC nodes")
    results: dict[str, RpcProbeResult] = {}
    failures: list[str] = []
    for item in selected:
        if not item.enabled:
            if selector != "all":
                failures.append(f"{item.name}: node is disabled")
            continue
        result = probe_rpc(item, timeout=timeout, allow_insecure_direct_rpc=allow_insecure_rpc)
        results[item.name] = result
        if not result.usable:
            if result.error:
                detail = result.error
            elif not result.protocol_compatible:
                detail = f"incompatible RPC protocol {result.version!r}"
            elif not result.device_count:
                detail = "RPC worker exposes zero devices"
            else:
                detail = "RPC device memory probe was incomplete"
            failures.append(f"{item.name}: {detail}")
    if failures:
        raise NodePlanningError(
            "; ".join(failures), hint="Start a compatible llama.cpp RPC worker and verify its endpoint."
        )
    plan = plan_placement(
        local_free_vram_mib,
        selected,
        allow_insecure_direct_rpc=allow_insecure_rpc,
        expected_engine=expected_engine,
        local_engine_version=local_engine_version,
        local_engine_commit=local_engine_commit,
        rpc_probe_results=results,
    )
    evidence = {
        name: {
            "tcp_reachable": result.tcp_reachable,
            "rpc_protocol": result.rpc_protocol,
            "protocol_compatible": result.protocol_compatible,
            "device_count": result.device_count,
            "device_memory": result.device_memory,
        }
        for name, result in sorted(results.items())
    }
    active = [item for item in selected if item.enabled]
    return {
        "status": "planned",
        "nodes": [dict(item.as_dict(), rpc_endpoint=item.rpc_endpoint) for item in active],
        "rpc_endpoints": list(plan.rpc_endpoints),
        "tensor_split": plan.tensor_split_arg,
        "tensor_split_ratios": plan.tensor_split,
        "device_order": list(plan.device_names),
        "capacities_mib": list(plan.capacities_mib),
        "total_capacity_mib": plan.total_capacity_mib,
        "total_devices": len(plan.tensor_split),
        "probe_evidence": evidence,
    }


__all__ = [
    "Endpoint",
    "EngineProvenance",
    "MAX_PROBE_TIMEOUT",
    "MAX_RPC_DEVICES",
    "Node",
    "NodeError",
    "NodePlanningError",
    "NodeSecurityError",
    "NodeStateError",
    "NodeStore",
    "NodeValidationError",
    "PlacementPlan",
    "RPC_CONN_CAPS_SIZE",
    "RPC_PROTO_MAJOR_VERSION",
    "RPC_PROTO_MINOR_VERSION",
    "RPC_PROTO_PATCH_VERSION",
    "RpcProbeResult",
    "format_endpoint",
    "is_loopback_host",
    "node_store_path",
    "parse_endpoint",
    "plan_placement",
    "probe_node",
    "probe_rpc",
    "resolve_node_plan",
    "probe_reachability",
]
