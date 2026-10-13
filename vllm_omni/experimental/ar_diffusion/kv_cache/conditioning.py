# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request-scoped validity around the existing AR-Diffusion conditioning pool."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING
from uuid import uuid4

import torch

if TYPE_CHECKING:
    from vllm_omni.experimental.ar_diffusion.kv_cache.manager import ARDiffusionKVCache


@dataclass(frozen=True, slots=True)
class ConditioningKVRepresentation:
    dtype: torch.dtype
    device: torch.device
    layout: str
    kv_length: int
    local_heads: int
    head_size: int


@dataclass(frozen=True, slots=True)
class ConditioningKVIdentity:
    cache_name: str
    kv_branch: str
    condition_revision: Hashable
    projection_revision: Hashable
    representation: ConditioningKVRepresentation

    def __post_init__(self) -> None:
        hash(self.condition_revision)
        hash(self.projection_revision)


@dataclass(frozen=True, slots=True)
class ConditioningKVBuildTicket:
    scope_token: str
    identity: ConditioningKVIdentity
    generation: int


class ConditioningKVScope:
    """Validate complete branch K/V without allocating a second storage pool.

    The manager's storage key contains a fresh incarnation, so recycling an
    external request ID cannot make a new request observe an older allocation.
    A ticket is checked at the final manager publication boundary, after every
    projection and copy has been queued. A CUDA event makes those writes visible
    to later borrowers on another stream. GPU borrowers must treat returned K/V
    as read only; ``record_stream`` protects their allocations on the current
    consumer stream when the scope closes before queued work finishes.
    """

    def __init__(self, kv_cache: ARDiffusionKVCache, session_id: str, num_layers: int) -> None:
        self.kv_cache = kv_cache
        self.session_id = session_id
        self.num_layers = num_layers
        self.scope_token = uuid4().hex
        self._storage_session_id = f"conditioning::{self.scope_token}"
        self._lock = RLock()
        self._generations: dict[tuple[str, str], int] = {}
        self._identities: dict[tuple[str, str], ConditioningKVIdentity] = {}
        self._ready_events: dict[tuple[str, str], torch.cuda.Event] = {}
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"Conditioning K/V scope for {self.session_id!r} is closed")

    def lookup(self, identity: ConditioningKVIdentity) -> list[dict[str, torch.Tensor | bool]] | None:
        key = (identity.cache_name, identity.kv_branch)
        with self._lock:
            self._ensure_open()
            if self._identities.get(key) != identity:
                return None
            if not self.kv_cache.is_cross_attention_populated(
                self._storage_session_id, identity.cache_name, identity.kv_branch
            ):
                self._identities.pop(key, None)
                self._ready_events.pop(key, None)
                return None
            ready_event = self._ready_events.get(key)
            if ready_event is not None:
                torch.cuda.current_stream(self.kv_cache.device).wait_event(ready_event)
            layers = [
                self.kv_cache.read_cross_attention_kv(
                    self._storage_session_id, identity.cache_name, layer, identity.kv_branch
                )
                for layer in range(self.num_layers)
            ]
            for layer in layers:
                for name in ("k", "v"):
                    tensor = layer[name]
                    assert isinstance(tensor, torch.Tensor)
                    if tensor.is_cuda:
                        tensor.record_stream(torch.cuda.current_stream(tensor.device))
            return layers

    def begin_build(self, identity: ConditioningKVIdentity) -> ConditioningKVBuildTicket:
        with self._lock:
            self._ensure_open()
            key = (identity.cache_name, identity.kv_branch)
            # A changed condition or projection retires the previous complete
            # value immediately, even if building its replacement later fails.
            if self._identities.get(key) not in (None, identity):
                self._identities.pop(key, None)
                self._ready_events.pop(key, None)
                self.kv_cache.release_cross_attention_branch(self._storage_session_id, *key)
            # A newer build supersedes an older in-flight build, even if both
            # target the same branch before either has been published.
            generation = self._generations.get(key, 0) + 1
            self._generations[key] = generation
            return ConditioningKVBuildTicket(self.scope_token, identity, generation)

    def publish(
        self,
        ticket: ConditioningKVBuildTicket,
        layer_kv: Iterable[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        identity = ticket.identity
        key = (identity.cache_name, identity.kv_branch)

        @contextmanager
        def publication_guard() -> Iterator[None]:
            with self._lock:
                self._ensure_open()
                if ticket.scope_token != self.scope_token or self._generations.get(key) != ticket.generation:
                    raise RuntimeError("Conditioning K/V build was invalidated before publication")
                yield
                try:
                    if self.kv_cache.device.type == "cuda":
                        event = torch.cuda.Event()
                        event.record(torch.cuda.current_stream(self.kv_cache.device))
                        self._ready_events[key] = event
                    else:
                        self._ready_events.pop(key, None)
                except Exception:
                    # The manager has inserted the new pool. Never expose it
                    # without a readiness fence, even after a same-key rebuild.
                    self._identities.pop(key, None)
                    self._ready_events.pop(key, None)
                    self.kv_cache.release_cross_attention_branch(self._storage_session_id, *key)
                    raise
                self._identities[key] = identity

        with self._lock:
            self._ensure_open()
            if ticket.scope_token != self.scope_token or self._generations.get(key) != ticket.generation:
                raise RuntimeError("Conditioning K/V build was invalidated before publication")
        self.kv_cache.populate_cross_attention(
            self._storage_session_id,
            identity.cache_name,
            identity.kv_branch,
            layer_kv,
            publish_guard=publication_guard,
        )

    def invalidate(self, cache_name: str | None = None, kv_branch: str | None = None) -> None:
        with self._lock:
            self._ensure_open()
            keys = set(self._generations) | set(self._identities)
            selected = {
                key
                for key in keys
                if (cache_name is None or key[0] == cache_name) and (kv_branch is None or key[1] == kv_branch)
            }
            for key in selected:
                self._generations[key] = self._generations.get(key, 0) + 1
                self._identities.pop(key, None)
                self._ready_events.pop(key, None)
            if cache_name is None and kv_branch is None:
                self.kv_cache.release_cross_attention(self._storage_session_id)
            else:
                for name, branch in selected:
                    self.kv_cache.release_cross_attention_branch(self._storage_session_id, name, branch)

    def retain(self, cache_names: Iterable[str]) -> None:
        keep = set(cache_names)
        with self._lock:
            self._ensure_open()
            names = {name for name, _ in self._generations} | {name for name, _ in self._identities}
        for name in names - keep:
            self.invalidate(cache_name=name)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._identities.clear()
            self._ready_events.clear()
            self.kv_cache.release_cross_attention(self._storage_session_id)
