"""Shared helpers for neural observation/action encoders."""

from __future__ import annotations


DIRECTIONS = ("N", "E", "S", "W")
CONNECTION_PAIRS = ("NE", "NS", "NW", "ES", "EW", "SW")


def normalize_range(value: int, minimum: int, maximum: int) -> float:
    if maximum == minimum:
        return 0.0
    clipped = min(max(value, minimum), maximum)
    return (2.0 * (clipped - minimum) / float(maximum - minimum)) - 1.0


def normalize_count(value: int, maximum: int) -> float:
    if maximum <= 0:
        return 0.0
    return min(max(value, 0), maximum) / float(maximum)


def rotation_or_zero(value: object) -> int:
    return value if isinstance(value, int) else 0


def validate_rotation(rotation: int) -> None:
    if rotation % 180 != 0:
        raise ValueError("Base Saboteur encoders support only 0 or 180 degree rotation")


def rotate_edge(edge: str, rotation: int) -> str:
    validate_rotation(rotation)
    if edge not in DIRECTIONS or rotation % 360 != 180:
        return edge
    opposite = {"N": "S", "E": "W", "S": "N", "W": "E"}
    return opposite[edge]


def rotated_edges_from_card(card: object, rotation: int) -> set[str]:
    validate_rotation(rotation)
    if not isinstance(card, dict):
        return set()
    edges = card.get("edges", [])
    if not isinstance(edges, list):
        return set()
    return {
        rotate_edge(edge, rotation)
        for edge in edges
        if isinstance(edge, str) and edge in DIRECTIONS
    }


def rotated_groups_from_card(card: object, rotation: int) -> tuple[frozenset[str], ...]:
    validate_rotation(rotation)
    if not isinstance(card, dict):
        return ()
    groups = card.get("groups", [])
    if not isinstance(groups, list):
        return ()
    rotated_groups: list[frozenset[str]] = []
    for raw_group in groups:
        if not isinstance(raw_group, list):
            continue
        rotated_groups.append(
            frozenset(
                rotate_edge(edge, rotation)
                for edge in raw_group
                if isinstance(edge, str) and edge in DIRECTIONS
            )
        )
    return tuple(group for group in rotated_groups if group)


def connection_pairs_from_card(card: object, rotation: int) -> set[str]:
    validate_rotation(rotation)
    pairs: set[str] = set()
    for raw_edges in rotated_groups_from_card(card, rotation):
        edges = sorted(raw_edges, key=DIRECTIONS.index)
        for left_index, left in enumerate(edges):
            for right in edges[left_index + 1 :]:
                pair = "".join(sorted((left, right), key=DIRECTIONS.index))
                if pair in CONNECTION_PAIRS:
                    pairs.add(pair)
    return pairs
