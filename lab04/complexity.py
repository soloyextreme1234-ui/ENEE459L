from __future__ import annotations

from typing import Any

from graph import (
    Graph,
    Layer,
    computed,
    dtype_bytes,
    is_answered,
    unknown,
)


FLOPS_PER_MAC = 2

# The conventions `to_flops` will honour by name. Anything else is unknown
# rather than an assumption, because the whole point of the parameter is that
# the caller has to say which one they mean.
FLOP_CONVENTIONS = {
    "mac_is_two_flops": 2,
    "mac_is_one_flop": 1,
}

# Batch normalisation holds two learnable vectors per channel (scale and shift)
# and two non-learnable ones (running mean and variance). The first pair are
# parameters; the second pair are buffers. Both are in the file.
BN_PARAMS_PER_CHANNEL = 2
BN_BUFFERS_PER_CHANNEL = 2

# Buffers are kept in FP32 even when the weights are not. Halving them saves
# nothing worth having and a denormal running variance is a real failure mode.
BUFFER_DTYPE = "fp32"

# Below this many models there is no line to fit and no residual to report.
MIN_MODELS_FOR_FIT = 3

# Two floats are the same MAC count when they are the same integer. There is no
# tolerance here on purpose: MAC counts are integers, and a tolerance would let
# two genuinely different architectures be reported as tied.
TIE_EXACT = True

# ===========================================================================
# 1. How many numbers are stored
# ===========================================================================

def _layer_parameters(ly: Layer) -> int:
    """Return the number of learnable parameters in one layer."""

    if ly.kind == "conv":
        c_out = ly.out_shape[0]
        c_in = ly.in_shape[0]

        kh, kw = ly.kernel if ly.kernel is not None else (1, 1)

        nweights = c_out * (c_in // ly.groups) * kh * kw

        if ly.bias:
            nweights += c_out

        return nweights

    if ly.kind == "linear":
        f_out = ly.out_shape[0]
        f_in = ly.in_shape[0]

        nweights = f_out * f_in

        if ly.bias:
            nweights += f_out

        return nweights

    if ly.kind == "bn":
        channels = ly.out_shape[0]
        return BN_PARAMS_PER_CHANNEL * channels

    return 0


def count_parameters(graph: Graph) -> dict[str, Any]:
    per_layer: dict[str, int] = {}
    total = 0

    for ly in graph:
        n = _layer_parameters(ly)

        per_layer[ly.name] = n
        total += n

    return computed(
        total,
        source=f"{graph.name}: {len(graph)} layers, shapes from the description",
        per_layer=per_layer,
        includes_bias=True,
        excludes_bn_buffers=True,
        bn_params_per_channel=BN_PARAMS_PER_CHANNEL,
    )

# ===========================================================================
# 2. What those numbers weigh
# ===========================================================================

def model_size_bytes(graph: Graph) -> dict[str, Any]:
    per_dtype: dict[str, float] = {}
    per_layer: dict[str, float] = {}

    buffer_bytes = 0.0

    for ly in graph:
        n = _layer_parameters(ly)

        param_bytes = n * dtype_bytes(ly.weight_dtype)

        per_dtype[ly.weight_dtype] = (
            per_dtype.get(ly.weight_dtype, 0.0) + param_bytes
        )

        layer_bytes = param_bytes

        if ly.kind == "bn":
            buffer_elements = BN_BUFFERS_PER_CHANNEL * ly.out_shape[0]

            bn_buffer_bytes = (
                buffer_elements * dtype_bytes(BUFFER_DTYPE)
            )

            buffer_bytes += bn_buffer_bytes
            layer_bytes += bn_buffer_bytes

            per_dtype[BUFFER_DTYPE] = (
                per_dtype.get(BUFFER_DTYPE, 0.0)
                + bn_buffer_bytes
            )

        per_layer[ly.name] = layer_bytes

    total = sum(per_layer.values())

    return computed(
        total,
        source=f"{graph.name}: per-layer dtypes, buffers at fp32",
        per_layer=per_layer,
        per_dtype=per_dtype,
        buffer_bytes=buffer_bytes,
        container_overhead_excluded=True,
        note="not the size of the file on disk; see the handout, Stage A step 3",
    )


# ===========================================================================
# 3. Activation memory
# ===========================================================================

def _elements(shape: tuple[int, ...]) -> int:
    total = 1

    for dimension in shape:
        total *= dimension

    return total


def _last_use(graph: Graph) -> dict[str, int]:
    last: dict[str, int] = {}

    names = [ly.name for ly in graph.layers]

    for i, ly in enumerate(graph.layers):

        if ly.reads:
            for tensor_name in ly.reads:
                last[tensor_name] = i

        elif i == 0:
            last["__input__"] = 0

        else:
            last[names[i - 1]] = i

        last.setdefault(ly.name, i)

    if graph.layers:
        last[graph.layers[-1].name] = len(graph) - 1

    return last


def _peak_elements(
    graph: Graph,
    last_use: dict[str, int],
) -> int:

    live: dict[str, int] = {
        "__input__": _elements(graph.input_shape)
    }

    peak = sum(live.values())

    for i, ly in enumerate(graph.layers):

        live[ly.name] = ly.out_elements

        current = sum(live.values())

        if current > peak:
            peak = current

        for tensor_name in list(live.keys()):
            if last_use.get(tensor_name) == i:
                del live[tensor_name]

    return peak

def count_activations(graph: Graph) -> dict[str, Any]:
    last_use = _last_use(graph)

    live: dict[str, float] = {
        "__input__":
            _elements(graph.input_shape)
            * dtype_bytes(graph.precision)
    }

    total_elements = 0
    total_bytes = 0.0

    peak_bytes = sum(live.values())
    peak_at = "__input__"

    for i, ly in enumerate(graph.layers):

        out_elements = ly.out_elements

        out_b = out_elements * dtype_bytes(ly.act_dtype)

        live[ly.name] = out_b

        total_elements += out_elements
        total_bytes += out_b

        resident = sum(live.values())

        if resident > peak_bytes:
            peak_bytes = resident
            peak_at = ly.name

        for tensor_name in list(live.keys()):
            if last_use.get(tensor_name) == i:
                del live[tensor_name]

    peak_elements = _peak_elements(graph, last_use)

    return computed(
        peak_bytes,
        source=f"{graph.name}: liveness over {len(graph)} layers, input included",
        peak_at=peak_at,
        peak_elements=peak_elements,
        total_elements=total_elements,
        total_bytes=total_bytes,
        includes_network_input=True,
        note="peak is the resident set, not the largest single tensor",
    )

# ===========================================================================
# 4. MAC -> FLOP conversion
# ===========================================================================

def to_flops(
    macs: dict[str, Any],
    convention: str = "mac_is_two_flops",
) -> dict[str, Any]:

    if not is_answered(macs):
        return unknown(
            "MAC to FLOP conversion",
            "no valid MAC count was provided",
        )

    if convention not in FLOP_CONVENTIONS:
        return unknown(
            macs.get("source", "MAC to FLOP conversion"),
            (
                f"unrecognised FLOP convention {convention!r}; "
                f"allowed: {sorted(FLOP_CONVENTIONS)}"
            ),
        )

    factor = FLOP_CONVENTIONS[convention]

    total_flops = macs["value"] * factor

    per_layer: dict[str, Any] = {}

    if isinstance(macs.get("per_layer"), dict):
        per_layer = {
            name: value * factor
            for name, value in macs["per_layer"].items()
        }

    return computed(
        total_flops,
        source=macs.get("source", "MAC count"),
        convention=convention,
        flops_per_mac=factor,
        per_layer=per_layer,
        note="a count of operations contains no unit of time",
    )