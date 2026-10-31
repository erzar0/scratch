"""Methods that change softmax attention's "read every key" step, per head.

Each is a drop-in for one head's softmax, f(q, k, v) -> out, used through
softmax.approx_attention (ForwardOptions.head_attention).
"""
