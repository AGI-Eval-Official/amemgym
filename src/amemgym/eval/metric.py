# METRICS = ["accuracy", "jaccard", "hamming"]
METRICS = ["accuracy"]


def state_similarity(state1: list, state2: list, metric: str = "accuracy") -> float:
    """Calculate similarity between two states using the specified metric.

    Args:
        state1: First state as a list of values.
        state2: Second state as a list of values.
        metric: Similarity metric to use. Options are "accuracy", "hamming", or "jaccard".
            Defaults to "accuracy".

    Returns:
        Similarity score as a float between 0 and 1.

    Raises:
        AssertionError: If states have different lengths.
        ValueError: If an invalid metric is specified.
    """
    assert len(state1) == len(state2), "States must have the same length."

    num_matched = sum(s1 == s2 for s1, s2 in zip(state1, state2))

    match metric:
        case "accuracy":
            return float(num_matched == len(state1))
        case "hamming":
            return num_matched / len(state1)
        case "jaccard":
            return num_matched / (len(state1) * 2 - num_matched)
        case _:
            raise ValueError(f"Invalid metric: {metric}")
