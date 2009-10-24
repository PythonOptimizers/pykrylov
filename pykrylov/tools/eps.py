def machine_epsilon():
    """
    Compute machine epsilon: smallest number e > 0 such that (1 + e) > 1.
    """
    one = 1.0
    eps = 1.0
    while (one + eps) > one:
        eps = eps / 2.0
    return eps*2.0
