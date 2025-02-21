def broadcast_axis(left, right):
    left_dim = len(left)
    right_dim = len(right)
    maxdim = max(left_dim, right_dim)

    left_new_shape = (1, ) * (maxdim - left_dim) + left
    right_new_shape = (1, ) * (maxdim - right_dim) + right

    assert len(left_new_shape) == len(right_new_shape)

    left_axes, right_axes = [], []

    for i in range(len(left_new_shape)):
        if left_new_shape[i] > right_new_shape[i]:
            right_axes.append(i)
        elif right_new_shape[i] > left_new_shape[i]:
            left_axes.append(i)

    return tuple(left_axes), tuple(right_axes)