"""
Chuong 5 - Dao ham so.

P1: diff_forward, diff_backward   (cong thuc 5.1, 5.2, 5.4)
P2: diff_central                  (cong thuc 5.3)
P3: diff2_central                 (dao ham cap 2, sai phan trung tam)
P4: diff_lagrange, diff_table     (dao ham da thuc noi suy Lagrange tai nut)
"""

from math import isfinite
from numbers import Real
from typing import Sequence

import numpy as np


def _check_step(h: float) -> None:
    if h <= 0:
        raise ValueError("h phai lon hon 0")


def _check_callable(f) -> None:
    if not callable(f):
        raise TypeError("f phai la mot ham so co the goi")


# ---------------------------------------------------------------- P1
def diff_forward(f, x, h, order=1) -> float:
    """Sai phan tien. order=1: 2 diem O(h); order=2: 3 diem O(h^2)."""
    _check_callable(f)
    _check_step(h)
    if order == 1:
        return (f(x + h) - f(x)) / h
    if order == 2:
        return (-3 * f(x) + 4 * f(x + h) - f(x + 2 * h)) / (2 * h)
    raise ValueError("order chi nhan gia tri 1 hoac 2")


def diff_backward(f, x, h, order=1) -> float:
    """Sai phan lui. order=1: 2 diem O(h); order=2: 3 diem O(h^2)."""
    _check_callable(f)
    _check_step(h)
    if order == 1:
        return (f(x) - f(x - h)) / h
    if order == 2:
        return (3 * f(x) - 4 * f(x - h) + f(x - 2 * h)) / (2 * h)
    raise ValueError("order chi nhan gia tri 1 hoac 2")


# ---------------------------------------------------------------- P2
def diff_central(f, x, h) -> float:
    """Sai phan trung tam: f'(x) ~ (f(x+h) - f(x-h)) / (2h), sai so O(h^2)."""
    _check_callable(f)
    _check_step(h)
    return (f(x + h) - f(x - h)) / (2 * h)


# ---------------------------------------------------------------- P3
def diff2_central(f, x, h) -> float:
    """Dao ham cap 2: f''(x) ~ (f(x+h) - 2f(x) + f(x-h)) / h^2, sai so O(h^2)."""
    _check_callable(f)
    _check_step(h)
    return (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2


# ---------------------------------------------------------------- P4
def diff_lagrange(xs: Sequence[float], ys: Sequence[float], k: int) -> float:
    """
    Dao ham cua da thuc noi suy Lagrange tai nut xs[k] (moc khong can cach deu).

    P'(x_k) = sum_i y_i * L_i'(x_k), trong do
        L_k'(x_k) = sum_{j != k} 1 / (x_k - x_j)
        L_i'(x_k) = prod_{j != i,k}(x_k - x_j) / prod_{j != i}(x_i - x_j),  i != k
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k phai la so nguyen.")

    try:
        x_values = list(xs)
        y_values = list(ys)
    except TypeError as exc:
        raise TypeError("xs va ys phai la cac day co the lap.") from exc

    if len(x_values) != len(y_values):
        raise ValueError("xs va ys phai co cung do dai.")

    n = len(x_values)
    if n < 2:
        raise ValueError("Can it nhat 2 nut noi suy de tinh dao ham.")
    if not 0 <= k < n:
        raise ValueError("k nam ngoai pham vi chi so cua xs.")

    for name, values in (("xs", x_values), ("ys", y_values)):
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"Moi phan tu cua {name} phai la so thuc.")
            if not isfinite(float(value)):
                raise ValueError(f"Moi phan tu cua {name} phai la so huu han.")

    x_values = [float(v) for v in x_values]
    y_values = [float(v) for v in y_values]

    for i in range(n):
        for j in range(i + 1, n):
            if x_values[i] == x_values[j]:
                raise ValueError("Cac moc noi suy xs phai doi mot khac nhau.")

    xk = x_values[k]

    # Dong gop cua y_k * L_k'(x_k)
    derivative = y_values[k] * sum(1.0 / (xk - x_values[j]) for j in range(n) if j != k)

    # Dong gop cua y_i * L_i'(x_k), i != k
    for i in range(n):
        if i == k:
            continue
        numerator = 1.0
        denominator = 1.0
        for j in range(n):
            if j != i:
                denominator *= x_values[i] - x_values[j]
                if j != k:
                    numerator *= xk - x_values[j]
        derivative += y_values[i] * numerator / denominator

    return float(derivative)


def diff_table(xs, ys) -> np.ndarray:
    """Dao ham tai moi nut cua bang so lieu (qua da thuc noi suy Lagrange)."""
    return np.array([diff_lagrange(xs, ys, k) for k in range(len(xs))])
