import numpy as np
import pytest

# IMPORT CÁC HÀM ĐẠO HÀM TỪ FILE THUẬT TOÁN CỦA NHÓM
from thuat_toan_nhom5 import (
    diff_forward,
    diff_backward,
    diff_central,
    diff2_central,
    diff_lagrange,
    diff_table
)


# ==============================================================================
# 1. KIỂM THỬ BẮT BUỘC: HÀM MẪU f(x) = exp(-x^2) TẠI x = 0.5
# ==============================================================================
def f_sample(x):
    """f(x) = exp(-x^2)"""
    return np.exp(-x**2)

def test_daoham_r7_bat_buoc_sample_func():
    """[R7 - Kiểm thử bắt buộc]: f(x) = e^(-x^2) tại x = 0.5.
    f'(0.5) = f''(0.5) = -e^(-1/4) ≈ -0.778801[cite: 4].
    """
    print("\n--- [R7: Kiểm thử hàm mẫu bắt buộc f(x) = exp(-x^2) tại x = 0.5] ---")
    x0 = 0.5
    h = 1e-4
    target_val = -np.exp(-0.25) # ≈ -0.7788007830714049
    
    # 1. Đạo hàm cấp 1 (Tiến, Lùi, Trung tâm)
    df_fwd = diff_forward(f_sample, x0, h, order=1)
    df_bwd = diff_backward(f_sample, x0, h, order=1)
    df_ctr = diff_central(f_sample, x0, h)
    
    # 2. Đạo hàm cấp 2 (Trung tâm)
    d2f_ctr = diff2_central(f_sample, x0, h)
    
    print(f"  + Giá trị lý thuyết: f'(0.5) = f''(0.5) = {target_val:.6f}")
    print(f"  + Sai phân tiến (O(h)):   {df_fwd:.6f} | Sai số = {abs(df_fwd - target_val):.2e}")
    print(f"  + Sai phân lùi (O(h)):    {df_bwd:.6f} | Sai số = {abs(df_bwd - target_val):.2e}")
    print(f"  + Sai phân trung tâm:      {df_ctr:.6f} | Sai số = {abs(df_ctr - target_val):.2e}")
    print(f"  + Đạo hàm cấp 2 trung tâm: {d2f_ctr:.6f} | Sai số = {abs(d2f_ctr - target_val):.2e}")

    assert pytest.approx(df_ctr, abs=1e-5) == target_val
    assert pytest.approx(d2f_ctr, abs=1e-4) == target_val


# ==============================================================================
# 2. KIỂM THỬ BÀI TẬP GIÁO TRÌNH (BÀI 5.1 & BÀI 5.2)
# ==============================================================================
def test_daoham_r7_bai_tap_5_1():
    """[R7 - Ca 1]: Bài tập 5.1 SGK - Bảng f(x) tại x=50, 55, 60."""
    print("\n--- [R7 - Ca 1: Bài tập 5.1 SGK] ---")
    xs = np.array([50.0, 55.0, 60.0])
    ys = np.array([1.6990, 1.7404, 1.7782])
    
    df_table = diff_table(xs, ys)
    print(f"  + Kết quả diff_table tại x=50: {df_table[0]:.6f}")
    print(f"  + Kết quả diff_table tại x=60: {df_table[-1]:.6f}")
    assert len(df_table) == len(xs)


def test_daoham_r7_bai_tap_5_2():
    """[R7 - Ca 1]: Bài tập 5.2 SGK - Tính f'(1) từ bảng số liệu."""
    print("\n--- [R7 - Ca 1: Bài tập 5.2 SGK (Tính f'(1))] ---")
    xs = np.array([0.98, 1.00, 1.02])
    ys = np.array([0.773933, 0.765198, 0.756332])
    
    df_table = diff_table(xs, ys)
    df_at_1 = df_table[1]
    expected_hand = (ys[2] - ys[0]) / (2.0 * 0.02)
    
    print(f"  + f'(1) tính được = {df_at_1:.6f} | Bằng tay = {expected_hand:.6f}")
    assert pytest.approx(df_at_1, abs=1e-5) == expected_hand


# ==============================================================================
# 3. KIỂM THỬ CA BIÊN (DIFF_TABLE NÚT BIÊN KHÔNG DÙNG TRUNG TÂM)
# ==============================================================================
def test_daoham_r7_ca_bien_diff_table():
    """[R7 - Ca 3]: Ca biên cho diff_table.
    Nút biên trái (đầu) BẮT BUỘC dùng sai phân tiến.
    Nút biên phải (cuối) BẮT BUỘC dùng sai phân lùi.
    """
    print("\n--- [R7 - Ca 3: Kiểm tra các ca biên diff_table] ---")
    xs = np.array([0.0, 1.0, 2.0, 3.0])
    ys = np.array([0.0, 1.0, 4.0, 9.0])
    h = 1.0
    
    res = diff_table(xs, ys)
    
    expected_first = (ys[1] - ys[0]) / h
    expected_last = (ys[-1] - ys[-2]) / h
    
    print(f"  + Nút biên đầu (x=0):  Tính = {res[0]:.4f} | Tiến bậc 1 = {expected_first:.4f}")
    print(f"  + Nút biên cuối (x=3): Tính = {res[-1]:.4f} | Lùi bậc 1  = {expected_last:.4f}")
    
    assert pytest.approx(res[0], abs=1e-5) == expected_first
    assert pytest.approx(res[-1], abs=1e-5) == expected_last


def test_daoham_r7_ca_bien_lagrange_khong_cach_deu():
    """[R7 - Ca 3]: Đạo hàm Lagrange cho mốc không cách đều."""
    print("\n--- [R7 - Ca 3: Kiểm thử diff_lagrange với mốc không cách đều] ---")
    xs = np.array([0.0, 0.5, 2.0])
    ys = xs**2
    
    df_lag = diff_lagrange(xs, ys, k=1)
    print(f"  + Lagrange tại x=0.5: Kết quả = {df_lag:.6f} | Giá trị đúng = 1.000000")
    assert pytest.approx(df_lag, abs=1e-5) == 1.0


# ==============================================================================
# CHẠY TRỰC TIẾP FILE TEST
# ==============================================================================
if __name__ == "__main__":
    test_daoham_r7_bat_buoc_sample_func()
    test_daoham_r7_bai_tap_5_1()
    test_daoham_r7_bai_tap_5_2()
    test_daoham_r7_ca_bien_diff_table()
    test_daoham_r7_ca_bien_lagrange_khong_cach_deu()
    print("\n=> PASSED: Toàn bộ kiểm thử Nhóm 5 (Đạo hàm - phần lý thuyết & ca biên) đã chạy thành công!")