import numpy as np
import pytest

# IMPORT TOÀN BỘ CÁC HÀM TỪ FILE THUẬT TOÁN CỦA NHÓM (thuat_toan.py)
from thuat_toan import (
    midpoint, 
    trapezoid, 
    trapezoid_table,
    simpson, 
    simpson38, 
    simpson_mixed, 
    simpson_table
)


# ==============================================================================
# CÁC HÀM BÀI TOÁN MẪU VÀ NGHIỆM ĐÚNG DÙNG CHUNG CHO KIỂM THỬ
# ==============================================================================
def f_bai_5_3(x):
    """Ví dụ 5.3: f(x) = 1 / (1 + x) trên [0, 1] -> Nghiệm đúng = ln(2) ≈ 0.693147"""
    return 1.0 / (1.0 + x)

def exact_bai_5_3():
    return np.log(2.0)

def f_vi_du_bo_sung(x):
    """Ví dụ bổ sung: f(x) = exp(-x^2) trên [0, 1]"""
    return np.exp(-x**2)

def f_nhom_tu_chon(x):
    """Ví dụ nhóm tự chọn: f(x) = x^2 trên [0, 1] -> Tích phân đúng = 1/3"""
    return x**2

def exact_nhom_tu_chon():
    return 1.0 / 3.0


# ==============================================================================
# 1. KIỂM THỬ PHƯƠNG PHÁP HÌNH CHỮ NHẬT TRUNG TÂM (MIDPOINT)
# ==============================================================================
def test_midpoint_r7_1_vi_du_bai_giang():
    """[R7 - Ca 1]: Ví dụ giáo trình (Ví dụ 5.3 và Ví dụ bổ sung) cho Midpoint."""
    print("\n--- [R7 - Ca 1: Ví dụ giáo trình cho Midpoint] ---")
    
    # Ví dụ 5.3: n = 10, M2 = 2.0
    I_53, E_53 = midpoint(f_bai_5_3, 0.0, 1.0, 10, M2=2.0)
    true_53 = exact_bai_5_3()
    print(f"  + Ví dụ 5.3 (n=10): I_midpoint = {I_53:.6f} | Nghiệm đúng = {true_53:.6f} | Cận sai số = {E_53:.2e}")
    assert pytest.approx(I_53, abs=1e-3) == 0.692835

    # Ví dụ bổ sung: n = 10, M2 = 2.0
    I_bs, E_bs = midpoint(f_vi_du_bo_sung, 0.0, 1.0, 10, M2=2.0)
    print(f"  + Ví dụ bổ sung (n=10): I_midpoint = {I_bs:.6f} | Cận sai số = {E_bs:.2e}")
    assert pytest.approx(I_bs, abs=1e-3) == 0.747131


def test_midpoint_r7_2_vi_du_nhom_tu_chon():
    """[R7 - Ca 2]: Ví dụ nhóm tự chọn cho Midpoint."""
    print("\n--- [R7 - Ca 2: Ví dụ nhóm tự chọn cho Midpoint (f(x) = x^2)] ---")

    a, b, n, M2 = 0.0, 1.0, 10, 2.0
    I_tc, E_tc = midpoint(f_nhom_tu_chon, a, b, n, M2)
    true_tc = exact_nhom_tu_chon()

    print(f"  + Tự chọn (n=10): I_midpoint = {I_tc:.6f} | Nghiệm đúng = {true_tc:.6f}")

    assert pytest.approx(I_tc, abs=1e-3) == true_tc
    assert np.isclose(I_tc, true_tc, atol=1e-3)


def test_midpoint_r7_3_ca_bien():
    """[R7 - Ca 3]: Ca biên (Cận dưới >= cận trên, n âm) cho Midpoint."""
    print("\n--- [R7 - Ca 3: Kiểm tra các ca biên cho Midpoint] ---")
    with pytest.raises(Exception):
        midpoint(f_bai_5_3, 1.0, 0.0, 10, 2.0)  # Cận a > b
    with pytest.raises(Exception):
        midpoint(f_bai_5_3, 0.0, 1.0, -5, 2.0)  # Số khoảng chia n âm
    print("  + Ca biên Midpoint: Đã bắt đúng ngoại lệ thành công!")


# ==============================================================================
# 2. KIỂM THỬ PHƯƠNG PHÁP HÌNH THANG (TRAPEZOID)
# ==============================================================================
def test_trapezoid_r7_1_vi_du_bai_giang():
    """[R7 - Ca 1]: Ví dụ giáo trình (Ví dụ 5.3 và Ví dụ bổ sung) cho Trapezoid."""
    print("\n--- [R7 - Ca 1: Ví dụ giáo trình cho Trapezoid] ---")
    
    I_53, E_53 = trapezoid(f_bai_5_3, 0.0, 1.0, 10, M2=2.0)
    print(f"  + Ví dụ 5.3 (n=10): I_trapezoid = {I_53:.6f} | Cận sai số = {E_53:.2e}")
    assert pytest.approx(I_53, abs=1e-3) == 0.693771

    I_bs, E_bs = trapezoid(f_vi_du_bo_sung, 0.0, 1.0, 10, M2=2.0)
    print(f"  + Ví dụ bổ sung (n=10): I_trapezoid = {I_bs:.6f} | Cận sai số = {E_bs:.2e}")
    assert pytest.approx(I_bs, abs=1e-3) == 0.746211


def test_trapezoid_r7_2_vi_du_nhom_tu_chon():
    """[R7 - Ca 2]: Ví dụ nhóm tự chọn cho Trapezoid."""
    print("\n--- [R7 - Ca 2: Ví dụ nhóm tự chọn cho Trapezoid (f(x) = x^2)] ---")

    a, b, n, M2 = 0.0, 1.0, 10, 2.0
    I_tc, E_tc = trapezoid(f_nhom_tu_chon, a, b, n, M2)
    true_tc = exact_nhom_tu_chon()

    print(f"  + Tự chọn (n=10): I_trapezoid = {I_tc:.6f} | Nghiệm đúng = {true_tc:.6f}")

    assert pytest.approx(I_tc, abs=1e-2) == true_tc
    assert np.isclose(I_tc, true_tc, atol=1e-2)


def test_trapezoid_r7_3_ca_bien():
    """[R7 - Ca 3]: Ca biên (n <= 0) cho Trapezoid."""
    print("\n--- [R7 - Ca 3: Kiểm tra các ca biên cho Trapezoid] ---")
    with pytest.raises(ValueError):
        trapezoid(f_bai_5_3, 0.0, 1.0, 0, 2.0)  # n = 0
    print("  + Ca biên Trapezoid (n=0): Đã bắt đúng lỗi ValueError!")


# ==============================================================================
# 3. KIỂM THỬ PHƯƠNG PHÁP SIMPSON 1/3 (SIMPSON)
# ==============================================================================
def test_simpson_r7_1_vi_du_bai_giang():
    """[R7 - Ca 1]: Ví dụ giáo trình (Ví dụ 5.3 và Ví dụ bổ sung) cho Simpson 1/3."""
    print("\n--- [R7 - Ca 1: Ví dụ giáo trình cho Simpson 1/3] ---")
    
    # M4 cho 1/(1+x) là 24.0
    I_53, E_53 = simpson(f_bai_5_3, 0.0, 1.0, 10, M4=24.0)
    print(f"  + Ví dụ 5.3 (n=10): I_simpson = {I_53:.6f} | Cận sai số = {E_53:.2e}")
    assert pytest.approx(I_53, abs=1e-5) == 0.693150

    # M4 cho exp(-x^2) xấp xỉ 12.0
    I_bs, E_bs = simpson(f_vi_du_bo_sung, 0.0, 1.0, 10, M4=12.0)
    print(f"  + Ví dụ bổ sung (n=10): I_simpson = {I_bs:.6f} | Cận sai số = {E_bs:.2e}")
    assert pytest.approx(I_bs, abs=1e-5) == 0.746825


def test_simpson_r7_2_vi_du_nhom_tu_chon():
    """[R7 - Ca 2]: Ví dụ nhóm tự chọn cho Simpson 1/3."""
    print("\n--- [R7 - Ca 2: Ví dụ nhóm tự chọn cho Simpson 1/3 (f(x) = x^2)] ---")

    a, b, n, M4 = 0.0, 1.0, 4, 0.0
    I_tc, E_tc = simpson(f_nhom_tu_chon, a, b, n, M4)
    true_tc = exact_nhom_tu_chon()

    print(f"  + Tự chọn (n=4): I_simpson = {I_tc:.6f} | Nghiệm đúng = {true_tc:.6f}")

    assert pytest.approx(I_tc, abs=1e-6) == true_tc
    assert np.isclose(I_tc, true_tc, atol=1e-6)


def test_simpson_r7_3_ca_bien():
    """[R7 - Ca 3]: Ca biên bắt buộc (Simpson với n lẻ phải báo lỗi)."""
    print("\n--- [R7 - Ca 3: Kiểm tra các ca biên cho Simpson 1/3] ---")
    with pytest.raises(ValueError, match="yêu cầu n phải là số chẵn"):
        simpson(f_bai_5_3, 0.0, 1.0, n=5, M4=24.0)  # n = 5 là số lẻ
    print("  + Ca biên n lẻ (n=5): Đã bắt đúng lỗi yêu cầu n chẵn!")


# ==============================================================================
# 4. KIỂM THỬ PHƯƠNG PHÁP SIMPSON 3/8, SIMPSON MIXED & VÍ DỤ 5.7
# ==============================================================================
def test_simpson_mixed_r7_1_vi_du_5_7():
    """[R7 - Ca 1]: Ví dụ 5.7 (Bảng số liệu với n = 5 lẻ, kết quả chuẩn 4.1036)."""
    print("\n--- [R7 - Ca 1: Kiểm thử Ví dụ 5.7 cho Simpson Mixed (n=5 lẻ)] ---")
    
    # Dữ liệu bảng số liệu Ví dụ 5.7
    xs = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
    ys = [1.5000, 2.0000, 2.0000, 1.6364, 1.2500, 0.9565]
    
    res_57 = simpson_table(xs, ys)
    print(f"  + Ví dụ 5.7 (n=5 lẻ): Kết quả tính được = {res_57:.4f} | Giá trị lý thuyết = 4.1036")
    assert pytest.approx(res_57, abs=1e-3) == 4.1036


def test_simpson_mixed_r7_2_vi_du_nhom_tu_chon():
    """[R7 - Ca 2]: Ví dụ nhóm tự chọn với simpson_mixed (n = 7 lẻ) + đối chiếu scipy."""
    print("\n--- [R7 - Ca 2: Ví dụ nhóm tự chọn Simpson Mixed (n=7 lẻ)] ---")
    a, b, n = 0.0, 1.0, 7
    res_mixed = simpson_mixed(f_nhom_tu_chon, a, b, n)
    true_val = exact_nhom_tu_chon()

    print(f"  + Tự chọn (n=7 lẻ): Simpson Mixed = {res_mixed:.6f} | Nghiệm đúng = {true_val:.6f}")

    assert pytest.approx(res_mixed, abs=1e-3) == true_val


def test_simpson_mixed_r7_3_ca_bien():
    """[R7 - Ca 3]: Ca biên cho Simpson Mixed và Simpson 3/8 đơn."""
    print("\n--- [R7 - Ca 3: Kiểm tra các ca biên cho Simpson Mixed và 3/8] ---")
    
    # Simpson 3/8 đơn với hàm x^3 trên [0, 3] -> Kết quả đúng = 20.25
    f_cube = lambda x: x**3
    res_38 = simpson38(f_cube, 0.0, 3.0)
    print(f"  + Simpson 3/8 đơn (x^3 trên [0,3]): Kết quả = {res_38:.4f} (Đúng = 20.25)")
    assert pytest.approx(res_38, abs=1e-5) == 20.25

    # Simpson mixed nhánh n chẵn (n = 4)
    res_even = simpson_mixed(f_bai_5_3, 0.0, 1.0, n=4)
    print(f"  + Biên n chẵn (n=4 nhánh 1/3): Kết quả = {res_even:.6f}")
    assert isinstance(res_even, float)


# ==============================================================================
# CHẠY TRỰC TIẾP FILE TEST
# ==============================================================================
if __name__ == "__main__":
    test_midpoint_r7_1_vi_du_bai_giang()
    test_midpoint_r7_2_vi_du_nhom_tu_chon()
    test_midpoint_r7_3_ca_bien()
    
    test_trapezoid_r7_1_vi_du_bai_giang()
    test_trapezoid_r7_2_vi_du_nhom_tu_chon()
    test_trapezoid_r7_3_ca_bien()
    
    test_simpson_r7_1_vi_du_bai_giang()
    test_simpson_r7_2_vi_du_nhom_tu_chon()
    test_simpson_r7_3_ca_bien()
    
    test_simpson_mixed_r7_1_vi_du_5_7()
    test_simpson_mixed_r7_2_vi_du_nhom_tu_chon()
    test_simpson_mixed_r7_3_ca_bien()
    
    print("\n=> PASSED: Toàn bộ kiểm thử các phương pháp tính gần đúng tích phân theo chuẩn R7 đã chạy thành công!")