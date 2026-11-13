import numbers
import numpy as np

#P1: Phương pháp hình chữ nhật trung tâm (midpoint)
def midpoint(f, a, b, n, M2) -> tuple[float, float]: #(I, E)
    """
    tên thuật toán: Phương pháp hình chữ nhật trung tâm kèm cận sai số
    require:
        f: hàm số cần tính tích phân (callable function)
        a: cận dưới của tích phân (float hoặc int)
        b: cận trên của tích phân (float hoặc int)
        n: số khoảng chia (số nguyên dương)
        M2: chặn trên của giá trị tuyệt đối đạo hàm cấp 2 ( dùng để đánh giá sai số)
    ensure:
        tuple(I_rec, E): giá trị xấp xỉ tích phân và cận sai số lí thuyết
    """
    #xử lí tình huống biên:
    if not callable(f):
        raise(Exception('tham so f phai la mot ham so hop le'))
    if a >= b:
        raise(Exception('can duoi a phai nho hon can tren b'))
    if not isinstance(n, int) or n <= 0:
        raise(Exception('so khoang chia n phai la so nguyen duong'))
    if M2<0:
        raise(Exception('can sai so M2 phai la so ko am'))
    h = (b-a)/n #dòng 1- thuật toán 13
    Sum_val = 0.0
    for i in range(n): #dòng 2 - thuật toán 13
        midpoint_val = ((a+i*h) + (a + (i+1)*h))/2 #dòng 3 - thuật toán 13
        Sum_val += f(midpoint_val) #dòng 4 - thuật toán 13
    I_rec = h*Sum_val #dòng 6 - thuật toán 13
    E = (M2 * (b-a) * (h**2))/24
    return I_rec, E #dòng7 - thuật toán 13

#P2: Phương pháp hình thang
'''
Require:
    -f: hàm số cần tính tích phân
    -a: cận dưới
    -b: cận trên
    -n: số khoảng chia (n>0)
    -M2: chặn trên đạo hàm cấp 2
    -h: độ rộng khoảng chia
Ensure:
    -trả về giá trị tuple[I, E]: giá trị tích phân xấp xỉ I và cận sai số E
'''
def trapezoid(f, a, b, n, M2): 
  if not isinstance(n, int):
        raise ValueError('Số khoảng chia n phải là số nguyên')
  if n<=0:
        raise ValueError('Số khoảng chia phải lớn hơn 0')
  if a >= b:
        raise ValueError('Cận dưới a phải nhỏ hơn cận trên b')
  sum=0
  h=(b-a)/n #dòng 1 thuật toán 14
  for i in range(1,n):
      sum+=f(a+i*h)
  I=(h/2)*(f(a)+2*sum+f(b)) #dòng 5 thuật toán 14
  E=M2*(b-a)**3/(12*n**2)
  return I, E
'''
Require:
    -x: danh sách mảng dữ liệu x
    -y: danh sách mảng dữ lệu y
    -h: độ rộng khoảng chia
Ensure:
    -trả về giá trị area: tổng diện tích bề mặt cắt ngang xấp xỉ tính được 
'''
def trapezoid_table(x, y):
    area=0
    for i in range(len(x)-1):
        h=x[i+1]-x[i]
        area+=0.5*h*(y[i]+y[i+1])
    return area

#P3: Simpson 1/3
def simpson(f, a: float, b: float, n: int, M4: float) -> tuple[float, float]:
    """
    Thuật toán 16: Tính tích phân xác định bằng phương pháp Simpson 1/3.

    Require (Đầu vào):
        f (callable): Hàm số f(x) liên tục trên [a, b].
        a (float): Cận dưới tích phân.
        b (float): Cận trên tích phân.
        n (int,  n >= 2): Số khoảng chia n chẵn.
        M4 (float): Chặn trên đạo hàm bậc 4 của f(x) trên [a, b].

    Ensure (Đầu ra):
        tuple[float, float]: (I_Simpson, E) trong đó:
            - I_Simpson (float): Giá trị tích phân xấp xỉ.
            - E (float): Cận sai số lý thuyết.
    """
    # Kiểm tra điều kiện biên
    if n <= 0:
        raise ValueError("Lỗi biên: Số đoạn chia n phải là số nguyên dương.")
    if n % 2 != 0:
        raise ValueError(f"Lỗi biên: Phương pháp Simpson yêu cầu n phải là số chẵn .")
    if M4 < 0:
        raise ValueError("Lỗi biên: Chặn trên đạo hàm M4 không thể âm.")
    if a > b:
        raise ValueError("Lỗi biên: Cận dưới a không thể lớn hơn cận trên b.")

    # BẮT ĐẦU THUẬT TOÁN

    h = (b - a) / n # Dòng 1 - Thuật toán 16: Tính h

    # Khởi tạo các biến tổng tích lũy
    sum1 = 0.0 # Tổng các vị trí i chẵn (nhân với 2)
    sum2 = 0.0 # Tổng các vị trí i lẻ (nhân với 4)

    # Định nghĩa hàm tính x_i = a + i*h
    def x(i):
        return a + i * h

    for i in range(1, n):     # Dòng 2 - Thuật toán 16
        if i % 2 == 0: # Dòng 3 - Thuật toán 16
            sum1 += f(x(i)) # Dòng 4 - Thuật toán 16
        else:
            sum2 += f(x(i)) # Dòng 6 - Thuật toán 16

    I_Simpson = (h / 3) * (f(a) + 2 * sum1 + 4 * sum2 + f(b))  # Dòng 9 - Thuật toán 16
    E = (M4 * (b - a) * h**4) / 180
    return I_Simpson, E  # Dòng 10 - Thuật toán 16

#P4: Simpson 3/8
# 1. Kiểm tra điều kiện biên
def check_bounds(a, b):
    """
    Kiểm tra hai cận tích phân.
    Require:
        a, b: Hai cận tích phân.
    Ensure:
        a và b là số hữu hạn, a < b.
    """
    if not isinstance(a, numbers.Real) or not isinstance(b, numbers.Real):
        raise TypeError("Hai cận tích phân phải là số.")
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("Hai cận tích phân phải là số hữu hạn.")
    if a >= b:
        raise ValueError("Cận dưới phải nhỏ hơn cận trên.")

# 2. Kiểm tra số khoảng chia
def check_n(n):
    """
    Kiểm tra số khoảng chia.
    Require:
        n: Số nguyên biểu thị số khoảng chia.
    Ensure:
        n là số nguyên và n >= 2.
    """
    if not isinstance(n, numbers.Integral) or isinstance(n, bool):
        raise TypeError("n phải là số nguyên.")
    if n < 2:
        raise ValueError("n phải lớn hơn hoặc bằng 2.")

# 3. Công thức Simpson 3/8
def simpson38(f, a, b):
    """
    Tính tích phân xác định bằng công thức Simpson 3/8 đơn.
    Chia đoạn [a, b] thành 3 khoảng bằng nhau và áp dụng công thức Simpson 3/8.
    Require:
        f: Hàm số cần tính tích phân.
        a: Cận dưới.
        b: Cận trên.
    Ensure:
        Trả về giá trị gần đúng của tích phân trên [a, b].
    """
    check_bounds(a, b)
    h = (b - a) / 3
    return 3 * h / 8 * ( f(a) + 3 * f(a + h) + 3 * f(a + 2 * h) + f(b) )

# 4. Kết hợp Simpson 3/8 và Simpson 1/3
def simpson_mixed(f, a, b, n):
    """
    Tính tích phân bằng phương pháp Simpson kết hợp.
    Nếu n chẵn, sử dụng Simpson 1/3.
    Nếu n lẻ, sử dụng Simpson 3/8 cho 3 khoảng đầu
    Require:
        f: Hàm số cần tính tích phân.
        a: Cận dưới.
        b: Cận trên.
        n: Số khoảng chia.

    Ensure:
        Trả về giá trị gần đúng của tích phân trên [a, b].
        và Simpson 1/3 cho các khoảng còn lại.
    """
    check_bounds(a, b)
    check_n(n)
    h = (b - a) / n
    # Trường hợp n chẵn: sử dụng Simpson 1/3
    if n % 2 == 0:
        s = f(a) + f(b)

        for i in range(1, n):
            if i % 2 == 1:
                s += 4 * f(a + i * h)
            else:
                s += 2 * f(a + i * h)
        return h * s / 3

    # Trường hợp n lẻ: sử dụng Simpson 3/8 cho 3 khoảng đầu
    result = 3 * h / 8 * ( f(a) + 3 * f(a + h) + 3 * f(a + 2 * h) + f(a + 3 * h) )

    # Các khoảng còn lại sử dụng Simpson 1/3
    if n > 3:
        s = f(a + 3 * h) + f(b)
        for i in range(4, n):
            if (i - 3) % 2 == 1:
                s += 4 * f(a + i * h)
            else:
                s += 2 * f(a + i * h)
        result += h * s / 3
    return result

# 5. Tính tích phân từ bảng dữ liệu (xs, ys)
def simpson_table(xs, ys):
    """
    Tính tích phân xác định từ bảng giá trị.
    Nếu số khoảng chia chẵn, sử dụng Simpson 1/3.
    Nếu số khoảng chia lẻ, kết hợp Simpson 3/8 và 1/3.
    Require:
        xs: Mảng chứa các giá trị của biến x.
        ys: Mảng chứa các giá trị tương ứng của hàm số f(x).

    Ensure:
        Trả về giá trị gần đúng của tích phân.
        Các giá trị x phải tăng đều và hai mảng có cùng độ dài.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if xs.ndim != 1 or ys.ndim != 1:
        raise ValueError("xs và ys phải là mảng một chiều.")
    if len(xs) != len(ys):
        raise ValueError("xs và ys phải có cùng số phần tử.")
    if len(xs) < 3:
        raise ValueError("Cần ít nhất 3 điểm dữ liệu.")
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        raise ValueError("Dữ liệu phải là các số hữu hạn.")
    h_values = np.diff(xs)
    if np.any(h_values <= 0):
        raise ValueError("Các giá trị xs phải tăng dần.")
    if not np.allclose(h_values, h_values[0]):
        raise ValueError("Các khoảng chia phải bằng nhau.")
    n = len(xs) - 1
    h = h_values[0]
    # Trường hợp n chẵn: sử dụng Simpson 1/3
    if n % 2 == 0:
        s = ys[0] + ys[n]
        for i in range(1, n):
            if i % 2 == 1:
                s += 4 * ys[i]
            else:
                s += 2 * ys[i]
        return h * s / 3
    # Trường hợp n lẻ: Simpson 3/8 cho 3 khoảng đầu
    result = 3 * h / 8 * ( ys[0] + 3 * ys[1] + 3 * ys[2] + ys[3] )
      # Các khoảng còn lại sử dụng Simpson 1/3
    if n > 3:
        s = ys[3] + ys[n]
        for i in range(4, n):
            if (i - 3) % 2 == 1:
                s += 4 * ys[i]
            else:
                s += 2 * ys[i]
        result += h * s / 3
    return result
