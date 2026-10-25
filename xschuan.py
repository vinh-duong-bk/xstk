import ast
import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def plot_normal_probability(
    data,
    title="Biểu đồ xác suất chuẩn",
    save_path=None,
    show_plot=True,
    normality_assumption=None,
):
    x = np.sort(np.asarray(data, dtype=float))
    n = len(x)
    
    j = np.arange(1, n + 1)
    p = (j - 0.5) / n
    z = stats.norm.ppf(p)
    
    df = pd.DataFrame({
        'j': j,
        'xⱼ': [int(val) if val.is_integer() else val for val in x],
        'pⱼ = (j - 0,5) / n': np.round(p, 2),
        'zⱼ = Φ⁻¹(pⱼ)': np.round(z, 2)
    })
    
    print("=" * 60)
    print("          BẢNG TÍNH BIỂU ĐỒ XÁC SUẤT CHUẨN")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)
    
    sample_mean = np.mean(x)
    sample_std = np.std(x, ddof=1)
    standard_error = sample_std / np.sqrt(n)
    slope, intercept, _, _, _ = stats.linregress(z, x)
    normality_p_value = stats.shapiro(x).pvalue if n >= 3 else np.nan
    is_approximately_normal = (
        normality_assumption
        if normality_assumption is not None
        else n >= 3 and normality_p_value >= 0.05
    )

    confidence_interval = None
    if n > 30 or is_approximately_normal:
        degrees_of_freedom = n - 1
        confidence_level = 0.95
        alpha = 1 - confidence_level
        critical_value = stats.t.ppf(1 - alpha / 2, degrees_of_freedom)
        margin_of_error = critical_value * standard_error
        confidence_interval = (
            sample_mean - margin_of_error,
            sample_mean + margin_of_error
        )

    print("\nCÁC ĐẠI LƯỢNG THỐNG KÊ:")
    print(f"  • Cỡ mẫu (n): {n}")
    print(f"  • Giá trị trung bình mẫu (x̄): {sample_mean:.4f}")
    print(f"  • Độ lệch chuẩn mẫu (s): {sample_std:.4f}")
    print(f"  • Sai số chuẩn của x̄, SE(x̄): {standard_error:.4f}")
    if confidence_interval is not None:
        print(f"  • Khoảng tin cậy 95% cho kỳ vọng toán μ: "
              f"[{confidence_interval[0]:.4f}; {confidence_interval[1]:.4f}]")
    else:
        print("  • Không tính khoảng tin cậy 95% cho kỳ vọng toán μ: "
              "cỡ mẫu nhỏ hơn hoặc bằng 30 và dữ liệu không gần phân phối chuẩn.")
    print()
    
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=120)
    
    ax.scatter(z, x, color='black', marker='*', s=60, zorder=3, label='(zⱼ, xⱼ)')
    
    z_line = np.linspace(-2.3, 2.2, 100)
    x_line = slope * z_line + intercept
    ax.plot(z_line, x_line, color='gray', linestyle='--', linewidth=1.5, zorder=2)
    
    ax.set_xlabel('zⱼ', fontsize=11, fontweight='bold')
    ax.set_ylabel('xⱼ', fontsize=11, fontweight='bold', rotation=0, labelpad=12)
    
    ax.set_xlim(-2.5, 2.5)
    ax.set_xticks([-2, -1, 0, 1, 2])
    
    y_min = int(np.floor(np.min(x) / 10) * 10 - 10)
    y_max = int(np.ceil(np.max(x) / 10) * 10 + 10)
    ax.set_ylim(y_min, y_max)
    ax.set_yticks(np.arange(y_min, y_max + 1, 10))
    
    ax.grid(True, linestyle=':', color='#b0b0b0', linewidth=0.8)
    
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.0)
        
    plt.title(title, fontsize=11, pad=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"-> Đã lưu ảnh đồ thị vào: {save_path}")
        
    if show_plot:
        plt.show()
        
    return df, fig


def show_console_plot(z, x):
    width = 81
    height = 25
    z_min, z_max = -2.5, 2.5
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    padding = (x_max - x_min) * 0.08 or 1.0
    y_min = x_min - padding
    y_max = x_max + padding
    y_range = y_max - y_min
    y_axis_value = 0.0 if y_min <= 0 <= y_max else y_min
    x_axis_row = round((y_max - y_axis_value) / y_range * (height - 1))
    y_axis_column = round((0.0 - z_min) / (z_max - z_min) * (width - 1))

    def coordinates(point_z, point_x):
        column = round((point_z - z_min) / (z_max - z_min) * (width - 1))
        row = round((y_max - point_x) / y_range * (height - 1))
        return (
            max(0, min(width - 1, column)),
            max(0, min(height - 1, row)),
        )

    for step in range(len(x)):
        canvas = [[" " for _ in range(width)] for _ in range(height)]
        for row in range(height):
            canvas[row][y_axis_column] = "|"
        canvas[x_axis_row] = ["-" for _ in range(width)]
        canvas[x_axis_row][y_axis_column] = "+"

        for point_z, point_x in zip(z[:step + 1], x[:step + 1]):
            column, row = coordinates(point_z, point_x)
            canvas[row][column] = "*"

        print("\033[2J\033[H", end="")
        print("BIỂU ĐỒ XÁC SUẤT CHUẨN - ĐIỂM THỨ "
              f"{step + 1}/{len(x)}")
        print(f"z: [{z_min:.1f}, {z_max:.1f}]   "
              f"x: [{y_min:.2f}, {y_max:.2f}]")
        print("Chú thích: * điểm dữ liệu   |/- hai trục tọa độ")
        print("\n".join("".join(row) for row in canvas))
        if step < len(x) - 1:
            input("Nhấn Enter để hiển thị điểm tiếp theo...")


def parse_data(raw_data):
    normalized_data = re.sub(r'(?<=\d),(?=\d)', '.', raw_data.strip())
    values = []
    for token in re.split(r'[\s,;]+', normalized_data):
        if token:
            value = float(token)
            if not np.isfinite(value):
                raise ValueError("Dữ liệu phải là các số hữu hạn.")
            values.append(value)
    if len(values) < 2:
        raise ValueError("Vui lòng cung cấp ít nhất 2 giá trị.")
    return values


def read_data_from_keyboard():
    print("Bạn có thể dùng dấu phẩy hoặc dấu chấm cho phần thập phân.")
    print("Hãy ngăn cách các giá trị bằng dấu cách hoặc dấu chấm phẩy.")
    while True:
        try:
            return parse_data(input("Nhập dữ liệu: "))
        except ValueError as error:
            print(f"Dữ liệu chưa hợp lệ: {error}")


def read_data_from_file():
    while True:
        print("\nCHỌN BỘ DỮ LIỆU")
        print("1. test1 - dữ liệu gần phân phối chuẩn")
        print("2. test2 - bộ dữ liệu kiểm tra")
        print("3. test3 - bộ dữ liệu bổ sung")
        file_choice = input("Nhập lựa chọn (1, 2 hoặc 3): ").strip()
        project_path = Path(__file__).resolve().parent
        test_files = {
            "1": project_path / "test1.txt",
            "2": project_path / "test2.txt",
            "3": project_path / "test3.txt",
        }
        if file_choice not in test_files:
            print("Lựa chọn không hợp lệ. Vui lòng nhập 1, 2 hoặc 3.")
            continue

        try:
            path = test_files[file_choice]
            if not path.is_file():
                raise FileNotFoundError(f"Không tìm thấy file: {path}")
            return (
                read_data_file(path),
                read_normality_assumption(path),
                path.stem,
            )
        except (OSError, UnicodeError, ValueError) as error:
            print(f"Không thể đọc bộ dữ liệu: {error}")


def read_data_file(path):
    raw_data = path.read_text(encoding="utf-8")
    try:
        return parse_data(raw_data)
    except ValueError as parse_error:
        list_match = re.search(r"\[[\s\S]*\]", raw_data)
        if list_match:
            try:
                return parse_data(list_match.group(0)[1:-1])
            except ValueError:
                pass

        try:
            parsed_file = ast.parse(raw_data, filename=str(path))
        except SyntaxError:
            raise parse_error

        for statement in parsed_file.body:
            if isinstance(statement, (ast.Assign, ast.AnnAssign)):
                value = statement.value
                if isinstance(value, (ast.List, ast.Tuple)):
                    values = [
                        element.value
                        for element in value.elts
                        if isinstance(element, ast.Constant)
                        and isinstance(element.value, (int, float))
                        and not isinstance(element.value, bool)
                    ]
                    if len(values) == len(value.elts):
                        return parse_data(" ".join(map(str, values)))

        raise parse_error


def read_normality_assumption(path):
    raw_data = path.read_text(encoding="utf-8")
    if re.search(r"\bdata_normal\s*=", raw_data):
        return True
    if re.search(r"\bdata_non_normal\s*=", raw_data):
        return False
    return None


def choose_data_source():
    while True:
        print("\nCHỌN NGUỒN DỮ LIỆU")
        print("1. Nhập dữ liệu trực tiếp từ bàn phím")
        print("2. Chọn một bộ dữ liệu có sẵn trong file")
        input_choice = input("Nhập lựa chọn (1 hoặc 2): ").strip()

        if input_choice == "1":
            return read_data_from_keyboard(), None, "keyboard"
        if input_choice == "2":
            return read_data_from_file()
        print("Lựa chọn không hợp lệ. Vui lòng nhập 1 hoặc 2.")


def choose_display_mode():
    while True:
        print("\nCHỌN CÁCH HIỂN THỊ BIỂU ĐỒ")
        print("1. Mở cửa sổ biểu đồ")
        print("2. Hiển thị lần lượt từng điểm trên console")
        choice = input("Nhập lựa chọn (1 hoặc 2): ").strip()
        if choice == "1":
            return "window"
        if choice == "2":
            return "console"
        print("Lựa chọn không hợp lệ. Vui lòng nhập 1 hoặc 2.")


if __name__ == '__main__':
    data_input, normality_assumption, source_name = choose_data_source()
    display_mode = choose_display_mode()
    output_path = Path(__file__).resolve().parent / (
        f"normal_probability_plot_{source_name}.png"
    )

    df_result, fig = plot_normal_probability(
        data=data_input,
        save_path=output_path,
        show_plot=display_mode == "window",
        normality_assumption=normality_assumption,
    )
    if display_mode == "console":
        sorted_data = np.sort(np.asarray(data_input, dtype=float))
        probabilities = (np.arange(1, len(sorted_data) + 1) - 0.5) / len(sorted_data)
        show_console_plot(stats.norm.ppf(probabilities), sorted_data)