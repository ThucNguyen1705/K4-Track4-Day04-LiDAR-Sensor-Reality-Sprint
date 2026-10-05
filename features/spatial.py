"""[Member 2] Feature theo lưới 3x3: glare / blur / bẩn kính thường chỉ ở một vùng ảnh

Tên cột: <feature>_r{row}c{col}, r0 = hàng trên, c0 = cột trái (vd. edge_density_r1c1 = vùng giữa)
"""


def grid_features(gray, funcs, rows=3, cols=3):
    h, w = gray.shape
    out = {}
    for i in range(rows):
        for j in range(cols):
            cell = gray[i * h // rows:(i + 1) * h // rows, j * w // cols:(j + 1) * w // cols]
            for name, fn in funcs.items():
                out[f"{name}_r{i}c{j}"] = fn(cell)
    return out
