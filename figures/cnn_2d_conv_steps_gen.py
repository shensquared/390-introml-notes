#!/usr/bin/env python3
"""Generate the step-by-step 2D filter-bank convolution animation for the
6.390 CNN chapter (convolutional_neural_networks.qmd, filter-bank example).

Layout mirrors the static tikz figure in the chapter; the last frame equals it.
Pipeline follows demos/cnn/conv_arithmetic: one standalone LaTeX file per
frame -> pdflatex -> png -> gif.
"""
import itertools
import pathlib
import subprocess

HERE = pathlib.Path(__file__).parent
PDF = HERE / "pdf"
PNG = HERE / "png"
PDF.mkdir(exist_ok=True)
PNG.mkdir(exist_ok=True)

# ---- data (row 0 = TOP row, matching the reading order of the figure) ----
X = [
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [1, 1, 1, 1, 1, 1],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0],
]
F2 = [[0, 0, 0], [1, 1, 1], [0, 0, 0]]  # horizontal detector
F1 = [[0, 1, 0], [0, 1, 0], [0, 1, 0]]  # vertical detector
N = 6


def conv(img, ker):
    out = [[0] * N for _ in range(N)]
    for r in range(N):
        for c in range(N):
            s = 0
            for i in range(3):
                for j in range(3):
                    rr, cc = r + i - 1, c + j - 1
                    if 0 <= rr < N and 0 <= cc < N:
                        s += img[rr][cc] * ker[i][j]
            out[r][c] = s
    return out


Y2 = conv(X, F2)          # f_2 channel
Y1 = conv(X, F1)          # f_1 channel
FINAL = [[Y1[r][c] + Y2[r][c] for c in range(N)] for r in range(N)]

# sanity against the corrected book figure
assert Y1[0][3] == 2 and Y1[1][3] == 3 and FINAL[0][3] == 3 and FINAL[1][3] == 4

PRE = r"""\documentclass[class=article,border=2pt]{standalone}
\usepackage{tikz}
\usepackage{xcolor}
\definecolor{winblue}{RGB}{38,139,210}
\definecolor{cellyellow}{RGB}{255,220,80}
\usetikzlibrary{calc}
\begin{document}
\begin{tikzpicture}[scale=0.4]
\useasboundingbox (-1.6,-8.6) rectangle (38.2,11.4);
"""
POST = "\\end{tikzpicture}\n\\end{document}\n"



def glines(ox, oy, n):
    out = [f"\\draw[black,very thick] ({ox},{oy}) rectangle ({ox+n},{oy+n});"]
    for k in range(1, n):
        out.append(f"\\draw[black,thick] ({ox+k},{oy}) -- ({ox+k},{oy+n});")
        out.append(f"\\draw[black,thick] ({ox},{oy+k}) -- ({ox+n},{oy+k});")
    return out

def grid6(ox, oy, values=None, upto=-1, black=None, white_nums=False):
    """6x6 grid at (ox,oy); values row 0 = top (tikz y = oy+5-r).
    upto: highest step index (row-major) whose value is drawn; -1 = none,
    None = all."""
    s = glines(ox, oy, 6)
    black = black or set()
    for r in range(N):
        for c in range(N):
            x, y = ox + c, oy + 5 - r
            if (r, c) in black:
                s.append(f"\\fill[black] ({x},{y}) rectangle ({x+1},{y+1});")
    if values is not None:
        for r in range(N):
            for c in range(N):
                k = r * N + c
                if upto is not None and k > upto:
                    continue
                x, y = ox + c, oy + 5 - r
                col = "white" if (r, c) in black else "black"
                s.append(
                    f"\\node[{col}] at ({x+0.5},{y+0.5}) {{{values[r][c]}}};")
    return "\n".join(s)


def filt3(ox, oy, ker, label, label_above):
    s = glines(ox, oy, 3)
    for i in range(3):
        for j in range(3):
            x, y = ox + j, oy + 2 - i
            if ker[i][j]:
                s.append(f"\\fill[black] ({x},{y}) rectangle ({x+1},{y+1});")
                s.append(f"\\node[white] at ({x+0.5},{y+0.5}) {{1}};")
            else:
                s.append(f"\\node at ({x+0.5},{y+0.5}) {{0}};")
    if label_above:
        s.append(f"\\node[above] at ({ox+1.5},{oy+3.3}) {{{label}}};")
    else:
        s.append(f"\\node[below] at ({ox+1.5},{oy-0.3}) {{{label}}};")
    return "\n".join(s)


def tensor_filter():
    s = []
    for shift in (0.5, 0.0):  # back slice then front slice
        ox, oy = 25 + shift, 1.5 + shift
        s.append(f"\\fill[white] ({ox},{oy}) rectangle ({ox+3},{oy+3});")
        s.extend(glines(ox, oy, 3))
        for i in range(3):
            for j in range(3):
                v = 1 if (i, j) == (1, 1) else 0
                s.append(
                    f"\\node at ({ox+j+0.5},{oy+2.5-i}) {{{v}}};")
    s.append("\\node[above] at (26.75,5.3) {tensor filter};")
    return "\n".join(s)


def window(ox, oy, r, c):
    """3x3 translucent window centered on cell (r,c) of a 6x6 grid at (ox,oy)."""
    x0, y0 = ox + c - 1, oy + 5 - r - 1
    return (f"\\fill[winblue,opacity=0.30] ({x0},{y0}) rectangle ({x0+3},{y0+3});\n"
            f"\\draw[winblue,line width=1.4pt] ({x0},{y0}) rectangle ({x0+3},{y0+3});")


def cell_hl(ox, oy, r, c):
    x, y = ox + c, oy + 5 - r
    return (f"\\fill[cellyellow,opacity=0.75] ({x},{y}) rectangle ({x+1},{y+1});\n"
            f"\\draw[black,line width=1.2pt] ({x},{y}) rectangle ({x+1},{y+1});")


ARROWS_STATIC = r"""
\draw[->,thick] (6.5,4.5) -- (8.5,8);
\draw[->,thick] (6.5,1.5) -- (8.5,-2);
\draw[->,thick] (12.5,8) -- (14.5,8);
\draw[->,thick] (12.5,-3) -- (14.5,-3);
\draw[->,thick] (21.5,8) -- (24.5,4);
\draw[->,thick] (21.5,-3) -- (24.5,2);
\draw[->,thick] (29,3) -- (31,3);
"""

X_BLACK = {(r, c) for r in range(N) for c in range(N) if X[r][c] == 1}


def frame(stage, step):
    """stage 0: bare layout; stage 1: filling channels; stage 2: filling final;
    stage 3: complete."""
    parts = [PRE]
    ch_upto = -1 if stage == 0 else (step if stage == 1 else None)
    fin_upto = -1 if stage in (0, 1) else (step if stage == 2 else None)
    # input, filters, tensor filter, arrows
    parts.append(grid6(0, 0, X, None, X_BLACK, True))
    parts.append(filt3(9, 6.5, F2, "$f_2$ (horizontal)", True))
    parts.append(filt3(9, -4.5, F1, "$f_1$ (vertical)", False))
    parts.append(tensor_filter())
    parts.append(ARROWS_STATIC)
    # channel outputs and final output
    parts.append(grid6(15, 5, Y2, ch_upto))
    parts.append(grid6(15, -6, Y1, ch_upto))
    parts.append(grid6(31.5, 0, FINAL, fin_upto))
    if stage == 1:
        r, c = divmod(step, N)
        parts.append(window(0, 0, r, c))
        parts.append(cell_hl(15, 5, r, c))
        parts.append(cell_hl(15, -6, r, c))
    if stage == 2:
        r, c = divmod(step, N)
        parts.append(window(15, 5, r, c))
        parts.append(window(15, -6, r, c))
        parts.append(cell_hl(31.5, 0, r, c))
    parts.append(POST)
    return "\n".join(parts)


frames = [frame(0, 0)]
frames += [frame(1, k) for k in range(N * N)]
frames += [frame(2, k) for k in range(N * N)]
frames.append(frame(3, 0))

for i, tex in enumerate(frames):
    name = f"frame_{i:03d}"
    (HERE / f"{name}.tex").write_text(tex)
    subprocess.run(
        ["pdflatex", "-interaction=batchmode", "-output-directory", str(PDF),
         f"{name}.tex"],
        cwd=HERE, check=True, capture_output=True)
    subprocess.run(
        ["magick", "-density", "300", str(PDF / f"{name}.pdf"),
         "-background", "white", "-flatten", "-resize", "45%",
         str(PNG / f"{name}.png")],
        check=True, capture_output=True)
    print(name, "ok")
print("frames done:", len(frames))
