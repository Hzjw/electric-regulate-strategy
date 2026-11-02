"""图表统一风格与标注自动避让工具 Placer"""
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

SEM = {
    "price":   "#E69F00",   # 电价
    "buy":     "#0072B2",   # 购电
    "adj":     "#56B4E9",   # 调整购电量
    "charge":  "#009E73",   # 充电
    "dischg":  "#D55E00",   # 放电
    "soc":     "#CC79A7",   # 储电量 SOC
    "pv":      "#F0E442",   # 光伏出力
    "pv_edge": "#B8A200",
    "load":    "#4D4D4D",   # 小区负载
    "emerg":   "#B2182B",   # 紧急购电
    "none":    "#9AA0A6",   # 无储能基准
    "ink":     "#333333",
    "ref":     "#8A8A8A",   # 参考线
}


# 接管 set_title / suptitle
import re as _re
_PAREN = _re.compile(r"[（(][^（()）]*[）)]")


def clean_title(s):
    """去掉标题里的括号及其内容"""
    if not isinstance(s, str):
        return s
    out = _PAREN.sub("", s)
    out = _re.sub(r"[ \t]{2,}", " ", out).strip()
    out = _re.sub(r"[：:]\s*$", "", out).strip()
    return out


def _install_title_cleaner():
    import matplotlib.axes as _ax
    import matplotlib.figure as _fg
    if not getattr(_ax.Axes.set_title, "_dsh_clean", False):
        _o = _ax.Axes.set_title
        def _set_title(self, label, *a, **k):
            return _o(self, clean_title(label), *a, **k)
        _set_title._dsh_clean = True
        _ax.Axes.set_title = _set_title
    if not getattr(_fg.Figure.suptitle, "_dsh_clean", False):
        _o2 = _fg.Figure.suptitle
        def _suptitle(self, t, *a, **k):
            return _o2(self, clean_title(t), *a, **k)
        _suptitle._dsh_clean = True
        _fg.Figure.suptitle = _suptitle


def apply_style():
    _install_title_cleaner()
    plt.rcParams.update({
        "font.family": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "axes.edgecolor": "#4A4A4A",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#DEDEDE",
        "grid.linewidth": 0.6,
        "grid.linestyle": ":",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 130,
    })


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


class Placer(object):
    # 方向偏好顺序
    ANGLES = [0, 90, 45, -90, 135, 180, -45, -135,
              20, -20, 70, 110, -70, -110, 160, -160,
              30, -30, 60, -60, 120, -120, 150, -150]

    def __init__(self, fig, ax, gap=9, pad=5):
        self.fig, self.ax = fig, ax
        self.gap, self.pad = gap, pad
        self.rects = []          # 显示坐标 (x0,y0,x1,y1)
        self.pts = []            # 显示坐标 Nx2
        self.placed = []
        self.warn = []
        self._sync()
        self.axbox = ax.get_window_extent()
        lg = ax.get_legend()
        if lg is not None:
            self.legend()
    def texts(self):
        """把轴内已有文字登记为障碍物"""
        self._sync()
        r = self.fig.canvas.get_renderer()
        skip = {id(self.ax.title), id(self.ax.xaxis.label), id(self.ax.yaxis.label)}
        for t in self.ax.texts:
            if id(t) in skip or not t.get_text().strip():
                continue
            try:
                bb = t.get_window_extent(r)
            except Exception:
                continue
            if bb.width > 0.5 and bb.height > 0.5:
                self.rects.append((bb.x0, bb.y0, bb.x1, bb.y1))

    def _sync(self):
        self.fig.canvas.draw()

    def bars(self, container):
        self._sync()
        items = getattr(container, "patches", container)
        for r in items:
            try:
                bb = r.get_window_extent()
            except Exception:
                continue
            self.rects.append((bb.x0, bb.y0, bb.x1, bb.y1))

    def legend(self):
        self._sync()
        lg = self.ax.get_legend()
        if lg is None:
            return
        bb = lg.get_window_extent()
        if bb.width > 1 and bb.height > 1:
            self.rects.append((bb.x0, bb.y0, bb.x1, bb.y1))

    @staticmethod
    def _densify(pts, step=3.0):
        """把折线按 <= step 像素加密"""
        if len(pts) < 2:
            return pts
        out = [pts[0]]
        for i in range(1, len(pts)):
            a, b = pts[i - 1], pts[i]
            d = float(np.hypot(b[0] - a[0], b[1] - a[1]))
            n = max(1, int(np.ceil(d / step)))
            for k in range(1, n + 1):
                out.append(a + (b - a) * (k / float(n)))
        return np.asarray(out)

    def line(self, x, y, ax=None):
        """登记折线/曲线为障碍物；从轴曲线传 ax=<从轴对象>"""
        src = ax if ax is not None else self.ax
        x = np.asarray(x, float).ravel()
        y = np.asarray(y, float).ravel()
        m = np.isfinite(x) & np.isfinite(y)
        if not m.any():
            return
        p = src.transData.transform(np.column_stack([x[m], y[m]]))
        if src is not self.ax:
            p = self.ax.transData.inverted().transform(p)
        self.pts.append(self._densify(p))

    def hline(self, yv, ax=None):
        """登记一条水平参考线"""
        src = ax if ax is not None else self.ax
        bb = src.get_window_extent()
        inv = src.transData.inverted()
        x0 = float(inv.transform([bb.x0, bb.y0])[0])
        x1 = float(inv.transform([bb.x1, bb.y0])[0])
        self.line(np.array([x0, (x0 + x1) / 2.0, x1]), np.full(3, float(yv)), ax=src)

    def _free(self, box):
        x0, y0, x1, y1 = box[0] - self.pad, box[1] - self.pad, box[2] + self.pad, box[3] + self.pad
        a = self.axbox
        if x0 < a.x0 + 1 or x1 > a.x1 - 1 or y0 < a.y0 + 1 or y1 > a.y1 - 1:
            return False
        for r in self.rects:
            if x0 < r[2] and r[0] < x1 and y0 < r[3] and r[1] < y1:
                return False
        for p in self.pts:
            if len(p) and ((p[:, 0] >= x0) & (p[:, 0] <= x1) &
                           (p[:, 1] >= y0) & (p[:, 1] <= y1)).any():
                return False
        for b in self.placed:
            if x0 < b[2] and b[0] < x1 and y0 < b[3] and b[1] < y1:
                return False
        return True

    def _arrow_ok(self, box, anchor, exclude_r=9.0, step=3.0):
        """引出线自框边到锚点是否无遮挡"""
        p1 = np.asarray(anchor, float)
        cx = min(max(p1[0], box[0]), box[2])
        cy = min(max(p1[1], box[1]), box[3])
        p0 = np.array([cx, cy], float)
        d = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
        if d < 1e-6:
            return True
        n = max(2, int(np.ceil(d / step)))
        ts = np.linspace(0.0, 1.0, n)
        seg = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
        keep = np.hypot(seg[:, 0] - p1[0], seg[:, 1] - p1[1]) > exclude_r
        seg = seg[keep]
        if len(seg) == 0:
            return True
        for r in self.rects:                      # 柱体 / 图例 / 已有文字
            if ((seg[:, 0] >= r[0]) & (seg[:, 0] <= r[2]) &
                    (seg[:, 1] >= r[1]) & (seg[:, 1] <= r[3])).any():
                return False
        for b in self.placed:                     # 已放下的其它标注
            if ((seg[:, 0] >= b[0]) & (seg[:, 0] <= b[2]) &
                    (seg[:, 1] >= b[1]) & (seg[:, 1] <= b[3])).any():
                return False
        for p in self.pts:                        # 数据折线 / 曲线
            if not len(p):
                continue
            dx = np.abs(p[:, 0][None, :] - seg[:, 0][:, None])
            if not (dx < 1.6).any():
                continue
            dy = np.abs(p[:, 1][None, :] - seg[:, 1][:, None])
            if ((dx < 1.6) & (dy < 1.6)).any():
                return False
        return True

    def _measure(self, text, fs, ls):
        t = self.ax.text(0, 0, text, fontsize=fs, linespacing=ls)
        self.fig.canvas.draw()
        bb = t.get_window_extent(self.fig.canvas.get_renderer())
        t.remove()
        return bb.width, bb.height

    def _search(self, w, h, ax_pt, radii, arrow):
        """枚举半径与方向，返回首个无碰撞位"""
        for r in radii:
            for ang in self.ANGLES:
                th = math.radians(ang)
                ux, uy = math.cos(th), math.sin(th)
                half = abs(ux) * w / 2.0 + abs(uy) * h / 2.0
                d = r + half
                cx, cy = ax_pt[0] + ux * d, ax_pt[1] + uy * d
                box = (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
                if self._free(box):
                    if arrow and not self._arrow_ok(box, ax_pt):
                        continue
                    return (box, cx, cy)
        return None

    def place(self, text, xy, fs=8.6, ls=1.3, color=None, arrow=True,
              gap=None, max_r=None, zorder=6, jitter=7.0):
        gap = self.gap if gap is None else gap
        w, h = self._measure(text, fs, ls)
        base = self.ax.transData.transform(np.asarray(xy, float))
        radii = [gap, gap + 8, gap + 16, gap + 26, gap + 38, gap + 54, gap + 74,
                 gap + 98, gap + 128, gap + 164]
        if max_r:
            radii = [r for r in radii if r <= max_r] or [max_r]
        # 锚点沿轴向微调 2.5 ~ jitter 像素
        offs = [(0.0, 0.0)]
        for d in (2.5, 5.0, jitter):
            offs += [(d, 0.0), (-d, 0.0), (0.0, d), (0.0, -d)]
        best = None
        for ox, oy in offs:
            got = self._search(w, h, base + np.array([ox, oy]), radii, arrow)
            if got:
                best = got
                break
        if best is None:                     # 兜底：取碰撞最少
            ax_pt = base
            worst = None
            for r in radii[:5]:
                for ang in self.ANGLES:
                    th = math.radians(ang)
                    ux, uy = math.cos(th), math.sin(th)
                    half = abs(ux) * w / 2.0 + abs(uy) * h / 2.0
                    d = r + half
                    cx, cy = ax_pt[0] + ux * d, ax_pt[1] + uy * d
                    box = (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)
                    n = 0
                    for rr in self.rects:
                        if box[0] < rr[2] and rr[0] < box[2] and box[1] < rr[3] and rr[1] < box[3]:
                            n += 1
                    for p in self.pts:
                        if len(p) and ((p[:, 0] >= box[0]) & (p[:, 0] <= box[2]) &
                                       (p[:, 1] >= box[1]) & (p[:, 1] <= box[3])).any():
                            n += 1
                    for b in self.placed:
                        if box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3]:
                            n += 1
                    if worst is None or n < worst[0]:
                        worst = (n, box, cx, cy)
            best = (worst[1], worst[2], worst[3])
            self.warn.append((text[:22], worst[0]))
        box, cx, cy = best
        self.placed.append(box)
        kw = dict(fontsize=fs, linespacing=ls, ha="center", va="center",
                  color=color or SEM["ink"], zorder=zorder,
                  textcoords="figure pixels", xytext=(cx, cy))
        if arrow:
            kw["arrowprops"] = dict(arrowstyle="-", color="#9A9A9A", lw=0.8,
                                    shrinkA=1, shrinkB=2)
        self.ax.annotate(text, xy=tuple(np.asarray(xy, float)), **kw)
        return box

    def report(self):
        if self.warn:
            print("告警：以下标注未能完全避让（残留碰撞数）")
            for t, n in self.warn:
                print("      %-26s %d" % (t, n))
        return not self.warn


def tag(ax, text, xy, xytext, color=None, ha="left", va="bottom", fs=8.6):
    """固定位置标注"""
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=fs, ha=ha, va=va,
                color=color or SEM["ink"],
                arrowprops=dict(arrowstyle="-|>", color="#8A8A8A", lw=0.9, shrinkA=2, shrinkB=3))
