#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Симулятор шахового хордового тора ШХТ(n; a, b) і порівняльних топологій. Версія 5.0.

Модель мережі міжвузлового обміну в кластері з N вузлів. Це НЕ реалізація розподіленого виконання YOLO26x:
навантаження (розміри повідомлень, FLOPs) береться з параметрів карт ознак і з model.info() користувача.

Реалізовано: топології (ШХТ, 2D-решітка, 2D/3D-тор, гіперкуб); ONE-TO-ONE (формульна відстань chess_plan, локальний вибір
порту); ONE-TO-ALL / ALL-TO-ALL (покривні дерева, хеш- і потокове балансування, два планувальники каналів);
ONE-TO-ONE-FT (фаза 1 з VC = номер переходу, фаза 2 escape-дерево); спільний CDG маршрутів і розсилок; обмін гранями без
замикання країв зображення; кадровий бюджет для YOLO26n/s/m/l/x; відображення ШХТ на гіперкуб.

Свідомо НЕ реалізовано (не мало коректного обґрунтування): динамічні відмови під час передавання, протокол реконфігурації
з оцінкою часу, повторні передачі. Гарантії стосуються лише статичного узгодженого знімка відмов.

Припущення: канали повнодуплексні, 1 фліт/такт; all-port; t_init, t_router, t_link — параметри SimConfig; черги необмежені
(скінченні буфери й backpressure не симулюються, безтупиковість перевіряється CDG); escape-ліс і каталог міток узгоджені до
початку операції; оцінка FLOPs за площею кадру не враховує дублювання обчислень на межах і ефективність апаратури.

Запуск: python simulator_final.py [--model n|s|m|l|x] [--quick] all|models|verify|broadcast|allgather|cdg|faults|traffic|
        halo|ft|deadlock|embed|yolo|directory
"""
from __future__ import annotations

import heapq
import math
import random
import statistics
import sys
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

# ======================================================================
# 1. Конфігурація
# ======================================================================


# ======================================================================
# 0. Моделі YOLO26 (Ultralytics) і параметри розбиття кадру
# ======================================================================
# Джерела параметрів: docs.ultralytics.com/models/yolo26 (таблиця: params, FLOPs для fused-моделі при 640x640)
# та yolo26.yaml (scales: [depth, width, max_channels]; рівень P3 основи: C3k2 з 512 каналами до масштабування).
# Канали шару: c = make_divisible(min(c_yaml, max_channels) * width, 8).
# Найменша модель сімейства — YOLO26n; далі s, m, l, x (x — найбільша, цільова для магістерської).

@dataclass(frozen=True)
class YoloModel:
    key: str
    width: float          # коефіцієнт ширини з yolo26.yaml
    max_channels: int     # обмеження каналів з yolo26.yaml
    gflops_640: float     # FLOPs (G) fused-моделі при 640x640 (документація Ultralytics)
    params_m: float       # параметри (M), fused-модель

YOLO26_MODELS: Dict[str, YoloModel] = {
    "n": YoloModel("n", 0.25, 1024, 6.2, 2.57),
    "s": YoloModel("s", 0.50, 1024, 23.1, 10.01),
    "m": YoloModel("m", 1.00, 512, 75.6, 21.90),
    "l": YoloModel("l", 1.00, 512, 94.2, 26.30),
    "x": YoloModel("x", 1.50, 512, 210.1, 58.99),
}
ACTIVE_MODEL = "x"            # модель за замовчуванням (змінюється set_model або опцією --model)
P3_YAML_CHANNELS = 512        # C3k2 рівня P3 основи (шар 4 у yolo26.yaml)
FRAME_W, FRAME_H = 1920, 1080 # Full HD
FRAME_H_PAD = 1088            # висота, доповнена до кратної 32 (letterbox)
GRID = 16                     # розбиття кадру 16 x 16 фрагментів (N = 256)
FP16_BYTES, FLIT_BYTES = 2, 8 # розрядність елемента та ширина каналу (64 біти)

def make_divisible(x: float, d: int = 8) -> int:
    return int(math.ceil(x / d) * d)

def set_model(key: str) -> None:
    global ACTIVE_MODEL
    if key not in YOLO26_MODELS:
        raise ValueError(f"невідома модель {key!r}; доступні: {', '.join(YOLO26_MODELS)}")
    ACTIVE_MODEL = key

def p3_channels(key: Optional[str] = None) -> int:
    m = YOLO26_MODELS[key or ACTIVE_MODEL]
    return make_divisible(min(P3_YAML_CHANNELS, m.max_channels) * m.width, 8)

def p3_tile(grid: int = GRID) -> Tuple[int, int]:
    """Розмір карти ознак P3 одного фрагмента (клітинок): ceil(Wf/8) x ceil(Hf/8)."""
    wf, hf = FRAME_W // grid, math.ceil(FRAME_H / grid)
    return math.ceil(wf / 8), math.ceil(hf / 8)

def model_flits(key: Optional[str] = None, grid: int = GRID) -> Tuple[int, int]:
    """(M, M_halo) у флітах: карта P3 фрагмента і крайовий ободок шириною 1 клітинка."""
    c = p3_channels(key)
    gw, gh = p3_tile(grid)
    m = math.ceil(gw * gh * c * FP16_BYTES / FLIT_BYTES)
    mh = math.ceil(2 * (gw + gh) * c * FP16_BYTES / FLIT_BYTES)
    return m, mh

def node_flops_per_frame(key: Optional[str] = None, nodes: int = GRID * GRID) -> float:
    """FLOPs одного вузла на кадр: FLOPs(640x640) * (W * H_pad) / 640^2 / N (оцінка за площею)."""
    m = YOLO26_MODELS[key or ACTIVE_MODEL]
    return m.gflops_640 * 1e9 * (FRAME_W * FRAME_H_PAD) / (640 * 640) / nodes


@dataclass(frozen=True)
class SimConfig:
    t_init: int = 10          # час ініціалізації передачі, такти
    t_router: int = 2         # затримка маршрутизатора на один перехід, такти
    t_link: int = 1           # затримка поширення по каналу, такти
    flits: int = field(default_factory=lambda: model_flits()[0])   # M, фліти: карта P3 фрагмента активної моделі
    freq_ghz: float = 1.0     # частота, ГГц
    frame_budget_ms: float = 1000.0 / 30.0   # кадровий бюджет при 30 к/с

    def ms(self, cycles: float) -> float:
        return cycles / (self.freq_ghz * 1e6)


# Оптимальні хорди з таблиці 1.1 записки (критерій: мін. D, далі мін. середня відстань).
OPTIMAL_CHORDS: Dict[int, Tuple[int, int]] = {
    4: (1, 2), 8: (1, 4), 10: (1, 4), 12: (2, 7), 16: (2, 5),
    20: (3, 10), 24: (4, 11), 28: (5, 14), 30: (6, 19),
}
# Очікувані (D, середній діаметр) з таблиці 1.1 — для регресійної перевірки.
TABLE_1_1: Dict[int, Tuple[int, float]] = {
    4: (3, 1.733), 8: (4, 2.540), 10: (4, 2.848), 12: (5, 3.133), 16: (5, 3.671),
    20: (6, 4.135), 24: (7, 4.553), 28: (7, 4.978), 30: (7, 5.086),
}

# ======================================================================
# 2. Відмови
# ======================================================================


@dataclass
class Faults:
    """Множина відмов: вузли та неорієнтовані канали (u, v), u < v."""
    nodes: Set[int] = field(default_factory=set)
    links: Set[Tuple[int, int]] = field(default_factory=set)

    def alive_node(self, u: int) -> bool:
        return u not in self.nodes

    def alive_link(self, u: int, v: int) -> bool:
        if u in self.nodes or v in self.nodes:
            return False
        return ((u, v) if u < v else (v, u)) not in self.links

    def count(self) -> int:
        return len(self.nodes) + len(self.links)


NO_FAULTS = Faults()


def sample_exact_faults(topo: "Topology", rng: random.Random, f_nodes: int,
                        f_links: int = 0, protect: Sequence[int] = ()) -> Faults:
    """Рівно f_nodes відмов вузлів і f_links відмов каналів (випадково, без повторів)."""
    cand_nodes = [u for u in range(topo.N) if u not in set(protect)]
    nodes = set(rng.sample(cand_nodes, min(f_nodes, len(cand_nodes))))
    edges = [e for e in topo.edges() if e[0] not in nodes and e[1] not in nodes]
    links = set(rng.sample(edges, min(f_links, len(edges))))
    return Faults(nodes, links)


def sample_poisson_faults(topo: "Topology", rng: random.Random, lam_node: float,
                          lam_link: float, horizon: float,
                          protect: Sequence[int] = ()) -> Faults:
    """
    Відмови з експоненційними часами життя елементів: елемент відмовив до моменту horizon, якщо
    його час життя Exp(lambda) < horizon. Імовірність відмови окремого елемента 1 - exp(-lambda*horizon);
    кількість відмов — сума незалежних бернуллівських величин (біноміальний розподіл), що при малих
    lambda*horizon добре наближається розподілом Пуассона. Інтенсивності — відмов на елемент за одиницю часу.
    """
    prot = set(protect)
    nodes = {u for u in range(topo.N)
             if u not in prot and lam_node > 0 and rng.expovariate(lam_node) < horizon}
    links = {e for e in topo.edges()
             if lam_link > 0 and rng.expovariate(lam_link) < horizon}
    return Faults(nodes, links)


# ======================================================================
# 3. Топології
# ======================================================================


class Topology:
    """Базовий клас. nbrs[u] — список сусідів у порядку портів."""
    name: str = ""
    N: int = 0
    nbrs: List[List[int]]
    prio: Optional[Tuple[int, ...]] = None   # порядок перевірки портів при виборі наступного кроку

    def dist(self, u: int, v: int) -> int:
        raise NotImplementedError

    def order(self, u: int):
        return self.prio if self.prio is not None else range(len(self.nbrs[u]))

    def next_hop(self, u: int, dst: int) -> Optional[int]:
        """Наступний вузол мінімального шляху без відмов (перший порт у порядку пріоритету)."""
        if u == dst:
            return None
        d = self.dist(u, dst)
        for p in self.order(u):
            v = self.nbrs[u][p]
            if self.dist(v, dst) == d - 1:
                return v
        raise RuntimeError("немає сусіда, що зменшує відстань")

    def edges(self) -> List[Tuple[int, int]]:
        if not hasattr(self, "_edges"):
            self._edges = sorted({(min(u, v), max(u, v)) for u in range(self.N) for v in self.nbrs[u]})
        return self._edges

    def degree(self) -> int:
        return len(self.nbrs[0])

    def grid_neighbors(self, u: int) -> Optional[List[int]]:
        """Чотири ортогональні сусіди (для обміну гранями фрагментів); None, якщо не застосовно."""
        return None

    def bfs(self, src: int, faults: Faults = NO_FAULTS) -> List[int]:
        """BFS (лише для верифікації/оцінки досяжності, не в алгоритмах маршрутизації)."""
        dist = [-1] * self.N
        if not faults.alive_node(src):
            return dist
        dist[src] = 0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in self.nbrs[u]:
                if dist[v] < 0 and faults.alive_link(u, v):
                    dist[v] = dist[u] + 1
                    q.append(v)
        return dist

    def metrics(self) -> Tuple[int, float]:
        """(діаметр, середній діаметр) повним BFS від усіх вузлів."""
        total, diam = 0, 0
        for s in range(self.N):
            d = self.bfs(s)
            total += sum(d)
            diam = max(diam, max(d))
        return diam, total / (self.N * (self.N - 1))


def _ring(n: int, t: int) -> int:
    t %= n
    return min(t, n - t)


@lru_cache(maxsize=None)
def chess_plan(n: int, a: int, b: int, c0: int, dx: int, dy: int) -> Tuple[int, int, int]:
    """
    Формульна довжина найкоротшого шляху в ШХТ(n; a, b) (без BFS; результат мемоізується, складність O(D^2) на пару).

    Вхід: колір джерела c0 (0 — білий, 1 — чорний) і відносний зсув (dx, dy) = (x_d - x_s, y_d - y_s) mod n.
    Ключова властивість: КОЖЕН перехід (і хордовий, і ортогональний) змінює колір вузла, тому знак
    хорди на кроці з номером i (i = 0, 1, ...) дорівнює s0 * (-1)^i, де s0 = +1 для білого джерела
    і -1 для чорного. Позиції 0, 2, 4, ... («парні») дають знак s0, позиції 1, 3, ... — знак -s0.

    Нехай використано alpha хорд типу A=(a,b) і beta хорд типу B=(b,a) (знак числа — знак внеску:
    додатне значення ставиться на парні позиції, від'ємне — на непарні). Тоді сумарний
    хордовий зсув дорівнює s0*(alpha*(a,b) + beta*(b,a)), а залишок покривається ортогональними
    кроками за кільцевими відстанями ring_n. Для K кроків потрібно:
        K >= |alpha|+|beta| + ring(dx - sx) + ring(dy - sy),
        ceil(K/2) >= P+ (кількість хорд на парних позиціях),  floor(K/2) >= P- (на непарних),
        K  ≡ (dx + dy) mod 2   (граф дводольний, n парне).
    Використання хорд одного типу з обома знаками дає лише зайві кроки (домінується), тому
    перебираються лише пари (alpha, beta), |alpha|+|beta| <= K_max = ring(dx)+ring(dy).
    Повертає (K, alpha, beta) для найменшого K (перший знайдений при рівності).
    """
    s0 = 1 if c0 == 0 else -1
    par = (dx + dy) & 1
    kmax = _ring(n, dx) + _ring(n, dy)
    best: Optional[Tuple[int, int, int]] = None
    for al in range(-kmax, kmax + 1):
        rest = kmax - abs(al)
        for be in range(-rest, rest + 1):
            c = abs(al) + abs(be)
            sx = s0 * (al * a + be * b)
            sy = s0 * (al * b + be * a)
            pp = max(al, 0) + max(be, 0)
            pm = max(-al, 0) + max(-be, 0)
            k = max(c + _ring(n, dx - sx) + _ring(n, dy - sy), 2 * pp - 1, 2 * pm)
            if (k - par) & 1:
                k += 1
            if best is None or k < best[0]:
                best = (k, al, be)
    return best  # type: ignore[return-value]


def canonical_chord(n: int, a: int, b: int) -> Tuple[int, int]:
    """(a,b), (b,a) та (-a,-b) дають ізоморфні графи; канонічний запис 0 <= a < b < n."""
    opts = []
    for p, q in ((a, b), (-a, -b)):
        p, q = p % n, q % n
        opts.append(tuple(sorted((p, q))))
    return min(opts)  # type: ignore[return-value]


class ChessChordalTorus(Topology):
    """ШХТ(n; a, b). Вузол u = x*n + y. Порти: 0:+X, 1:-X, 2:+Y, 3:-Y, 4: хорда (a,b), 5: хорда (b,a)
    (у нумерації записки це порти 1..6). Білий вузол ((x+y) парне) використовує +(a,b), +(b,a);
    чорний — -(a,b), -(b,a)."""

    def __init__(self, n: int, a: int, b: int):
        if n % 2 or (a + b) % 2 == 0:
            raise ValueError("потрібно: n парне, a+b непарне")
        self.n, self.a, self.b, self.N = n, a, b, n * n
        self.name = f"ШХТ({n};{a},{b})"
        self.nbrs = [self._neigh(u) for u in range(self.N)]
        for u in (0, 1):  # білий (0,0) і чорний (0,1): усі 6 сусідів різні, без петель
            if len(set(self.nbrs[u])) != 6 or u in self.nbrs[u]:
                raise ValueError("недопустима пара хорд")
        self.prio = (4, 5, 0, 1, 2, 3)   # спершу хорди, потім ортогональні порти

    def _id(self, x: int, y: int) -> int:
        return (x % self.n) * self.n + (y % self.n)

    def colour(self, u: int) -> int:
        x, y = divmod(u, self.n)
        return (x + y) & 1

    def _neigh(self, u: int) -> List[int]:
        x, y = divmod(u, self.n)
        s = 1 if (x + y) % 2 == 0 else -1
        a, b = self.a, self.b
        return [self._id(x + 1, y), self._id(x - 1, y), self._id(x, y + 1), self._id(x, y - 1),
                self._id(x + s * a, y + s * b), self._id(x + s * b, y + s * a)]

    def plan(self, u: int, v: int) -> Tuple[int, int, int]:
        ux, uy = divmod(u, self.n)
        vx, vy = divmod(v, self.n)
        return chess_plan(self.n, self.a, self.b, (ux + uy) & 1, (vx - ux) % self.n, (vy - uy) % self.n)

    def dist(self, u: int, v: int) -> int:
        return self.plan(u, v)[0]

    def grid_neighbors(self, u: int) -> Optional[List[int]]:
        return self.nbrs[u][:4]


class Mesh2D(Topology):
    def __init__(self, n: int):
        self.n, self.N = n, n * n
        self.name = f"2D-решітка {n}x{n}"
        self.nbrs = []
        for u in range(self.N):
            x, y = divmod(u, n)
            lst = []
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if 0 <= x + dx < n and 0 <= y + dy < n:
                    lst.append((x + dx) * n + y + dy)
            self.nbrs.append(lst)

    def dist(self, u: int, v: int) -> int:
        ux, uy = divmod(u, self.n)
        vx, vy = divmod(v, self.n)
        return abs(ux - vx) + abs(uy - vy)

    def degree(self) -> int:
        return 4

    def grid_neighbors(self, u: int) -> Optional[List[int]]:
        return self.nbrs[u]


class Torus(Topology):
    """k-вимірний тор dims = (d1, d2, ...); маршрутизація за порядком вимірів (DOR), на рівності — «+»."""

    def __init__(self, dims: Tuple[int, ...]):
        self.dims = tuple(dims)
        self.N = 1
        for d in self.dims:
            self.N *= d
        self.name = ("2D-тор " if len(dims) == 2 else f"{len(dims)}D-тор ") + "x".join(map(str, dims))
        self.nbrs = []
        for u in range(self.N):
            c = self._coords(u)
            lst = []
            for i, d in enumerate(self.dims):
                for s in (1, -1):
                    if d == 2 and s == -1:
                        continue
                    cc = list(c)
                    cc[i] = (cc[i] + s) % d
                    lst.append(self._index(cc))
            self.nbrs.append(lst)

    def _coords(self, u: int) -> List[int]:
        c = []
        for d in reversed(self.dims):
            u, r = divmod(u, d)
            c.append(r)
        return c[::-1]

    def _index(self, c: Sequence[int]) -> int:
        u = 0
        for d, ci in zip(self.dims, c):
            u = u * d + ci
        return u

    def dist(self, u: int, v: int) -> int:
        cu, cv = self._coords(u), self._coords(v)
        return sum(_ring(d, a - b) for d, a, b in zip(self.dims, cu, cv))

    def grid_neighbors(self, u: int) -> Optional[List[int]]:
        return self.nbrs[u] if len(self.dims) == 2 else None


class Hypercube(Topology):
    """Гіперкуб Q_k; маршрутизація e-cube (виправляється молодший біт)."""

    def __init__(self, k: int):
        self.k, self.N = k, 1 << k
        self.name = f"гіперкуб Q{k}"
        self.nbrs = [[u ^ (1 << i) for i in range(k)] for u in range(self.N)]

    def dist(self, u: int, v: int) -> int:
        return bin(u ^ v).count("1")


# ======================================================================
# 4. Вибір хорд (етап проєктування; BFS допустимий)
# ======================================================================


def find_optimal_chords(n: int) -> Tuple[int, int]:
    """Повний перебір канонічних допустимих (a, b); мін. D, далі мін. середня відстань.
    Завдяки автоморфізмам (зсув на вектор з парною сумою) достатньо BFS від (0,0) і (0,1)."""
    best, best_ab = None, None
    for a in range(n):
        for b in range(a + 1, n):
            if (a + b) % 2 == 0 or canonical_chord(n, a, b) != (a, b):
                continue
            try:
                T = ChessChordalTorus(n, a, b)
            except ValueError:
                continue
            d0, d1 = T.bfs(0), T.bfs(1)
            if min(d0) < 0:
                continue
            key = (max(max(d0), max(d1)), round((sum(d0) + sum(d1)) / (2 * (T.N - 1)), 9))
            if best is None or key < best:
                best, best_ab = key, (a, b)
    return best_ab  # type: ignore[return-value]


def chords_for(n: int) -> Tuple[int, int]:
    return OPTIMAL_CHORDS.get(n) or find_optimal_chords(n)


# ======================================================================
# 5. ONE-TO-ONE
# ======================================================================


@dataclass
class RouteResult:
    delivered: bool
    path: List[int]
    reason: str = ""

    @property
    def hops(self) -> int:
        return len(self.path) - 1


def route_one_to_one(topo: Topology, src: int, dst: int, faults: Faults = NO_FAULTS,
                     ttl: Optional[int] = None) -> RouteResult:
    """
    Локальна топологічно-орієнтована маршрутизація (без BFS і таблиць).
    У кожному вузлі: з сусідів із живим каналом обирається той, що зменшує формульну
    відстань до призначення (продуктивний порт; порядок: хорди, потім ортогональні).
    Якщо продуктивних живих портів немає, виконується обхід: обирається сусід за ключем
    (менше відвідувань, менша відстань, порт), що не дає зациклитись.
    """
    if not faults.alive_node(src) or not faults.alive_node(dst):
        return RouteResult(False, [src], "кінець маршруту несправний")
    d0 = topo.dist(src, dst)
    if ttl is None:
        ttl = 2 * d0 + 4 * faults.count() + 8
    cur, path, visits = src, [src], {src: 1}
    while cur != dst:
        if len(path) - 1 >= ttl:
            return RouteResult(False, path, "перевищено TTL")
        dcur = topo.dist(cur, dst)
        best: Optional[Tuple[Tuple[int, int, int, int], int]] = None
        for rank, p in enumerate(topo.order(cur)):
            v = topo.nbrs[cur][p]
            if not faults.alive_link(cur, v):
                continue
            dv = topo.dist(v, dst)
            key = (0 if dv < dcur else 1, visits.get(v, 0), dv, rank)
            if best is None or key < best[0]:
                best = (key, v)
        if best is None:
            return RouteResult(False, path, "вузол ізольований")
        cur = best[1]
        visits[cur] = visits.get(cur, 0) + 1
        path.append(cur)
    return RouteResult(True, path)


# ======================================================================
# 6. ONE-TO-ALL: покривне дерево
# ======================================================================


@dataclass
class Tree:
    root: int
    parent: Dict[int, int]
    depth: Dict[int, int]
    children: Dict[int, List[int]]
    unreachable: List[int]
    repair_rounds: int

    @property
    def height(self) -> int:
        return max(self.depth.values())

    @property
    def covered(self) -> int:
        return len(self.depth)


def build_broadcast_tree(topo: Topology, root: int, faults: Faults = NO_FAULTS,
                         balance: Union[bool, str] = False) -> Tree:
    """
    Фаза 1 (номінальне правило, локальне): батько вузла u — перший (за пріоритетом портів) живий
    сусід v, що вже входить у дерево і лежить на мінімальному шляху від u до кореня
    (dist(v, root) = dist(u, root) - 1). Вузол обчислює це лише за координатами та кольором.
    Без відмов це дерево найкоротших шляхів: глибина = ексцентриситет кореня,
    діти вузла v — ті його сусіди w, для яких next_hop(w, root) = v.
    balance=False: серед допустимих батьків береться перший за пріоритетом портів (як у DOR;
    навантаження на канали нерівномірне). balance=True: допустимі батьки (їх ≥ 1) індексуються в
    порядку портів і береться номер (7*root + 13*u) mod k — детермінований локальний вибір за
    ідентифікаторами, що розподіляє дерева різних коренів по різних портах.
    balance="flow": для вершинно-транзитивних топологій (ШХТ, тор, гіперкуб) вузол бере номер порту до батька
    з таблиці flow_balance_table за зсувом до базового кореня свого класу (розв'язується на етапі проєктування),
    що вирівнює навантаження каналів; якщо такий батько недоступний (відмова), діє хеш-правило.
    Фаза 2 (локальний ремонт при відмовах): «осиротілі» вузли за раунди приєднуються до
    сусіда, що вже в дереві (із мінімальною глибиною); кількість раундів — repair_rounds.
    """
    if not faults.alive_node(root):
        raise ValueError("корінь несправний")
    alive = [u for u in range(topo.N) if faults.alive_node(u)]
    depth: Dict[int, int] = {root: 0}
    parent: Dict[int, int] = {}
    droot = {u: topo.dist(u, root) for u in alive}
    ftab = flow_balance_table(topo) if balance == "flow" else None
    bmap = _base_map(topo) if ftab is not None else None
    for u in sorted(alive, key=lambda x: (droot[x], x)):
        if u == root:
            continue
        cands = []
        for p in topo.order(u):
            v = topo.nbrs[u][p]
            if v in depth and faults.alive_link(u, v) and droot.get(v, -1) == droot[u] - 1:
                cands.append(v)
        if cands:
            if ftab is not None:
                c_cls, w_base = bmap["to_base"](u, root)
                v0 = topo.nbrs[u][ftab[c_cls][w_base]]
                v = v0 if v0 in cands else cands[(7 * root + 13 * u) % len(cands)]
            elif balance:
                v = cands[(7 * root + 13 * u) % len(cands)]
            else:
                v = cands[0]
            parent[u] = v
            depth[u] = depth[v] + 1
    orphans = [u for u in alive if u not in depth]
    rounds = 0
    while orphans:
        attach = []
        for u in orphans:
            cands = [(depth[topo.nbrs[u][p]], rank, topo.nbrs[u][p])
                     for rank, p in enumerate(topo.order(u))
                     if topo.nbrs[u][p] in depth and faults.alive_link(u, topo.nbrs[u][p])]
            if cands:
                attach.append((u, min(cands)[2]))
        if not attach:
            break
        rounds += 1
        for u, v in attach:
            parent[u] = v
            depth[u] = depth[v] + 1
        done = {u for u, _ in attach}
        orphans = [u for u in orphans if u not in done]
    children: Dict[int, List[int]] = {}
    for u, v in parent.items():
        children.setdefault(v, []).append(u)
    for v in children:
        children[v].sort()
    return Tree(root, parent, depth, children, orphans, rounds)


def check_tree(topo: Topology, tree: Tree, faults: Faults = NO_FAULTS) -> bool:
    """Перевірка: кожне ребро дерева існує і живе, глибина = глибина батька + 1, циклів немає."""
    for u, v in tree.parent.items():
        if v not in topo.nbrs[u] or not faults.alive_link(u, v):
            return False
        if tree.depth[u] != tree.depth[v] + 1:
            return False
    return tree.depth[tree.root] == 0 and tree.root not in tree.parent


# ======================================================================
# 7. Дискретно-подійна модель каналів
# ======================================================================




def _subtree_heights(t: "Tree") -> Dict[int, int]:
    """Висота піддерева кожного вузла дерева розсилки (листок = 0)."""
    h: Dict[int, int] = {}
    for v in sorted(t.depth, key=lambda x: -t.depth[x]):
        h[v] = max([h[c] + 1 for c in t.children.get(v, ())] or [0])
    return h


def _collective_critical(trees: Dict[int, "Tree"], cfg: SimConfig, M: int,
                         load: Dict[Tuple[int, int], int]) -> int:
    """Планувальник каналів «критичний шлях»: коли канал вільний, першим передається повідомлення,
    за яким у дереві лишається найвищий піддерев (за рівності — менший номер кореня). Це зменшує простої
    каналів порівняно з порядком за часом готовності. Канал зайнятий M тактів від початку проходження голови."""
    hop = cfg.t_router + cfg.t_link
    sub = {r: _subtree_heights(t) for r, t in trees.items()}
    link_free: Dict[Tuple[int, int], int] = {}
    queue: Dict[Tuple[int, int], List] = {}
    ev: List[Tuple[int, int, int, int, Tuple[int, int]]] = [(cfg.t_init, 0, r, r, (-1, -1)) for r in trees]
    heapq.heapify(ev)
    t_done = cfg.t_init + M if trees else 0

    def try_start(link: Tuple[int, int], t: int) -> None:
        nonlocal t_done
        q = queue.get(link)
        if link_free.get(link, 0) > t or not q:
            return
        _, r, c = heapq.heappop(q)
        link_free[link] = t + M
        load[link] = load.get(link, 0) + 1
        head = t + hop
        t_done = max(t_done, head + M)
        heapq.heappush(ev, (head, 0, r, c, (-1, -1)))
        heapq.heappush(ev, (t + M, 1, r, 0, link))
    while ev:
        t, kind, r, v, link = heapq.heappop(ev)
        if kind == 1:
            try_start(link, t)
            continue
        for c in trees[r].children.get(v, ()):
            lk = (v, c)
            heapq.heappush(queue.setdefault(lk, []), ((-sub[r][c], r), r, c))
            try_start(lk, t)
    return t_done








# ======================================================================
# 8. Граф залежностей каналів (CDG)
# ======================================================================


def channel_dependency_graph(topo: Topology, vc_mode: str = "none") -> Dict[Tuple, Set[Tuple]]:
    """
    CDG маршрутизації без відмов за всіма парами (s, d). Канал = (u, v, vc).
    vc_mode: 'none' — один віртуальний канал; 'hop' — номер віртуального каналу = номер переходу
    (структурований пул буферів: залежності ведуть лише з класу i у клас i+1, тож цикли
    неможливі за побудовою; потрібно D_max віртуальних каналів).
    """
    deps: Dict[Tuple, Set[Tuple]] = {}
    for s in range(topo.N):
        for d in range(topo.N):
            if s == d:
                continue
            path = [s]
            while path[-1] != d:
                path.append(topo.next_hop(path[-1], d))  # type: ignore[arg-type]
            for i in range(len(path) - 2):
                vc1 = i if vc_mode == "hop" else 0
                vc2 = i + 1 if vc_mode == "hop" else 0
                c1 = (path[i], path[i + 1], vc1)
                c2 = (path[i + 1], path[i + 2], vc2)
                deps.setdefault(c1, set()).add(c2)
                deps.setdefault(c2, set())
    return deps


def cdg_cycle_nodes(deps: Dict[Tuple, Set[Tuple]]) -> int:
    """Алгоритм Кана: повертає залишок — кількість вершин, які не вдалося видалити. Залишок 0 означає ациклічність;
    ненульовий залишок НЕ дорівнює кількості вершин у циклах."""
    indeg = {c: 0 for c in deps}
    for c, outs in deps.items():
        for o in outs:
            indeg[o] += 1
    stack = [c for c, k in indeg.items() if k == 0]
    removed = 0
    while stack:
        c = stack.pop()
        removed += 1
        for o in deps[c]:
            indeg[o] -= 1
            if indeg[o] == 0:
                stack.append(o)
    return len(deps) - removed


# ======================================================================
# 9. Статистика
# ======================================================================


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Частка і 95% інтервал Вілсона (у відсотках)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * p, 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def mean_ci(xs: Sequence[float], z: float = 1.96) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    m = statistics.mean(xs)
    if len(xs) < 2:
        return m, 0.0
    return m, z * statistics.stdev(xs) / math.sqrt(len(xs))


# ======================================================================
# 10. Експерименти
# ======================================================================


def exp_verify(ns=(4, 6, 8, 10, 12, 16, 20, 24, 30), walk_full_max: int = 16, sample: int = 20000,
               seed: int = 1) -> bool:
    """E1. Формульна відстань і пройдені маршрути проти BFS (BFS лише як еталон) для всіх n;
    регресія характеристик до таблиці 1.1."""
    print("\n=== E1. Верифікація ONE-TO-ONE проти BFS ===")
    print(f"{'n':>3} {'N':>4} {'(a,b)':>9} | {'dist==BFS':>14} {'маршрути==BFS':>16} | {'D':>2} {'D̄':>7} | табл.1.1")
    ok_all = True
    rng = random.Random(seed)
    for n in ns:
        a, b = chords_for(n)
        T = ChessChordalTorus(n, a, b)
        bad_d, tot = 0, 0
        bfs_all = []
        for s in range(T.N):
            ref = T.bfs(s)
            bfs_all.append(ref)
            for t in range(T.N):
                tot += 1
                if T.dist(s, t) != ref[t]:
                    bad_d += 1
        if n <= walk_full_max:
            pairs = [(s, t) for s in range(T.N) for t in range(T.N)]
        else:
            pairs = [(rng.randrange(T.N), rng.randrange(T.N)) for _ in range(sample)]
        bad_w = 0
        for s, t in pairs:
            r = route_one_to_one(T, s, t)
            if not r.delivered or r.hops != bfs_all[s][t]:
                bad_w += 1
        D = max(max(x) for x in bfs_all)
        Dbar = sum(sum(x) for x in bfs_all) / (T.N * (T.N - 1))
        ref = TABLE_1_1.get(n)
        tag = "-" if ref is None else ("OK" if (ref[0] == D and abs(ref[1] - Dbar) < 6e-4) else f"РОЗБІЖНІСТЬ {ref}")
        print(f"{n:>3} {T.N:>4} {str((a, b)):>9} | {tot - bad_d:>7}/{tot:<7}{'' :>0} {len(pairs) - bad_w:>8}/{len(pairs):<7} | "
              f"{D:>2} {Dbar:>7.3f} | {tag}")
        ok_all &= (bad_d == 0 and bad_w == 0 and tag in ("-", "OK"))
    print("РЕЗУЛЬТАТ:", "усі перевірки пройдено" if ok_all else "ЄВІДХИЛЕННЯ — виправте алгоритм")
    return ok_all


def exp_broadcast(n: int = 16) -> bool:
    """E2. ONE-TO-ALL: коректність дерева, глибина = D для коренів обох кольорів, час за моделлю."""
    print("\n=== E2. ONE-TO-ALL (покривне дерево) ===")
    a, b = chords_for(n)
    T = ChessChordalTorus(n, a, b)
    cfg = SimConfig()
    ecc, ok = {}, True
    for r in range(T.N):
        tr = build_broadcast_tree(T, r)
        ok &= check_tree(T, tr) and tr.covered == T.N and tr.repair_rounds == 0
        ecc.setdefault(T.colour(r), set()).add(tr.height)
    D = T.metrics()[0]
    print(f"{T.name}: ексцентриситети коренів: білі {sorted(ecc[0])}, чорні {sorted(ecc[1])}, D = {D}")
    ok &= all(e == {D} for e in ecc.values())
    for r in (0, 1):
        res = simulate_collective(T, cfg, [r])
        pred = cfg.t_init + res["max_depth"] * (cfg.t_router + cfg.t_link) + cfg.flits
        print(f"корінь {r} ({'білий' if T.colour(r) == 0 else 'чорний'}): глибина {res['max_depth']}, "
              f"{res['cycles']} тактів ({cfg.ms(res['cycles']) * 1000:.3f} мкс), формула {pred}")
        ok &= (res["cycles"] == pred)
    print("РЕЗУЛЬТАТ:", "ОК" if ok else "ПОМИЛКА")
    return ok


def standard_topologies() -> List[Topology]:
    return [ChessChordalTorus(16, 2, 5), Mesh2D(16), Torus((16, 16)), Torus((8, 8, 4)), Hypercube(8)]


def exp_cdg(ns=(4, 8, 16)) -> None:
    """E4. Ациклічність CDG: один віртуальний канал проти «hop-class» віртуальних каналів."""
    print("\n=== E4. Граф залежностей каналів ===")
    for n in ns:
        a, b = chords_for(n)
        T = ChessChordalTorus(n, a, b)
        for mode in ("none", "hop"):
            deps = channel_dependency_graph(T, mode)
            left = cdg_cycle_nodes(deps)
            vcs = 1 if mode == "none" else T.metrics()[0]
            print(f"{T.name:<12} режим {mode:<4} віртуальних каналів: {vcs:>2} "
                  f"вершин CDG: {len(deps):>6}  у циклічній частині: {left:>6}  "
                  f"=> {'ациклічний' if left == 0 else 'ЦИКЛИ Є'}")
    print("Висновок: з одним віртуальним каналом цикли можливі; hop-class (D віртуальних каналів) їх усуває за побудовою.")


def evaluate_faults(topo: Topology, sampler: Callable[[random.Random], Faults], trials: int,
                    pairs: int, seed: int) -> Dict:
    """Монте-Карло: ONE-TO-ONE (серед досяжних пар) і ONE-TO-ALL для відмов із sampler."""
    rng = random.Random(seed)
    att = succ = 0
    stretch: List[float] = []
    full = 0
    depth_ratio: List[float] = []
    cov: List[float] = []
    for _ in range(trials):
        f = sampler(rng)
        alive = [u for u in range(topo.N) if f.alive_node(u)]
        bfs_cache: Dict[int, List[int]] = {}
        for _ in range(pairs):
            s, d = rng.sample(alive, 2)
            ref = bfs_cache.setdefault(s, topo.bfs(s, f))
            if ref[d] < 0:
                continue
            att += 1
            r = route_one_to_one(topo, s, d, f)
            if r.delivered:
                succ += 1
                stretch.append(r.hops / ref[d])
        root = rng.choice(alive)
        comp = sum(1 for x in topo.bfs(root, f) if x >= 0)
        tr = build_broadcast_tree(topo, root, f)
        cov.append(100.0 * tr.covered / comp)
        full += (tr.covered == comp)
        d0 = max(topo.bfs(root))
        depth_ratio.append(tr.height / d0)
    p, lo, hi = wilson(succ, att)
    sm, sc = mean_ci(stretch)
    dm, dc = mean_ci(depth_ratio)
    return dict(attempts=att, success=p, lo=lo, hi=hi, stretch=sm, stretch_ci=sc,
                bc_full=100.0 * full / trials, bc_cov=statistics.mean(cov), depth_ratio=dm, depth_ci=dc)


def _print_fault_header() -> None:
    print(f"{'відмов':>8} | {'доставка серед досяжних, %':>30} | {'подовж.':>9} | {'bcast повне покр.,%':>20} | {'h/h0':>6}")


def _print_fault_row(label: str, r: Dict) -> None:
    print(f"{label:>8} | {r['success']:>8.2f} [{r['lo']:>6.2f};{r['hi']:>6.2f}] (n={r['attempts']:<6}) | "
          f"{r['stretch']:>5.3f}±{r['stretch_ci']:<4.3f}| {r['bc_full']:>20.1f} | {r['depth_ratio']:>6.3f}")


def exp_faults(quick: bool = False, n: int = 16) -> None:
    """E5. Відмовостійкість: точна кількість відмов вузлів і пуассонівський потік."""
    trials, pairs = (60, 10) if quick else (500, 20)
    cfg = SimConfig()
    a, b = chords_for(n)
    T = ChessChordalTorus(n, a, b)
    print(f"\n=== E5. Відмовостійкість {T.name}: {trials} випробувань x {pairs} пар на точку ===")
    print("--- (а) точна кількість відмов вузлів ---")
    _print_fault_header()
    for f in (0, 5, 10, 20, 30, 40, 50):
        r = evaluate_faults(T, lambda rng, f=f: sample_exact_faults(T, rng, f), trials, pairs, seed=100 + f)
        _print_fault_row(str(f), r)
    print("--- (б) відмови каналів (порти), вузли справні ---")
    _print_fault_header()
    for f in (0, 10, 30, 60, 100, 150):
        r = evaluate_faults(T, lambda rng, f=f: sample_exact_faults(T, rng, 0, f), trials, pairs, seed=200 + f)
        _print_fault_row(str(f), r)
    print("--- (в) пуассонівський потік на інтервалі T=1: lambda_вузла = 2*lambda_каналу; підпис = очік. відмов вузлів/каналів ---")
    _print_fault_header()
    n_edges = len(T.edges())
    for lt in (0.002, 0.005, 0.01, 0.02, 0.04):
        lam_l, lam_n = lt, 2 * lt
        r = evaluate_faults(T, lambda rng, ln=lam_n, ll=lam_l: sample_poisson_faults(T, rng, ln, ll, 1.0),
                            trials, pairs, seed=300 + int(lt * 1000))
        en, el = T.N * (1 - math.exp(-lam_n)), n_edges * (1 - math.exp(-lam_l))
        _print_fault_row(f"{en:.0f}в/{el:.0f}к", r)
    print("--- (г) час all-gather при відмовах (кілька випробувань, оцінка за моделлю з ремонтом дерев) ---")
    reps = 2 if quick else 5
    base = simulate_collective(T, cfg, list(range(T.N)), balance=True)["cycles"]
    print(f"{'відмов':>8} | {'такти (середнє)':>16} | {'відн. до 0':>10} | {'макс.глибина':>12} | недосяжних")
    rng = random.Random(7)
    for f in (0, 5, 10, 20, 40):
        cyc, dep, unr = [], [], []
        for _ in range(reps):
            fl = sample_exact_faults(T, rng, f)
            roots = [u for u in range(T.N) if fl.alive_node(u)]
            res = simulate_collective(T, cfg, roots, fl, balance=True)
            cyc.append(res["cycles"])
            dep.append(res["max_depth"])
            unr.append(res["unreachable"])
        print(f"{f:>8} | {statistics.mean(cyc):>16.0f} | {statistics.mean(cyc) / base:>10.3f} | {max(dep):>12} | {sum(unr)}")


def exp_traffic(quick: bool = False) -> None:
    """E6. Одночасна передача: кожен вузол шле повідомлення випадковому адресату (випадкова
    перестановка без нерухомих точок); конфлікти за канали враховано. Порівняння топологій."""
    print(f"\n=== E6. Випадкова перестановка (всі вузли одночасно), M = {SimConfig().flits} флітів (YOLO26{ACTIVE_MODEL}) ===")
    cfg = SimConfig()
    reps = 5 if quick else 30
    print(f"{'топологія':<22}{'середня затримка':>18}{'макс. затримка':>16}{'сер. шлях (переходи)':>22}")
    for T in standard_topologies():
        rng = random.Random(11)
        means, maxs, hops = [], [], []
        for _ in range(reps):
            perm = list(range(T.N))
            while True:
                rng.shuffle(perm)
                if all(perm[i] != i for i in range(T.N)):
                    break
            msgs = [(i, perm[i], 0) for i in range(T.N)]
            lat = simulate_traffic(T, cfg, msgs)
            means.append(statistics.mean(lat))   # type: ignore[arg-type]
            maxs.append(max(lat))                # type: ignore[type-var]
            hops.append(statistics.mean(T.dist(i, perm[i]) for i in range(T.N)))
        print(f"{T.name:<22}{statistics.mean(means):>18.0f}{statistics.mean(maxs):>16.0f}{statistics.mean(hops):>22.3f}")
    T = ChessChordalTorus(16, 2, 5)
    lat = simulate_traffic(T, cfg, [(0, 137, 0)])[0]
    h = T.dist(0, 137)
    pred = cfg.t_init + h * (cfg.t_router + cfg.t_link) + cfg.flits
    print(f"Перевірка формули (3.1): H={h}, симуляція {lat}, формула {pred} -> {'ОК' if lat == pred else 'ПОМИЛКА'}")




# ======================================================================
# 9а. Відмовостійкий ONE-TO-ONE з гарантією доставки та безтупиковістю (E8-E11)
# ======================================================================
#
# Двофазна схема (ONE-TO-ONE-FT):
#   Фаза 1 (швидка, безтабличної): локальне правило route_one_to_one (ключ kappa) не довше
#           HC переходів; на переході h пакет використовує віртуальний канал VC = h.
#   Фаза 2 (escape): якщо призначення не досягнуто за HC переходів, пакет переходить на
#           ОДИН виділений віртуальний канал VC = HC і далі рухається по escape-дереву
#           (дерево ONE-TO-ALL п. 2.3 з локальним ремонтом) вгору до найменшого спільного
#           предка і вниз до призначення. Рішення приймаються локально за мітками інтервалів
#           (pre/post-order), які вузли отримують під час реконфігурації.
# Гарантія доставки: escape-дерево охоплює всю компоненту зв'язності залишкового графа,
#   тому для будь-якої пари вузлів з однієї компоненти пакет доставляється за не більш ніж
#   HC + depth(cur) + depth(dst) переходів; для пар з різних компонент це виявляється
#   на джерелі (різні корені дерев) і пакет не відправляється.
# Безтупиковість: канали впорядковані (VC=0 < 1 < ... < HC-1 < escape-канали вгору < вниз),
#   кожен маршрут використовує їх у строго зростаючому порядку, тому CDG ациклічний
#   (умова Далі-Зейтца), що перевіряється програмно (check_route_cdg).


@dataclass
class EscapeForest:
    """Escape-ліс: по одному дереву на компоненту зв'язності залишкового графа."""
    trees: List[Tree]
    comp: Dict[int, int]                 # вузол -> номер дерева (компоненти)
    pre: Dict[int, int]                  # мітка pre-order
    post: Dict[int, int]                 # найбільша мітка у піддереві вузла
    kids: Dict[int, List[int]]           # діти, впорядковані за pre

    def same_component(self, u: int, v: int) -> bool:
        return u in self.comp and v in self.comp and self.comp[u] == self.comp[v]

    def next_hop(self, cur: int, dst: int) -> int:
        """Локальне рішення escape-маршрутизації (за мітками, без таблиць маршрутів)."""
        if self.pre[cur] <= self.pre[dst] <= self.post[cur]:
            for c in self.kids[cur]:
                if self.pre[c] <= self.pre[dst] <= self.post[c]:
                    return c
            raise RuntimeError("призначення у піддереві, але серед дітей не знайдено")
        return self.trees[self.comp[cur]].parent[cur]


def build_escape_forest(topo: Topology, faults: Faults = NO_FAULTS) -> EscapeForest:
    """
    Реконфігурація при зміні множини відмов. Корінь дерева компоненти — вузол з найменшим
    номером (у мережі це вибір мінімального ідентифікатора розсиланням). Дерево будується
    правилом п. 2.3 з локальним ремонтом; далі мітки pre/post обчислюються обходом
    (у мережі — зворотне збирання розмірів піддерев і розсилка інтервалів).
    """
    trees: List[Tree] = []
    comp: Dict[int, int] = {}
    for u in range(topo.N):
        if not faults.alive_node(u) or u in comp:
            continue
        t = build_broadcast_tree(topo, u, faults, balance=False)
        for w in t.depth:
            comp[w] = len(trees)
        trees.append(t)
    pre: Dict[int, int] = {}
    post: Dict[int, int] = {}
    kids: Dict[int, List[int]] = {}
    counter = 0
    for t in trees:
        stack: List[Tuple[int, int]] = [(t.root, 0)]
        while stack:
            u, i = stack.pop()
            ch = t.children.get(u, [])
            if i == 0:
                pre[u] = counter
                counter += 1
                kids[u] = list(ch)
            if i < len(ch):
                stack.append((u, i + 1))
                stack.append((ch[i], 0))
            else:
                post[u] = counter - 1
    return EscapeForest(trees, comp, pre, post, kids)


@dataclass
class FTRoute:
    delivered: bool
    path: List[int]
    vcs: List[int]            # віртуальний канал кожного переходу
    escaped: bool = False     # чи використано фазу 2
    reason: str = ""

    @property
    def hops(self) -> int:
        return len(self.path) - 1


def route_one_to_one_ft(topo: Topology, src: int, dst: int, faults: Faults,
                        forest: EscapeForest, hc: int) -> FTRoute:
    """ONE-TO-ONE-FT: гарантована доставка в межах компоненти зв'язності залишкового графа."""
    if hc < 0:
        raise ValueError("hc має бути невід'ємним")
    if not faults.alive_node(src) or not faults.alive_node(dst):
        return FTRoute(False, [src], [], False, "кінець маршруту несправний")
    if src == dst:
        return FTRoute(True, [src], [])
    if not forest.same_component(src, dst):
        return FTRoute(False, [src], [], False, "недосяжне (різні компоненти)")
    cur, path, vcs, visits = src, [src], [], {src: 1}
    while cur != dst and len(path) - 1 < hc:          # фаза 1
        dcur = topo.dist(cur, dst)
        best: Optional[Tuple[Tuple[int, int, int, int], int]] = None
        for rank, p in enumerate(topo.order(cur)):
            v = topo.nbrs[cur][p]
            if not faults.alive_link(cur, v):
                continue
            dv = topo.dist(v, dst)
            key = (0 if dv < dcur else 1, visits.get(v, 0), dv, rank)
            if best is None or key < best[0]:
                best = (key, v)
        if best is None:
            break
        vcs.append(len(path) - 1)
        cur = best[1]
        visits[cur] = visits.get(cur, 0) + 1
        path.append(cur)
    escaped = False
    while cur != dst:                                  # фаза 2 (escape)
        escaped = True
        nxt = forest.next_hop(cur, dst)
        vcs.append(hc)
        cur = nxt
        path.append(cur)
    return FTRoute(True, path, vcs, escaped)


def route_cdg_add(deps: Dict[Tuple, Set[Tuple]], path: Sequence[int], vcs: Sequence[int],
                  single_vc: bool = False) -> None:
    """Додає залежності каналів (u, v, vc) одного маршруту до CDG."""
    for i in range(len(path) - 2):
        c1 = (path[i], path[i + 1], 0 if single_vc else vcs[i])
        c2 = (path[i + 1], path[i + 2], 0 if single_vc else vcs[i + 1])
        deps.setdefault(c1, set()).add(c2)
        deps.setdefault(c2, set())


def tree_cdg(topo: Topology, forest: EscapeForest) -> Dict[Tuple, Set[Tuple]]:
    """CDG чистого escape-режиму для ВСІХ пар вузлів кожної компоненти (один віртуальний канал)."""
    deps: Dict[Tuple, Set[Tuple]] = {}
    for t in forest.trees:
        nodes = list(t.depth)
        for s in nodes:
            for d in nodes:
                if s == d:
                    continue
                path = [s]
                while path[-1] != d:
                    path.append(forest.next_hop(path[-1], d))
                route_cdg_add(deps, path, [0] * (len(path) - 1))
    return deps


def multicast_cdg(topo: Topology, faults: Faults, roots: Sequence[int],
                  balance: Union[bool, str] = True) -> Tuple[Dict[Tuple, Set[Tuple]], int]:
    """CDG розсилки за деревами з VC = глибина дитини; повертає (CDG, max висота дерева)."""
    deps: Dict[Tuple, Set[Tuple]] = {}
    hmax = 0
    for r in roots:
        t = build_broadcast_tree(topo, r, faults, balance=balance)
        hmax = max(hmax, t.height)
        for u, p in t.parent.items():
            c1 = (p, u, t.depth[u] - 1)
            deps.setdefault(c1, set())
            for w in t.children.get(u, ()):
                c2 = (u, w, t.depth[w] - 1)
                deps[c1].add(c2)
                deps.setdefault(c2, set())
    return deps, hmax


def evaluate_faults_ft(topo: Topology, sampler: Callable[[random.Random], Faults], trials: int,
                       pairs: int, seed: int, hc: int) -> Dict:
    """Монте-Карло: порівняння route_one_to_one (E5) і ONE-TO-ONE-FT на одних і тих самих парах."""
    rng = random.Random(seed)
    reach = old_ok = ft_ok = unreach = unreach_ok = esc = 0
    stretch: List[float] = []
    longest = 0
    for _ in range(trials):
        f = sampler(rng)
        alive = [u for u in range(topo.N) if f.alive_node(u)]
        forest = build_escape_forest(topo, f)
        cache: Dict[int, List[int]] = {}
        for _ in range(pairs):
            s, d = rng.sample(alive, 2)
            ref = cache.setdefault(s, topo.bfs(s, f))
            r = route_one_to_one_ft(topo, s, d, f, forest, hc)
            if ref[d] < 0:
                unreach += 1
                unreach_ok += (not r.delivered and r.reason.startswith("недосяжне"))
                continue
            reach += 1
            old_ok += route_one_to_one(topo, s, d, f).delivered
            if r.delivered:
                ft_ok += 1
                stretch.append(r.hops / ref[d])
                esc += r.escaped
                longest = max(longest, r.hops)
    sm, sc = mean_ci(stretch)
    return dict(reach=reach, old=old_ok, ft=ft_ok, unreach=unreach, unreach_ok=unreach_ok,
                esc=esc, stretch=sm, stretch_ci=sc, longest=longest)


def _adversarial_faults(topo: Topology, rng: random.Random, k_links: int) -> Tuple[Faults, int]:
    """k_links із 6 каналів випадкового вузла-призначення виходять з ладу."""
    dst = rng.randrange(topo.N)
    nb = list(topo.nbrs[dst])
    links = {(min(dst, v), max(dst, v)) for v in rng.sample(nb, k_links)}
    return Faults(set(), links), dst


def exp_ft(quick: bool = False, n: int = 16) -> None:
    """E8. ONE-TO-ONE-FT: гарантія доставки, виявлення недосяжності, ціна (подовження, частка escape)."""
    trials, pairs = (40, 10) if quick else (200, 20)
    a, b = chords_for(n)
    T = ChessChordalTorus(n, a, b)
    D, _ = T.metrics()
    print(f"\n=== E8. ONE-TO-ONE-FT на {T.name}: {trials} випробувань x {pairs} пар; HC = діаметр + запас ===")
    hdr = (f"{'сценарій':<16}{'HC':>3} | {'досяжних':>8} | {'старий, %':>9} | {'FT, %':>7} | "
           f"{'недосяжн.':>9} | {'виявл.':>7} | {'подовж.':>12} | {'escape, %':>9} | {'макс.шлях':>9}")
    print(hdr)
    scen = [("5 вузлів", lambda rng: sample_exact_faults(T, rng, 5), 111),
            ("20 вузлів", lambda rng: sample_exact_faults(T, rng, 20), 112),
            ("50 вузлів", lambda rng: sample_exact_faults(T, rng, 50), 113),
            ("60 каналів", lambda rng: sample_exact_faults(T, rng, 0, 60), 114),
            ("150 каналів", lambda rng: sample_exact_faults(T, rng, 0, 150), 115),
            ("100к+30в", lambda rng: sample_exact_faults(T, rng, 30, 100), 116)]
    for hc in (D, D + 2, D + 4):
        for name, smp, seed in scen:
            r = evaluate_faults_ft(T, smp, trials, pairs, seed, hc)
            print(f"{name:<16}{hc:>3} | {r['reach']:>8} | {100 * r['old'] / r['reach']:>9.2f} | "
                  f"{100 * r['ft'] / r['reach']:>7.2f} | {r['unreach']:>9} | "
                  f"{(str(r['unreach_ok']) + '/' + str(r['unreach'])):>7} | "
                  f"{r['stretch']:>6.3f}±{r['stretch_ci']:<5.3f} | {100 * r['esc'] / max(r['ft'], 1):>9.2f} | {r['longest']:>9}")
    # змагальний сценарій: k із 6 каналів призначення відмовили (k=5 — шлях існує, k=6 — вузол відрізано)
    print("--- змагальний сценарій: відмовили k з 6 каналів вузла-призначення (HC = D + 2) ---")
    rng = random.Random(2026)
    hc = D + 2
    for k in (4, 5, 6):
        reach = old_ok = ft_ok = det = unreach = 0
        stretch: List[float] = []
        for _ in range(trials * 2):
            f, dst = _adversarial_faults(T, rng, k)
            forest = build_escape_forest(T, f)
            for _ in range(pairs):
                s = rng.randrange(T.N)
                if s == dst:
                    continue
                ref = T.bfs(s, f)
                r = route_one_to_one_ft(T, s, dst, f, forest, hc)
                if ref[dst] < 0:
                    unreach += 1
                    det += (not r.delivered)
                    continue
                reach += 1
                old_ok += route_one_to_one(T, s, dst, f).delivered
                ft_ok += r.delivered
                if r.delivered:
                    stretch.append(r.hops / ref[dst])
        sm = statistics.mean(stretch) if stretch else float("nan")
        print(f"k={k}: досяжних {reach}, старий алгоритм {100 * old_ok / max(reach, 1):.2f} %, "
              f"FT {100 * ft_ok / max(reach, 1):.2f} %; недосяжних {unreach}, виявлено {det}; подовження {sm:.3f}")







# ======================================================================
# 11. Потокове балансування дерев розсилки (вершинно-транзитивні топології)
# ======================================================================
#
# ШХТ, тор і гіперкуб мають групу зсувів, що діє на вузли транзитивно (у ШХТ — зсуви з парною сумою
# координат; два класи коренів: білі та чорні). Тому дерево розсилки від довільного кореня — зсув
# базового дерева свого класу, і навантаження напрямленого каналу залежить лише від його «орбіти»
# (клас батька, порт). Вибір батьків у базових деревах — це задача розподілу з обмеженням ємності
# (рівномірний розподіл дуг дерева по орбітах), що розв'язується ОДИН РАЗ на етапі проєктування
# (алгоритм Куна для b-паросполучення). На етапі роботи вузол u для кореня r за координатами обчислює
# зсув до базового вузла w і за таблицею (2(N-1) значень по 3 біти) бере номер порту до батька.
# Для топологій без цієї симетрії (2D-решітка) таблицю не будують — використовується хеш-правило.

def _base_map(topo: "Topology"):
    if hasattr(topo, "_base_map_cache"):
        return topo._base_map_cache  # type: ignore[attr-defined]
    bm = None
    if isinstance(topo, ChessChordalTorus):
        n = topo.n

        def to_base(u: int, r: int) -> Tuple[int, int]:
            c = topo.colour(r)
            ux, uy = divmod(u, n)
            rx, ry = divmod(r, n)
            return c, ((ux - rx) % n) * n + ((uy - ry + c) % n)
        bm = dict(classes=(0, 1), base={0: 0, 1: 1}, to_base=to_base, cls=topo.colour)
    elif isinstance(topo, Torus):
        dims = topo.dims

        def to_base(u: int, r: int) -> Tuple[int, int]:
            cu, cr = topo._coords(u), topo._coords(r)
            return 0, topo._index([(a - b) % d for a, b, d in zip(cu, cr, dims)])
        bm = dict(classes=(0,), base={0: 0}, to_base=to_base, cls=lambda u: 0)
    elif isinstance(topo, Hypercube):
        bm = dict(classes=(0,), base={0: 0}, to_base=lambda u, r: (0, u ^ r), cls=lambda u: 0)
    topo._base_map_cache = bm  # type: ignore[attr-defined]
    return bm


def flow_balance_table(topo: "Topology") -> Optional[Dict[int, Dict[int, int]]]:
    """tab[клас кореня][базовий вузол w] = номер порту вузла w до його батька в базовому дереві.
    Обирається так, щоб навантаження орбіт каналів було максимально рівномірним (мінімум max-навантаження)."""
    if hasattr(topo, "_flow_tab"):
        return topo._flow_tab  # type: ignore[attr-defined]
    bm = _base_map(topo)
    if bm is None:
        topo._flow_tab = None  # type: ignore[attr-defined]
        return None
    tasks: List[Tuple[int, int]] = []
    opts: List[List[Tuple[Tuple[int, int], int]]] = []
    for c in bm["classes"]:
        b = bm["base"][c]
        for w in range(topo.N):
            if w == b:
                continue
            dw = topo.dist(w, b)
            o = []
            for port, v in enumerate(topo.nbrs[w]):
                if topo.dist(v, b) == dw - 1:
                    o.append(((bm["cls"](v), topo.nbrs[v].index(w)), port))
            tasks.append((c, w))
            opts.append(o)
    orbits = sorted({orb for o in opts for orb, _ in o})
    by_cls: Dict[int, int] = {}
    n_orb: Dict[int, int] = {}
    for o in orbits:
        n_orb[o[0]] = n_orb.get(o[0], 0) + 1
    for o in opts:
        pc = o[0][0][0]
        by_cls[pc] = by_cls.get(pc, 0) + 1
    L = max(math.ceil(by_cls[pc] / n_orb[pc]) for pc in by_cls)

    def solve(cap: int):
        load = {o: 0 for o in orbits}
        mem: Dict[Tuple[int, int], List[int]] = {o: [] for o in orbits}
        where: Dict[int, Tuple[int, int]] = {}

        def aug(t: int, seen: Set) -> bool:
            for o, _ in opts[t]:
                if o in seen:
                    continue
                seen.add(o)
                if load[o] < cap:
                    load[o] += 1
                    mem[o].append(t)
                    where[t] = o
                    return True
                for t2 in list(mem[o]):
                    if aug(t2, seen):
                        mem[o].remove(t2)
                        mem[o].append(t)
                        where[t] = o
                        return True
            return False
        for t in range(len(tasks)):
            if not aug(t, set()):
                return None
        return where

    while True:
        where = solve(L)
        if where is not None:
            break
        L += 1
    tab: Dict[int, Dict[int, int]] = {c: {} for c in bm["classes"]}
    for t, (c, w) in enumerate(tasks):
        tab[c][w] = next(p for orb, p in opts[t] if orb == where[t])
    topo._flow_tab = tab  # type: ignore[attr-defined]
    topo._flow_L = L      # type: ignore[attr-defined]
    return tab


def ideal_allgather_estimate(topo: "Topology", cfg: SimConfig, flits: Optional[int] = None) -> Optional[int]:
    """Оцінка часу all-gather (такти) для відомих алгоритмів без конфліктів за канали.
    Гіперкуб Q_k: k копій рекурсивного подвоєння з частинами M/k, зсунутими за вимірами:
        T = k*(t_init+t_router+t_link) + (N-1)*M/k.
    k-вимірний тор: послідовні півкільцеві all-gather по вимірах, дані розбито на k частин з циклічними
    порядками вимірів: T = sum_j ceil((L_j-1)/2)*(t_init+t_router+t_link) + (N-1)*M/S.
    Це ОПТИМІСТИЧНА оцінка (для нерівних вимірів розклади частин не синхронні). Для ШХТ і 2D-решітки
    відомого оптимального алгоритму немає — повертає None."""
    M = flits if flits is not None else cfg.flits
    to = cfg.t_init + cfg.t_router + cfg.t_link
    if isinstance(topo, Hypercube):
        return topo.k * to + math.ceil((topo.N - 1) * M / topo.k)
    if isinstance(topo, Torus):
        lat = sum(math.ceil((d - 1) / 2) for d in topo.dims)
        return lat * to + math.ceil((topo.N - 1) * M / topo.degree())
    return None


ALLGATHER_MODES: Tuple[Tuple[str, Union[bool, str], str], ...] = (
    ("пріоритет портів", False, "fifo"),
    ("хеш-баланс", True, "fifo"),
    ("потоковий баланс", "flow", "fifo"),
    ("потік + крит. шлях", "flow", "critical"),
)


def exp_allgather_modes(T: "Topology", cfg: SimConfig) -> List[Dict]:
    """Результати all-gather для способів вибору батька (+ планувальник каналів)."""
    rows = []
    for tag, bal, sc in ALLGATHER_MODES:
        if bal == "flow" and flow_balance_table(T) is None:
            continue
        r = simulate_collective(T, cfg, list(range(T.N)), balance=bal, sched=sc)
        r["mode"] = tag
        rows.append(r)
    return rows

def simulate_paths(topo, cfg, messages, paths, faults=NO_FAULTS):
    """messages = (src, dst, t0, flits). Незалежні напрямлені канали з чергами; повертає затримки (такти).
    Канал зайнятий flits тактів від початку проходження голови; арбітраж — за часом готовності, потім за номером."""
    if len(messages) != len(paths):
        raise ValueError("кількість шляхів не збігається з кількістю повідомлень")
    heap = []
    res = [None] * len(messages)
    for i, ((src, dst, t0, m), p) in enumerate(zip(messages, paths)):
        if m <= 0 or t0 < 0:
            raise ValueError("некоректний розмір або час старту повідомлення")
        if p is None:
            continue
        if not p or p[0] != src or p[-1] != dst:
            raise ValueError("шлях не з'єднує джерело з призначенням")
        for u, v in zip(p, p[1:]):
            if v not in topo.nbrs[u] or not faults.alive_link(u, v):
                raise ValueError("шлях містить відсутній або несправний канал")
        if len(p) == 1:
            res[i] = 0
        else:
            heapq.heappush(heap, (t0 + cfg.t_init, i, 0))
    free = {}
    while heap:
        t, i, h = heapq.heappop(heap)
        src, dst, t0, m = messages[i]
        p = paths[i]
        link = (p[h], p[h + 1])
        start = max(t, free.get(link, 0))
        free[link] = start + m
        head = start + cfg.t_router + cfg.t_link
        if h + 2 == len(p):
            res[i] = head + m - t0
        else:
            heapq.heappush(heap, (head, i, h + 1))
    return res


def simulate_traffic(topo, cfg, messages, faults=NO_FAULTS, flits=None):
    """ONE-TO-ONE трафік: messages = [(src, dst, t0)]. Без відмов — route_one_to_one; з відмовами — ONE-TO-ONE-FT для
    статичного знімка. Затримка без конфліктів: T = t_init + H*(t_router + t_link) + M."""
    M = flits if flits is not None else cfg.flits
    paths = []
    if faults.count() == 0:
        for s, d, _ in messages:
            r = route_one_to_one(topo, s, d, faults)
            paths.append(r.path if r.delivered else None)
    else:
        forest = build_escape_forest(topo, faults)
        hc = _diam_fast(topo) + 2
        for s, d, _ in messages:
            r = route_one_to_one_ft(topo, s, d, faults, forest, hc)
            paths.append(r.path if r.delivered else None)
    return simulate_paths(topo, cfg, [(s, d, t, M) for s, d, t in messages], paths, faults)


def side_flits(axis, key=None, grid=GRID):
    """Розмір грані карти P3 у флітах. axis=0 — сусід по x (вертикальна грань, gh клітинок), axis=1 — сусід по y (gw клітинок)."""
    gw, gh = p3_tile(grid)
    cells = gh if axis == 0 else gw
    return math.ceil(cells * p3_channels(key) * FP16_BYTES / FLIT_BYTES)


def halo_plan(topo, key=None, grid=GRID):
    """Кожен вузол (x, y) надсилає відповідну грань кожному існуючому сусідові сітки grid x grid; замикання країв НЕМАЄ."""
    two_d = isinstance(topo, (ChessChordalTorus, Mesh2D)) or (isinstance(topo, Torus) and len(topo.dims) == 2)
    if not two_d or topo.N != grid * grid:
        return None
    messages, paths = [], []
    for u in range(topo.N):
        x, y = divmod(u, grid)
        for axis, (dx, dy) in ((0, (1, 0)), (0, (-1, 0)), (1, (0, 1)), (1, (0, -1))):
            if 0 <= x + dx < grid and 0 <= y + dy < grid:
                v = (x + dx) * grid + y + dy
                if v not in topo.nbrs[u]:
                    raise ValueError("логічний сусід не є фізичним сусідом")
                messages.append((u, v, 0, side_flits(axis, key, grid)))
                paths.append([u, v])
    return messages, paths


def simulate_halo(topo, cfg, key=None):
    plan = halo_plan(topo, key)
    if plan is None:
        return None
    messages, paths = plan
    lat = simulate_paths(topo, cfg, messages, paths)
    return dict(cycles=max(lat), messages=len(messages), payload_flits=sum(m[3] for m in messages))


def simulate_collective(topo, cfg, roots, faults=NO_FAULTS, flits=None, balance=False, sched="fifo"):
    """ONE-TO-ALL (один корінь) і ALL-TO-ALL (кілька коренів): кожен корінь розсилає M флітів власним деревом.
    sched="fifo" — за часом готовності; "critical" — першим іде повідомлення з найвищим піддеревом.
    Нижня межа за портами визначена лише для повного завершення (complete=True)."""
    roots = list(roots)
    M = flits if flits is not None else cfg.flits
    if M <= 0 or len(set(roots)) != len(roots):
        raise ValueError("M має бути додатним, корені не повинні повторюватися")
    if sched not in ("fifo", "critical"):
        raise ValueError("невідомий планувальник")
    for r in roots:
        if not faults.alive_node(r):
            raise ValueError("корінь несправний")
    if not roots:
        return dict(cycles=0, ms=0.0, max_depth=0, mean_depth=0.0, max_link_load=0, lb_port_cycles=0, lb_link_cycles=0,
                    unreachable=0, repair_rounds=0, frame_pct=0.0, complete=True)
    trees = {r: build_broadcast_tree(topo, r, faults, balance) for r in roots}
    load = {}
    if sched == "critical":
        t_done = _collective_critical(trees, cfg, M, load)
    else:
        free = {}
        heap = [(cfg.t_init, r, r) for r in roots]
        heapq.heapify(heap)
        t_done = cfg.t_init + M
        while heap:
            t, r, v = heapq.heappop(heap)
            for c in trees[r].children.get(v, ()):
                link = (v, c)
                start = max(t, free.get(link, 0))
                free[link] = start + M
                load[link] = load.get(link, 0) + 1
                head = start + cfg.t_router + cfg.t_link
                t_done = max(t_done, head + M)
                heapq.heappush(heap, (head, r, c))
    unreachable = sum(len(t.unreachable) for t in trees.values())
    complete = unreachable == 0
    lb = None
    if complete:
        lb = 0
        rs = set(roots)
        for u in range(topo.N):
            if not faults.alive_node(u):
                continue
            deg = sum(1 for v in topo.nbrs[u] if faults.alive_link(u, v))
            need = (len(roots) - (1 if u in rs else 0)) * M
            if need and deg:
                lb = max(lb, math.ceil(need / deg))
    mx = max(load.values(), default=0)
    return dict(cycles=t_done, ms=cfg.ms(t_done), max_depth=max(t.height for t in trees.values()),
                mean_depth=statistics.mean(t.height for t in trees.values()), max_link_load=mx,
                lb_port_cycles=lb, lb_link_cycles=cfg.t_init + mx * M, unreachable=unreachable,
                repair_rounds=max(t.repair_rounds for t in trees.values()),
                frame_pct=100.0 * cfg.ms(t_done) / cfg.frame_budget_ms, complete=complete)

def combined_cdg(topo, faults, hc, balance="flow"):
    """Об'єднаний CDG усіх маршрутів ONE-TO-ONE-FT та всіх дерев розсилки (корінь — кожен живий вузол) зі СПІЛЬНИМ пулом
    VC: у маршрутів VC = номер переходу (фаза 1) або hc (escape); у розсилки VC = глибина дитини - 1.
    Потрібна кількість VC = max(hc+1, висота розсилки). Повертає залишок алгоритму Кана."""
    forest = build_escape_forest(topo, faults)
    deps = {}
    pairs = 0
    for tr in forest.trees:
        nodes = list(tr.depth)
        for s in nodes:
            for d in nodes:
                if s == d:
                    continue
                r = route_one_to_one_ft(topo, s, d, faults, forest, hc)
                if not r.delivered:
                    raise AssertionError("FT не доставив пару всередині компоненти")
                for i, (u, v) in enumerate(zip(r.path, r.path[1:])):
                    deps.setdefault((u, v, r.vcs[i]), set())
                route_cdg_add(deps, r.path, r.vcs)
                pairs += 1
    hmax = 0
    for root in forest.comp:
        t = build_broadcast_tree(topo, root, faults, balance)
        hmax = max(hmax, t.height)
        for v, u in t.parent.items():
            a = (u, v, t.depth[v] - 1)
            deps.setdefault(a, set())
            for w in t.children.get(v, ()):
                b = (v, w, t.depth[w] - 1)
                deps[a].add(b)
                deps.setdefault(b, set())
    return dict(residual=cdg_cycle_nodes(deps), vertices=len(deps), pairs=pairs, hmax=hmax, vcs_required=max(hc + 1, hmax))


def torus_metrics(dims):
    """Аналітичні D, середня відстань, степінь і кількість ребер k-вимірного тора (усі виміри >= 3)."""
    N = 1
    for d in dims:
        N *= d
    ring_mean = [sum(min(k, d - k) for k in range(d)) / d for d in dims]
    return dict(N=N, D=sum(d // 2 for d in dims), avg=sum(ring_mean) * N / (N - 1), degree=2 * len(dims), edges=N * len(dims))


def best_torus3(N):
    best = None
    for a in range(3, N + 1):
        if N % a:
            continue
        for b in range(3, a + 1):
            if (N // a) % b:
                continue
            c = N // a // b
            if c < 3 or c > b:
                continue
            m = torus_metrics((a, b, c))
            key = (m["D"], m["avg"])
            if best is None or key < best[0]:
                best = (key, (a, b, c))
    return None if best is None else best[1]


def hypercube_metrics(k):
    N = 1 << k
    return dict(N=N, D=k, avg=k * N / 2 / (N - 1), degree=k, edges=N * k // 2)


def embed_hypercube(topo, method="gray"):
    """Відображення ШХТ(n) на гіперкуб Q_{2k}, k=ceil(log2 n): вузол (x,y) -> G(x)*2^k + G(y) (код Грея) або x*n+y.
    Ребро гостя реалізується e-cube шляхом (виправлення бітів від молодшого)."""
    n = topo.n
    k = max(1, (n - 1).bit_length())
    H = 2 * k
    if method == "gray":
        g = lambda i: i ^ (i >> 1)
        f = lambda u: (g(u // n) << k) | g(u % n)
    elif method == "row":
        f = lambda u: u
    else:
        raise ValueError("method має бути gray або row")
    loads, lengths, grid_lengths = {}, [], []
    for u, v in topo.edges():
        a, b = f(u), f(v)
        diff = a ^ b
        ln = bin(diff).count("1")
        lengths.append(ln)
        if v in topo.nbrs[u][:4] or u in topo.nbrs[v][:4]:
            grid_lengths.append(ln)
        for bit in range(H):
            if (diff >> bit) & 1:
                nxt = a ^ (1 << bit)
                e = (min(a, nxt), max(a, nxt))
                loads[e] = loads.get(e, 0) + 1
                a = nxt
    return dict(host_dim=H, expansion=(1 << H) / topo.N, dilation_max=max(lengths), dilation_mean=statistics.mean(lengths),
                grid_dilation_max=max(grid_lengths), congestion_max=max(loads.values()))


def yolo_frame_budget(topo, key, n_halos=(50, 100, 200), modes=(("потік + крит. шлях", "flow", "critical"), ("пріоритет портів", False, "fifo"))):
    """Обмін за кадр = n_halo*T_halo + T_allgather + T_bcast; n_halo — параметр чутливості. Потрібна продуктивність
    вузла = FLOPs фрагмента на кадр / (T_frame - T_обмін) (оцінка за площею кадру)."""
    t_frame = 1000.0 / 30
    M, _ = model_flits(key)
    cfg = SimConfig(flits=M)
    halo = simulate_halo(topo, cfg, key)
    if halo is None:
        raise ValueError("для цієї топології обмін гранями не визначено")
    t_bc = simulate_collective(topo, cfg, [0])["cycles"]
    fl = node_flops_per_frame(key, topo.N)
    rows = []
    for tag, bal, sc in modes:
        ag = simulate_collective(topo, cfg, list(range(topo.N)), balance=bal, sched=sc)["cycles"]
        for nh in n_halos:
            tc = (nh * halo["cycles"] + ag + t_bc) / 1e6
            left = t_frame - tc
            rows.append(dict(model=f"YOLO26{key}", channels=p3_channels(key), M=M, mode=tag, n_halo=nh, t_bcast=t_bc,
                             t_halo=halo["cycles"], t_allgather=ag, comm_ms=tc, comm_pct=100 * tc / t_frame, left_ms=left,
                             flops_node=fl, req_gflops=(fl / (left * 1e-3) / 1e9 if left > 0 else None),
                             req_gflops_no_comm=fl * 30 / 1e9))
    return rows


def vc_buffer_cost(cfg, vcs, ports=6, flit_bytes=FLIT_BYTES, buf_flits=None):
    """Пам'ять буферів вузла: vcs каналів на порт. VCT — буфер на повне повідомлення; wormhole — малий буфер."""
    per_vc = (buf_flits if buf_flits is not None else cfg.flits) * flit_bytes
    return dict(vcs=vcs, per_vc_B=per_vc, node_B=per_vc * vcs * ports)


def _diam_fast(topo):
    if not hasattr(topo, "_diam_cache"):
        if isinstance(topo, ChessChordalTorus):
            topo._diam_cache = max(max(topo.bfs(0)), max(topo.bfs(1)))
        else:
            topo._diam_cache = topo.metrics()[0]
    return topo._diam_cache


def directory_bytes(topo, forest):
    """Пам'ять escape-фази на вузол: каталог міток (N записів по ceil(log2 N) біт) + власний інтервал, батько, діти."""
    bits = max(1, math.ceil(math.log2(topo.N)))
    return dict(directory_B=math.ceil(topo.N * bits / 8), local_B=math.ceil((3 + topo.degree()) * bits / 8))


def exp_halo_and_frame():
    cfg = SimConfig()
    print(f"\n=== E7. Обмін гранями P3 (YOLO26{ACTIVE_MODEL}): грані {side_flits(0)} і {side_flits(1)} флітів ===")
    for T in standard_topologies()[:3]:
        r = simulate_halo(T, cfg)
        print(f"{T.name:<20} повідомлень {r['messages']}, флітів {r['payload_flits']}, час {r['cycles']} тактів ({cfg.ms(r['cycles']) * 1000:.2f} мкс)")


def exp_deadlock(quick=False, n=16):
    a, b = chords_for(n)
    T = ChessChordalTorus(n, a, b)
    hc = _diam_fast(T) + 2
    rng = random.Random(909)
    cases = [(0, 0), (5, 0), (20, 0), (0, 60), (30, 100)] if not quick else [(0, 0), (5, 10)]
    print(f"\n=== E9. Спільний CDG маршрутів FT і розсилок, {T.name}, hc={hc} ===")
    ok = True
    for fn, fl in cases:
        f = sample_exact_faults(T, rng, fn, fl)
        r = combined_cdg(T, f, hc)
        ok &= r["residual"] == 0
        print(f"відмов {fn:>2} вузлів + {fl:>3} каналів: пар {r['pairs']:>6}, вершин {r['vertices']:>6}, залишок Кана {r['residual']}, "
              f"висота розсилки {r['hmax']}, потрібно VC {r['vcs_required']}")
    print("РЕЗУЛЬТАТ:", "залишок 0 для всіх сценаріїв" if ok else "ЗНАЙДЕНО ЗАЛИШОК")
    return ok


def exp_embed():
    print("\n=== E14. Відображення ШХТ на гіперкуб ===")
    for n in (4, 8, 16):
        T = ChessChordalTorus(n, *chords_for(n))
        for m in ("row", "gray"):
            r = embed_hypercube(T, m)
            print(f"N={T.N:<4} {m:<5} Q{r['host_dim']}: розтягнення max {r['dilation_max']}, сер. {r['dilation_mean']:.3f}, "
                  f"зв'язки тора max {r['grid_dilation_max']}, перевантаження {r['congestion_max']}, розширення {r['expansion']:g}")


def exp_models():
    print("\n=== Параметри моделей YOLO26 (model.info() користувача) ===")
    for k, m in YOLO26_MODELS.items():
        M, Mh = model_flits(k)
        print(f"YOLO26{k}: C(P3)={p3_channels(k)}, M={M}, грані {side_flits(0, k)}/{side_flits(1, k)}, GFLOPs@640={m.gflops_640}, "
              f"GFLOPs/вузол/кадр={node_flops_per_frame(k) / 1e9:.3f}")


def exp_yolo():
    T0 = ChessChordalTorus(16, 2, 5)
    print("\n=== E11. Кадровий бюджет YOLO26, ШХТ(16;2,5), N=256 (оцінка чутливості) ===")
    for k in YOLO26_MODELS:
        for r in yolo_frame_budget(T0, k):
            print(f"{r['model']:<9} {r['mode']:<20} n_halo={r['n_halo']:<4} обмін {r['comm_ms']:.3f} мс ({r['comm_pct']:.1f} %), потрібно {r['req_gflops']:.1f} GFLOPS/вузол")


def exp_directory(n=16):
    cfg = SimConfig()
    T = ChessChordalTorus(n, *chords_for(n))
    m = directory_bytes(T, build_escape_forest(T))
    print(f"\n=== E13. Пам'ять escape-фази і буферів, {T.name} ===\nкаталог міток {m['directory_B']} Б, локальний стан {m['local_B']} Б на вузол")
    for vcs in (8, 9):
        for label, bf in (("VCT", None), ("wormhole 16 флітів", 16)):
            print(f"{vcs} VC, {label}: {vc_buffer_cost(cfg, vcs, T.degree(), buf_flits=bf)['node_B'] / 1048576:.3f} МіБ на вузол")


def exp_allgather():
    cfg = SimConfig()
    print(f"\n=== E3. ALL-TO-ALL (all-gather), YOLO26{ACTIVE_MODEL}, M = {cfg.flits} флітів ===")
    for T in standard_topologies():
        ideal = ideal_allgather_estimate(T, cfg)
        for r in exp_allgather_modes(T, cfg):
            print(f"{T.name:<18}{r['mode']:<20}{r['cycles'] / 1000:>9.1f} мкс  Lmax={r['max_link_load']:<4} межа {r['lb_port_cycles'] / 1000:>8.1f} мкс  "
                  f"відношення {r['cycles'] / r['lb_port_cycles']:.3f}" + ("" if ideal is None else f"  еталон {ideal / 1000:.1f} мкс"))


def main(argv):
    quick = "--quick" in argv
    if "--model" in argv:
        set_model(argv[argv.index("--model") + 1])
    cmds = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] != "--model")] or ["all"]
    cmd = cmds[0]
    ok = True
    if cmd in ("models", "all"):
        exp_models()
    if cmd in ("verify", "all"):
        ok &= exp_verify(ns=(4, 8, 12, 16) if quick else (4, 6, 8, 10, 12, 16, 20, 24, 30))
    if cmd in ("broadcast", "all"):
        ok &= exp_broadcast()
    if cmd in ("allgather", "all"):
        exp_allgather()
    if cmd in ("cdg", "all"):
        exp_cdg((4, 8) if quick else (4, 8, 16))
    if cmd in ("faults", "all"):
        exp_faults(quick)
    if cmd in ("traffic", "all"):
        exp_traffic(quick)
    if cmd in ("halo", "all"):
        exp_halo_and_frame()
    if cmd in ("ft", "all"):
        exp_ft(quick)
    if cmd in ("deadlock", "all"):
        ok &= exp_deadlock(quick)
    if cmd in ("embed", "all"):
        exp_embed()
    if cmd in ("yolo", "all"):
        exp_yolo()
    if cmd in ("directory", "all"):
        exp_directory()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
