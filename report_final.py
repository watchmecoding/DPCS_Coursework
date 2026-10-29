#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
report_final.py — табличний звіт за експериментами simulator_final.py (версія 5.0).
    python report_final.py [--quick] [--model n|s|m|l|x] [--out DIR] [e3 e8 ...] [--export-only]
Експерименти: m0 e0 e1 e2 e3 e4 e5 e6 e7 e8 e9 e11 e13 e14. Динамічні відмови свідомо відсутні.
Кожен результат одразу зберігається в CSV та checkpoint.json; Excel потребує openpyxl, але не обов'язковий.
"""
from __future__ import annotations
import argparse, hashlib, json, platform, random, statistics, sys, time
from pathlib import Path
try:
    import pandas as pd
except ImportError:
    raise SystemExit("Потрібен pandas: python -m pip install pandas")
try:
    import simulator_final as S
except ImportError as exc:
    raise SystemExit("Поруч має лежати simulator_final.py (версія 5.0): " + str(exc))
if not hasattr(S, "combined_cdg") or not hasattr(S, "embed_hypercube"):
    raise SystemExit("Несумісний simulator_final.py: потрібна версія 5.0.")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
NOTES = {
 "m0": "Параметри з model.info() користувача: GFLOPs округлені, на 640x640. P3 — вихід шару 4 при вході 640x640.",
 "e0": "D — діаметр; d — середня відстань. Q=2d/S — умовний топологічний індекс, а не виміряна пропускна здатність. C=N·D·S — умовна вартість у портах·переходах.",
 "e1": "Відстань верифікується BFS для двох класів вузлів (решта — зсуви з парною сумою). Маршрути перевіряються на всіх парах для n<=16, далі вибірково.",
 "e2": "Розсилка за покривним деревом від кожного кореня; час порівнюється з t_init+h·(t_router+t_link)+M.",
 "e3": "Усі вузли надсилають однакове повідомлення всім. Нижня межа (N-1)M/S не може бути перевищена. Усі топології оцінюються тим самим класом деревних алгоритмів; оцінка еталонного алгоритму — аналітична, лише для гіперкуба й тора.",
 "e4": "CDG безвідмовних найкоротших маршрутів. Залишок Кана 0 означає ациклічність; ненульовий залишок не дорівнює кількості вершин у циклах.",
 "e5": "Базове правило без escape-фази при статичних відмовах: локальне правило без гарантії доставляє не всі досяжні пари.",
 "e6": "Кожен вузол одночасно надсилає одному адресату (перестановка без нерухомих точок). Показано середнє середніх, середнє максимумів і абсолютний максимум.",
 "e7": "Синтетичний обмін гранями P3: кожен сусід отримує лише відповідну грань; країв зображення не замкнуто. Не відтворює реальні шари, кути й padding YOLO.",
 "e8": "Статичний узгоджений знімок відмов. FT має доставляти всі пари в межах компоненти й відхиляти недосяжні. Емпіричні 100 % не замінюють структурного обґрунтування.",
 "e9": "Спільний CDG усіх маршрутів FT і дерев розсилки в одному пулі VC. Потрібна кількість VC = max(hc+1, висота розсилки).",
 "e11": "Чутливість мережевого бюджету, а не виконання YOLO. n_halo — параметр сценарію. FLOPs оцінено за площею кадру 1920x1088; не враховано дублювання на межах і ефективність апаратури.",
 "e13": "Кошторис буферів для заданої кількості VC на порт: VCT — один пакет на VC; wormhole — 16 флітів. Не вимір пам'яті реалізації.",
 "e14": "Відображення вузла (x,y) на Q_{2k}: код Грея або рядкове; ребра гостя реалізуються e-cube шляхами. Розширення 1 для n=2^k.",
}
ORDER = ["m0", "e0", "e1", "e2", "e3", "e4", "e5", "e6", "e7", "e8", "e9", "e11", "e13", "e14"]
FAULT_SETS = [("5 вузлів", 5, 0), ("20 вузлів", 20, 0), ("50 вузлів", 50, 0), ("30 каналів", 0, 30),
              ("100 каналів", 0, 100), ("150 каналів", 0, 150), ("30 вузлів + 100 каналів", 30, 100)]


class Runner:
    def __init__(self, args):
        self.a = args; S.set_model(args.model); self.cfg = S.SimConfig(); self.T = S.ChessChordalTorus(16, 2, 5)
        self.dir = Path(args.out); self.dir.mkdir(parents=True, exist_ok=True); self.tables, self.checks = {}, {}
        sim = Path(S.__file__)
        self.meta = dict(version="5.0", simulator=str(sim), simulator_sha256=hashlib.sha256(sim.read_bytes()).hexdigest(),
                         python=platform.python_version(), model=args.model, quick=args.quick,
                         assumptions="1 ГГц, 1 фліт/такт, all-port, необмежені черги, статичний узгоджений знімок відмов")

    def us(self, cycles):
        return self.cfg.ms(cycles) * 1000.0

    def show(self, key, title, rows, check=None):
        df = pd.DataFrame(rows)
        self.tables[key] = dict(title=title, notes=NOTES[key], records=json.loads(df.to_json(orient="records", force_ascii=False)))
        if check is not None:
            self.checks[key] = bool(check)
        print("\n" + "=" * 100 + f"\n{key.upper()}. {title}\n{NOTES[key]}\n", flush=True)
        print(df.to_string(index=False, na_rep="—", float_format=lambda x: f"{x:.3f}"))
        if check is not None:
            print("ПЕРЕВІРКА:", "OK" if check else "ПОМИЛКА")
        df.to_csv(self.dir / f"{key}.csv", index=False, encoding="utf-8-sig"); self.checkpoint()

    def checkpoint(self):
        p = self.dir / "checkpoint.json"; tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(dict(meta=self.meta, tables=self.tables, checks=self.checks), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)

    def m0(self):
        rows = []
        for k, m in S.YOLO26_MODELS.items():
            M, Mh = S.model_flits(k)
            rows.append({"Модель": "YOLO26" + k, "width": m.width, "max_channels": m.max_channels, "Каналів P3": S.p3_channels(k),
                         "M, фліти": M, "Грань по x": S.side_flits(0, k), "Грань по y": S.side_flits(1, k),
                         "GFLOPs@640": m.gflops_640, "Параметрів, М": m.params_m,
                         "GFLOPs/вузол/кадр (оцінка)": S.node_flops_per_frame(k) / 1e9,
                         "Мін. GFLOPS/вузол без обміну": S.node_flops_per_frame(k) * 30 / 1e9})
        self.show("m0", "Моделі YOLO26 і розміри повідомлень", rows)

    def e0(self):
        rows = []
        for n in ((4, 8, 16, 24, 30) if self.a.quick else tuple(range(4, 31, 2))):
            T = S.ChessChordalTorus(n, *S.chords_for(n)); d0, d1 = T.bfs(0), T.bfs(1)
            D = max(max(d0), max(d1)); d = (sum(d0) + sum(d1)) / (2 * (T.N - 1))
            rows.append({"N": T.N, "Топологія": T.name, "S": 6, "D": D, "d": d, "Q": 2 * d / 6, "C": T.N * D * 6})
            m2 = S.torus_metrics((n, n))
            rows.append({"N": T.N, "Топологія": f"2D-тор {n}x{n}", "S": 4, "D": m2["D"], "d": m2["avg"], "Q": 2 * m2["avg"] / 4, "C": T.N * m2["D"] * 4})
            dims = S.best_torus3(T.N)
            if dims:
                m3 = S.torus_metrics(dims)
                rows.append({"N": T.N, "Топологія": "3D-тор " + "x".join(map(str, dims)), "S": 6, "D": m3["D"], "d": m3["avg"], "Q": 2 * m3["avg"] / 6, "C": T.N * m3["D"] * 6})
            if T.N & (T.N - 1) == 0:
                mh = S.hypercube_metrics(T.N.bit_length() - 1)
                rows.append({"N": T.N, "Топологія": f"гіперкуб Q{mh['degree']}", "S": mh["degree"], "D": mh["D"], "d": mh["avg"],
                             "Q": 2 * mh["avg"] / mh["degree"], "C": T.N * mh["D"] * mh["degree"]})
        self.show("e0", "Топологічні характеристики при N<1000", rows)

    def e1(self):
        rows, ok, rng = [], True, random.Random(2026)
        for n in ((4, 8, 16) if self.a.quick else (4, 6, 8, 10, 12, 16, 20, 24, 28, 30)):
            T = S.ChessChordalTorus(n, *S.chords_for(n)); ref = {0: T.bfs(0), 1: T.bfs(1)}
            bad_d = sum(T.dist(u, v) != ref[u][v] for u in (0, 1) for v in range(T.N))
            full = n <= 16
            pairs = [(u, v) for u in range(T.N) for v in range(T.N)] if full else [(rng.randrange(T.N), rng.randrange(T.N)) for _ in range(20000)]
            cache, bad_r = {}, 0
            for u, v in pairs:
                if u not in cache:
                    cache[u] = T.bfs(u)
                r = S.route_one_to_one(T, u, v)
                bad_r += (not r.delivered) or r.hops != cache[u][v]
            ok &= bad_d == 0 and bad_r == 0
            rows.append({"n": n, "N": T.N, "Хорди": str(S.chords_for(n)), "Відстаней (2 класи)": 2 * T.N, "Помилок відстані": bad_d,
                         "Маршрутів": len(pairs), "Вибірка": "ні" if full else "так", "Не найкоротших": bad_r})
        self.show("e1", "Верифікація ONE-TO-ONE без відмов", rows, ok)

    def e2(self):
        ok, rows = True, []
        for root in range(self.T.N):
            tr = S.build_broadcast_tree(self.T, root)
            ok &= S.check_tree(self.T, tr) and tr.covered == self.T.N and tr.height == 5 and tr.repair_rounds == 0
        for root in (0, 1):
            r = S.simulate_collective(self.T, self.cfg, [root])
            pred = self.cfg.t_init + r["max_depth"] * (self.cfg.t_router + self.cfg.t_link) + self.cfg.flits
            ok &= pred == r["cycles"]
            rows.append({"Корінь": root, "Висота": r["max_depth"], "Час, такти": r["cycles"], "Формула, такти": pred, "Час, мкс": self.us(r["cycles"])})
        self.show("e2", f"ONE-TO-ALL (M={self.cfg.flits})", rows, ok)

    def e3(self):
        rows, ok = [], True
        for T in S.standard_topologies():
            ideal = S.ideal_allgather_estimate(T, self.cfg)
            for r in S.exp_allgather_modes(T, self.cfg):
                ok &= r["complete"] and r["cycles"] >= r["lb_port_cycles"]
                rows.append({"Топологія": T.name, "S": T.degree(), "Режим": r["mode"], "Час, мкс": self.us(r["cycles"]), "% кадру": r["frame_pct"],
                             "Макс. навантаження каналу": r["max_link_load"], "Нижня межа, мкс": self.us(r["lb_port_cycles"]),
                             "Час/межа": r["cycles"] / r["lb_port_cycles"], "Оцінка еталонного алгоритму, мкс": None if ideal is None else self.us(ideal)})
        self.show("e3", f"ALL-TO-ALL / all-gather, M={self.cfg.flits}", rows, ok)

    def e4(self):
        rows, ok = [], True
        for n in ((4, 8) if self.a.quick else (4, 8, 16)):
            T = S.ChessChordalTorus(n, *S.chords_for(n)); D = S._diam_fast(T)
            for mode, label in (("none", "один VC"), ("hop", "VC = номер переходу")):
                deps = S.channel_dependency_graph(T, mode); res = S.cdg_cycle_nodes(deps)
                ok &= mode == "none" or res == 0
                rows.append({"Мережа": T.name, "Режим": label, "VC": 1 if mode == "none" else D, "Вершин CDG": len(deps), "Залишок Кана": res})
        self.show("e4", "CDG без відмов", rows, ok)

    def e5(self):
        rows = []
        trials, pairs = (40, 10) if self.a.quick else (300, 20)
        for i, (name, fn, fl) in enumerate(FAULT_SETS):
            r = S.evaluate_faults(self.T, lambda g, fn=fn, fl=fl: S.sample_exact_faults(self.T, g, fn, fl), trials, pairs, 100 + i)
            rows.append({"Відмови": name, "Досяжних пар": r["attempts"], "Доставлено, %": r["success"], "95% інтервал": f"[{r['lo']:.2f}; {r['hi']:.2f}]",
                         "Подовження": r["stretch"], "Розсилка: повне покриття, %": r["bc_full"], "Глибина/діаметр": r["depth_ratio"]})
        self.show("e5", "Базове правило ONE-TO-ONE при статичних відмовах", rows)

    def e6(self):
        rows = []
        for T in S.standard_topologies():
            rng = random.Random(11); means, maxima, abs_max, hops = [], [], 0, []
            for _ in range(5 if self.a.quick else 30):
                p = list(range(T.N))
                while True:
                    rng.shuffle(p)
                    if all(i != d for i, d in enumerate(p)):
                        break
                lat = S.simulate_traffic(T, self.cfg, [(i, p[i], 0) for i in range(T.N)])
                means.append(statistics.mean(lat)); maxima.append(max(lat)); abs_max = max(abs_max, max(lat))
                hops.append(statistics.mean(T.dist(i, p[i]) for i in range(T.N)))
            rows.append({"Топологія": T.name, "Серед. шлях": statistics.mean(hops), "Серед. затримка, мкс": self.us(statistics.mean(means)),
                         "Сер. максимумів, мкс": self.us(statistics.mean(maxima)), "Абсолютний максимум, мкс": self.us(abs_max)})
        lat = S.simulate_traffic(self.T, self.cfg, [(0, 137, 0)])[0]
        pred = self.cfg.t_init + self.T.dist(0, 137) * (self.cfg.t_router + self.cfg.t_link) + self.cfg.flits
        self.show("e6", f"Перестановка, M={self.cfg.flits}", rows, lat == pred)

    def e7(self):
        rows = []
        for T in S.standard_topologies()[:3]:
            r = S.simulate_halo(T, self.cfg)
            rows.append({"Топологія": T.name, "Повідомлень": r["messages"], "Флітів": r["payload_flits"], "Грані": f"{S.side_flits(0)}/{S.side_flits(1)}",
                         "Час, такти": r["cycles"], "Час, мкс": self.us(r["cycles"])})
        self.show("e7", "Обмін гранями P3", rows)

    def e8(self):
        rng, rows, ok = random.Random(2026), [], True
        D = S._diam_fast(self.T); trials, pairs = (10, 15) if self.a.quick else (200, 20)
        for hc in (D, D + 2, D + 4):
            for name, fn, fl in FAULT_SETS:
                reach = succ = old = unreach = detected = esc = 0; stretch = []
                for _ in range(trials):
                    f = S.sample_exact_faults(self.T, rng, fn, fl); fo = S.build_escape_forest(self.T, f); alive = list(fo.comp); cache = {}
                    for _ in range(pairs):
                        u, v = rng.sample(alive, 2)
                        if u not in cache:
                            cache[u] = self.T.bfs(u, f)
                        ref = cache[u][v]; r = S.route_one_to_one_ft(self.T, u, v, f, fo, hc)
                        if ref < 0:
                            unreach += 1; detected += (not r.delivered); continue
                        reach += 1; succ += r.delivered; esc += r.escaped; old += S.route_one_to_one(self.T, u, v, f).delivered
                        if r.delivered:
                            stretch.append(r.hops / ref)
                ok &= succ == reach and detected == unreach
                rows.append({"hc": hc, "Відмови": name, "Досяжних": reach, "FT доставлено": succ, "Базове правило, %": 100 * old / max(1, reach),
                             "Недосяжних": unreach, "Виявлено": detected, "Через escape, %": 100 * esc / max(1, reach),
                             "Серед. подовження": statistics.mean(stretch) if stretch else None})
        self.show("e8", "ONE-TO-ONE-FT", rows, ok)

    def e9(self):
        rng, rows, ok = random.Random(909), [], True
        T = self.T if not self.a.quick else S.ChessChordalTorus(8, *S.chords_for(8)); hc = S._diam_fast(T) + 2
        for fn, fl in ([(0, 0), (5, 10)] if self.a.quick else [(0, 0), (5, 0), (20, 0), (0, 60), (30, 100)]):
            f = S.sample_exact_faults(T, rng, fn, fl); r = S.combined_cdg(T, f, hc); ok &= r["residual"] == 0
            rows.append({"N": T.N, "hc": hc, "Відмов вузлів": fn, "Відмов каналів": fl, "Пар": r["pairs"], "Вершин CDG": r["vertices"],
                         "Залишок Кана": r["residual"], "Висота розсилки": r["hmax"], "Потрібно VC": r["vcs_required"]})
        self.show("e9", "Спільний CDG маршрутів FT і розсилок", rows, ok)

    def e11(self):
        rows = []
        for k in S.YOLO26_MODELS:
            for r in S.yolo_frame_budget(self.T, k):
                rows.append({"Модель": r["model"], "M": r["M"], "Режим all-gather": r["mode"], "n_halo": r["n_halo"], "All-gather, мс": r["t_allgather"] / 1e6,
                             "Halo, мкс": r["t_halo"] / 1e3, "Обмін, мс": r["comm_ms"], "% бюджету": r["comm_pct"], "GFLOPs/вузол/кадр": r["flops_node"] / 1e9,
                             "Потрібно GFLOPS (оцінка)": r["req_gflops"], "Без обміну GFLOPS": r["req_gflops_no_comm"]})
        self.show("e11", "Чутливість кадрового бюджету, 30 кадрів/с", rows)

    def e13(self):
        rows = []
        for vcs in (8, 9):
            for label, bf in (("VCT: пакет у кожному VC", None), ("wormhole: 16 флітів", 16)):
                for k in S.YOLO26_MODELS:
                    c = S.vc_buffer_cost(S.SimConfig(flits=S.model_flits(k)[0]), vcs, 6, buf_flits=bf)
                    rows.append({"Модель": "YOLO26" + k, "VC": vcs, "Режим": label, "Буфери вузла, КіБ": c["node_B"] / 1024, "Буфери вузла, МіБ": c["node_B"] / 1048576})
        self.show("e13", "Оцінка буферів", rows)

    def e14(self):
        rows = []
        for n in (4, 8, 16):
            T = S.ChessChordalTorus(n, *S.chords_for(n))
            for m, label in (("row", "рядкове"), ("gray", "код Грея")):
                r = S.embed_hypercube(T, m)
                rows.append({"N": T.N, "Хост": f"Q{r['host_dim']}", "Метод": label, "Розтягнення max": r["dilation_max"], "Розтягнення сер.": r["dilation_mean"],
                             "Зв'язки тора max": r["grid_dilation_max"], "Перевантаження": r["congestion_max"], "Розширення": r["expansion"]})
        self.show("e14", "Відображення ШХТ на гіперкуб", rows)

    def export(self):
        checks = pd.DataFrame([{"Експеримент": k, "OK": v} for k, v in self.checks.items()]); checks.to_csv(self.dir / "checks.csv", index=False, encoding="utf-8-sig")
        with (self.dir / "report.md").open("w", encoding="utf-8") as fh:
            for k, t in self.tables.items():
                df = pd.DataFrame(t["records"]); fh.write(f"## {k.upper()}. {t['title']}\n\n{t['notes']}\n\n")
                try:
                    fh.write(df.to_markdown(index=False, floatfmt=".4f") + "\n\n")
                except ImportError:
                    fh.write("```text\n" + df.to_string(index=False) + "\n```\n\n")
        try:
            import openpyxl  # noqa: F401
            with pd.ExcelWriter(self.dir / "report.xlsx", engine="openpyxl") as xw:
                for k, t in self.tables.items():
                    pd.DataFrame(t["records"]).to_excel(xw, sheet_name=k.upper(), index=False); ws = xw.sheets[k.upper()]; ws.freeze_panes = "A2"; ws.auto_filter.ref = ws.dimensions
                    for col in ws.columns:
                        ws.column_dimensions[col[0].column_letter].width = min(48, max(12, max(len(str(c.value or "")) for c in col) + 2))
                checks.to_excel(xw, sheet_name="CHECKS", index=False)
                pd.DataFrame([{"Експеримент": k, "Пояснення": t["notes"]} for k, t in self.tables.items()]).to_excel(xw, sheet_name="NOTES", index=False)
        except (ImportError, OSError) as exc:
            print(f"Excel не збережено ({exc}). CSV, JSON, Markdown збережено. Після python -m pip install openpyxl: python report_final.py --export-only --out \"{self.dir}\"")


def main():
    p = argparse.ArgumentParser(description="Звіт ШХТ v5.0")
    p.add_argument("experiments", nargs="*", help=" ".join(ORDER)); p.add_argument("--model", choices=list(S.YOLO26_MODELS), default="x")
    p.add_argument("--quick", action="store_true"); p.add_argument("--out", default="report_out_final"); p.add_argument("--export-only", action="store_true")
    a = p.parse_args(); r = Runner(a)
    if a.export_only:
        data = json.loads((r.dir / "checkpoint.json").read_text(encoding="utf-8")); r.tables, r.checks, r.meta = data["tables"], data["checks"], data["meta"]; r.export(); return 0
    sel = a.experiments or ORDER
    bad = set(sel) - set(ORDER)
    if bad:
        p.error("Невідомі експерименти: " + ", ".join(sorted(bad)))
    print("Симулятор:", r.meta["simulator"], "\nМодель: YOLO26" + a.model, flush=True)
    for key in sel:
        t0 = time.perf_counter()
        try:
            getattr(r, key)()
        except Exception:
            r.checkpoint(); r.export(); raise
        print(f"[{key}: {time.perf_counter() - t0:.1f} с]", flush=True)
    r.export(); print("\nПЕРЕВІРКИ:", r.checks, "\nЗбережено:", r.dir)
    return 0 if all(r.checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
