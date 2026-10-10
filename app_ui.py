"""MindProof - polished UI layer (v2).

Put next to app.py and run:  python app_pretty.py
Reuses all logic from app.py (clean_text, prediction, history, exports,
stance override) and only overrides how things look.
"""
import tkinter as tk
import customtkinter as ctk
import app as base

# ---------- richer palette (app.py reads these dicts) ----------
base.DARK.update({
    "bg": "#07080D", "sidebar": "#0B0D14", "panel": "#10131C",
    "card": "#141824", "card2": "#1A1F2E", "input": "#0D1017",
    "border": "#232A3B", "muted": "#A9B1C6", "dim": "#6F7890",
})
base.LIGHT.update({"bg": "#EEF1F7", "card2": "#EDF0F7", "border": "#D5DCE8"})
base.THEMES["Cyan"] = ("#06B6D4", "#22D3EE")

STANCE = {"FOR": "#10B981", "AGAINST": "#F43F5E", "NEUTRAL": "#64748B"}
FONT = "Segoe UI"


def fmt(v, nd=4):
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "-"


def short(text, n=170):
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n].rstrip() + "…"


class PrettyApp(base.MindProofApp):
    # ---------- small widgets ----------
    def pill(self, parent, text, color):
        return ctk.CTkLabel(parent, text=f"  {text}  ", height=26, corner_radius=13,
                            fg_color=color, text_color="#FFFFFF",
                            font=ctk.CTkFont(family=FONT, size=11, weight="bold"))

    def chip(self, parent, text, fg=None):
        return ctk.CTkLabel(parent, text=f"  {text}  ", height=28, corner_radius=8,
                            fg_color=fg or self.colors["card2"], text_color=self.colors["text"],
                            font=ctk.CTkFont(family=FONT, size=11))

    def ring(self, parent, value, size=160):
        c = tk.Canvas(parent, width=size, height=size, bg=self.colors["card"], highlightthickness=0)
        p, w = 14, 13
        c.create_oval(p, p, size - p, size - p, outline=self.colors["border"], width=w)
        extent = -min(359.9, max(0.1, 3.6 * value))
        c.create_arc(p, p, size - p, size - p, start=90, extent=extent, style="arc",
                     outline=self.accent, width=w)
        c.create_text(size / 2, size / 2 - 8, text=f"{value:.1f}", fill=self.colors["text"], font=(FONT, 27, "bold"))
        c.create_text(size / 2, size / 2 + 22, text="out of 100", fill=self.colors["muted"], font=(FONT, 9))
        return c

    def _section(self, parent, row, col, title, span=1):
        f = ctk.CTkFrame(parent, fg_color=self.colors["card2"], corner_radius=14)
        f.grid(row=row, column=col, columnspan=span, sticky="nsew", padx=5, pady=5)
        self.label(f, title, 14, "bold").pack(anchor="w", padx=16, pady=(14, 8))
        return f

    # ---------- new analysis: restyle hero and pipeline ----------
    def show_new_analysis(self):
        super().show_new_analysis()
        for w in self.page.grid_slaves(row=0):
            w.destroy()
        hero = self.card()
        hero.grid(row=0, column=0, sticky="ew", pady=(8, 12))
        hero.grid_columnconfigure(1, weight=1)
        ctk.CTkFrame(hero, width=6, corner_radius=3, fg_color=self.accent).grid(
            row=0, column=0, sticky="ns", padx=(22, 0), pady=22)
        box = ctk.CTkFrame(hero, fg_color="transparent")
        box.grid(row=0, column=1, sticky="w", padx=18, pady=20)
        self.label(box, "Understand the argument.", 32, "bold").pack(anchor="w")
        self.label(box, "Enter a topic and two arguments. MindProof predicts quality and stance, "
                        "then shows the linguistic signals behind them.", 13,
                   color=self.colors["muted"], wraplength=900, justify="left").pack(anchor="w", pady=(6, 0))

        for w in self.page.grid_slaves(row=4):
            w.destroy()
        info = self.card()
        info.grid(row=4, column=0, sticky="ew", pady=7)
        self.label(info, "How it works", 15, "bold").pack(anchor="w", padx=22, pady=(18, 10))
        flow = ctk.CTkFrame(info, fg_color="transparent")
        flow.pack(anchor="w", padx=22)
        steps = ["Topic + argument", "TF-IDF", "Handcrafted features", "MiniLM embeddings", "Quality + stance"]
        for i, s in enumerate(steps):
            self.chip(flow, s, self.accent if i == len(steps) - 1 else None).pack(side="left")
            if i < len(steps) - 1:
                self.label(flow, "›", 18, color=self.colors["dim"]).pack(side="left", padx=6)
        self.label(info, "The quality score reflects the dataset-trained target. It is not a truth score "
                         "or a verdict on who is correct.", 12, color=self.colors["muted"],
                   wraplength=1000, justify="left").pack(anchor="w", padx=22, pady=(12, 18))

    # ---------- results ----------
    def _result_card(self, parent, col, p):
        c = self.card(parent)
        c.grid(row=0, column=col, sticky="nsew", padx=(0, 8) if col == 0 else (8, 0))
        ctk.CTkFrame(c, height=5, corner_radius=3, fg_color=self.accent).pack(fill="x", padx=22, pady=(18, 0))
        self.label(c, p["name"], 20, "bold").pack(anchor="w", padx=24, pady=(14, 0))
        self.label(c, "Predicted argument quality", 12, color=self.colors["muted"]).pack(anchor="w", padx=24, pady=(0, 6))
        body = ctk.CTkFrame(c, fg_color="transparent")
        body.pack(fill="x", padx=18, pady=(0, 20))
        self.ring(body, float(p["quality"])).pack(side="left")
        side = ctk.CTkFrame(body, fg_color="transparent")
        side.pack(side="left", padx=18, anchor="n", pady=14)
        lab = p.get("stance_label", "NEUTRAL")
        self.label(side, "Stance", 11, color=self.colors["dim"]).pack(anchor="w")
        self.pill(side, lab, STANCE.get(lab, self.accent)).pack(anchor="w", pady=(2, 10))
        self.label(side, "Stance source", 11, color=self.colors["dim"]).pack(anchor="w")
        self.label(side, p.get("stance_source", "V4 classifier"), 12, color=self.colors["muted"]).pack(anchor="w")

    def _detail_card(self, p, row):
        c = self.card()
        c.grid(row=row, column=0, sticky="ew", pady=8)
        head = ctk.CTkFrame(c, fg_color="transparent")
        head.pack(fill="x", padx=22, pady=(18, 8))
        self.label(head, f'{p["name"]}: details', 18, "bold").pack(side="left")
        lab = p.get("stance_label", "NEUTRAL")
        self.pill(head, lab, STANCE.get(lab, self.accent)).pack(side="right")

        keys = [("word_count", "Words"), ("unique_ratio", "Unique ratio"), ("sentence_count", "Sentences"),
                ("average_sentence_length", "Avg length"), ("reasoning_markers", "Reasoning"),
                ("evidence_markers", "Evidence"), ("example_markers", "Examples"), ("contrast_markers", "Contrast")]
        tiles = ctk.CTkFrame(c, fg_color="transparent")
        tiles.pack(fill="x", padx=18, pady=(4, 6))
        for i, (k, name) in enumerate(keys):
            tiles.grid_columnconfigure(i % 4, weight=1, uniform="t")
            v = p["profile"].get(k, "-")
            v = f"{v:.2f}" if isinstance(v, float) else str(v)
            t = ctk.CTkFrame(tiles, fg_color=self.colors["card2"], corner_radius=12)
            t.grid(row=i // 4, column=i % 4, sticky="nsew", padx=4, pady=4)
            self.label(t, v, 20, "bold", self.accent).pack(anchor="w", padx=14, pady=(10, 0))
            self.label(t, name, 11, color=self.colors["muted"]).pack(anchor="w", padx=14, pady=(0, 10))

        grid = ctk.CTkFrame(c, fg_color="transparent")
        grid.pack(fill="x", padx=18, pady=(4, 0))
        grid.grid_columnconfigure((0, 1), weight=1, uniform="d")
        for col, (title, key) in enumerate((("What the text shows", "explanation"),
                                            ("How to strengthen it", "improvements"))):
            body = self._section(grid, 0, col, title)
            self.label(body, self._bullets(p[key]), 12, color=self.colors["muted"],
                       justify="left", wraplength=470).pack(anchor="w", padx=16, pady=(0, 14))
        body = self._section(grid, 1, 0, "Possible fallacy indicators", span=2)
        fs = p.get("fallacies") or []
        if not fs:
            self.label(body, "No rule-based indicator detected.", 12, color=self.colors["muted"]).pack(anchor="w", padx=16, pady=(0, 14))
        else:
            r = ctk.CTkFrame(body, fg_color="transparent")
            r.pack(anchor="w", padx=16, pady=(0, 14))
            for f in fs:
                self.chip(r, f.get("name", "Indicator"), self.colors["card"]).pack(side="left", padx=(0, 6))

        self.label(c, "Original argument (as entered)", 12, "bold", self.colors["dim"]).pack(anchor="w", padx=24, pady=(14, 4))
        self.label(c, p.get("argument", ""), 13, color=self.colors["text"], wraplength=1050, justify="left").pack(anchor="w", padx=24)
        self.label(c, "Cleaned text (sent to the V4 predictor)", 12, "bold", self.colors["dim"]).pack(anchor="w", padx=24, pady=(14, 4))
        cleaned = p.get("cleaned_argument", base.clean_text(p.get("argument", "")))
        self.label(c, cleaned, 12, color=self.colors["muted"], wraplength=1050, justify="left").pack(anchor="w", padx=24, pady=(0, 22))

    # ---------- history ----------
    def show_history(self):
        self.current = self.show_history
        self._activate_nav("history")
        self.page_title.configure(text="Analysis History")
        self.page_subtitle.configure(text="Click any argument or Open to reload the full saved analysis.")
        self._clear_page()
        hist = base.load_history()
        if not hist:
            c = self.card()
            c.grid(row=0, column=0, sticky="ew", pady=10)
            self.label(c, "No analyses yet", 22, "bold").pack(pady=(34, 4))
            self.label(c, "Run your first analysis and it will appear here.", 13, color=self.colors["muted"]).pack(pady=(0, 34))
            return
        for i, r in enumerate(hist):
            c = self.card()
            c.grid(row=i, column=0, sticky="ew", pady=7)
            c.grid_columnconfigure(0, weight=1)
            self.label(c, r.get("topic", "Untitled"), 17, "bold").grid(row=0, column=0, sticky="w", padx=20, pady=(16, 0))
            self.label(c, r.get("timestamp", "").replace("T", "  "), 11, color=self.colors["dim"]).grid(row=1, column=0, sticky="w", padx=20)
            ctk.CTkButton(c, text="Open", width=90, height=34, corner_radius=10, fg_color=self.accent,
                          hover_color=self.accent2, command=lambda rec=r: self.open_history_record(rec)
                          ).grid(row=0, column=1, rowspan=2, padx=20, pady=(14, 0))
            for n, k in enumerate(("person1", "person2")):
                p = r.get(k, {})
                row = ctk.CTkFrame(c, fg_color=self.colors["card2"], corner_radius=12)
                row.grid(row=2 + n, column=0, columnspan=2, sticky="ew", padx=16, pady=(10, 0) if n == 0 else (6, 0))
                row.grid_columnconfigure(0, weight=1)
                top = ctk.CTkFrame(row, fg_color="transparent")
                top.grid(row=0, column=0, sticky="w", padx=14, pady=(10, 0))
                self.label(top, p.get("name", k), 13, "bold", self.accent).pack(side="left")
                self.label(top, f'  {fmt(p.get("quality"), 1)} / 100  ', 12, color=self.colors["muted"]).pack(side="left")
                lab = p.get("stance_label")
                if lab:
                    self.pill(top, lab, STANCE.get(lab, self.accent)).pack(side="left")
                arg = self.label(row, short(p.get("argument", "")), 12, color=self.colors["text"],
                                 wraplength=900, justify="left")
                arg.grid(row=1, column=0, sticky="w", padx=14, pady=(4, 12))
                for w in (row, arg):
                    w.configure(cursor="hand2")
                    w.bind("<Button-1>", lambda e, rec=r: self.open_history_record(rec))
            ctk.CTkFrame(c, height=14, fg_color="transparent").grid(row=4, column=0)

    # ---------- research ----------
    def show_research(self):
        self.current = self.show_research
        self._activate_nav("research")
        self.page_title.configure(text="Research Dashboard")
        self.page_subtitle.configure(text="Model comparison and ablation from your training run.")
        self._clear_page()
        try:
            data = base.json.loads(base.METRICS_FILE.read_text(encoding="utf-8"))
        except Exception:
            data = {}
        row = 0
        for name in ("V2", "V3", "V4"):
            d = data.get(name, {})
            q, s = d.get("quality_test", {}), d.get("stance_test", {})
            c = self.card()
            c.grid(row=row, column=0, sticky="ew", pady=6)
            row += 1
            head = ctk.CTkFrame(c, fg_color="transparent")
            head.pack(fill="x", padx=22, pady=(16, 6))
            self.label(head, f"Model {name}", 18, "bold", self.accent).pack(side="left")
            if name == "V4":
                self.pill(head, "PRODUCTION", self.accent).pack(side="left", padx=12)
            tiles = ctk.CTkFrame(c, fg_color="transparent")
            tiles.pack(fill="x", padx=18, pady=(0, 16))
            items = [("MAE", fmt(q.get("MAE"))), ("RMSE", fmt(q.get("RMSE"))), ("R²", fmt(q.get("R2"))),
                     ("Stance accuracy", f'{float(s.get("Accuracy", 0)) * 100:.2f}%'), ("Stance F1", fmt(s.get("F1")))]
            for i, (k, v) in enumerate(items):
                tiles.grid_columnconfigure(i, weight=1, uniform="m")
                t = ctk.CTkFrame(tiles, fg_color=self.colors["card2"], corner_radius=12)
                t.grid(row=0, column=i, sticky="nsew", padx=4)
                self.label(t, v, 18, "bold").pack(anchor="w", padx=14, pady=(10, 0))
                self.label(t, k, 11, color=self.colors["muted"]).pack(anchor="w", padx=14, pady=(0, 10))
        c = self.card()
        c.grid(row=row, column=0, sticky="ew", pady=8)
        self.label(c, "Ablation analysis", 18, "bold").pack(anchor="w", padx=22, pady=(16, 8))
        abl = data.get("Ablation", [])
        best = min((a.get("MAE", 9) for a in abl), default=None)
        for a in abl:
            r = ctk.CTkFrame(c, fg_color=self.colors["card2"], corner_radius=10)
            r.pack(fill="x", padx=22, pady=3)
            self.label(r, a.get("feature_set", "-"), 13, "bold").pack(side="left", padx=14, pady=9)
            if a.get("MAE") == best:
                self.pill(r, "lowest MAE", self.accent).pack(side="left")
            self.label(r, f'MAE {fmt(a.get("MAE"))}    RMSE {fmt(a.get("RMSE"))}    R² {fmt(a.get("R2"))}', 12,
                       color=self.colors["muted"]).pack(side="right", padx=14)
        self.label(c, "Tree-based baselines are independent evaluation models; V4 remains the production hybrid.", 11,
                   color=self.colors["dim"]).pack(anchor="w", padx=22, pady=(12, 16))


if __name__ == "__main__":
    base.ensure_history()
    PrettyApp().mainloop()