import os
import json
import csv
import inspect
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

from predict_v4 import analyze_two_arguments
from analysis.fallacy_detector import detect_fallacies
from analysis.argument_profile import get_argument_profile
from analysis.explainability import explain_argument
from analysis.improvement import generate_improvement_suggestions

try:
    from reports.pdf_report import generate_pdf_report
except Exception:
    generate_pdf_report = None

BASE = Path(__file__).resolve().parent
HISTORY_FILE = BASE / "data" / "history.json"
METRICS_FILE = BASE / "research" / "metrics.json"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

DARK = {
    "bg": "#0A0B10", "sidebar": "#0D0F16", "panel": "#12151F",
    "card": "#171B27", "card2": "#1B2030", "input": "#0F121A",
    "border": "#252B3A", "text": "#F5F7FB", "muted": "#A6AEC0",
    "dim": "#70798D", "white": "#FFFFFF"
}
LIGHT = {
    "bg": "#F4F6FA", "sidebar": "#FFFFFF", "panel": "#FFFFFF",
    "card": "#FFFFFF", "card2": "#F1F4F9", "input": "#F8F9FC",
    "border": "#DCE2EC", "text": "#172033", "muted": "#596579",
    "dim": "#8490A4", "white": "#172033"
}
THEMES = {
    "Purple": ("#8B5CF6", "#A78BFA"),
    "Blue": ("#3B82F6", "#60A5FA"),
    "Green": ("#10B981", "#34D399"),
    "Rose": ("#F43F5E", "#FB7185"),
}


def clean_text(text):
    """Normalize argument text before passing it to the prediction pipeline.

    The original text is retained separately for display/history.
    Do not remove negation words such as no, not, or never.
    """
    if text is None:
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ensure_history():
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text("[]", encoding="utf-8")


def load_history():
    ensure_history()
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def save_history(record):
    history = load_history()
    history.insert(0, record)
    HISTORY_FILE.write_text(json.dumps(history, indent=2), encoding="utf-8")


# ---------- high-confidence explicit stance correction ----------
# The trained V4 classifier remains the primary model. These rules only
# override it when the argument contains a very explicit first-person stance
# declaration (e.g. "I oppose..." / "I support..."). This prevents obvious
# cases where the statistical classifier misses a direct stance cue.
import re

_EXPLICIT_FOR = [
    r"\bi\s+(?:strongly\s+)?support\b",
    r"\bi\s+(?:strongly\s+)?agree\s+with\b",
    r"\bi\s+am\s+(?:strongly\s+)?in\s+favor\s+of\b",
    r"\bi['’]?m\s+(?:strongly\s+)?in\s+favor\s+of\b",
    r"\bi\s+(?:strongly\s+)?favor\b",
    r"\bi\s+(?:strongly\s+)?favour\b",
    r"\bi\s+approve\s+of\b",
    r"\bi\s+am\s+(?:strongly\s+)?for\b",
    r"\bi['’]?m\s+(?:strongly\s+)?for\b",
]

_EXPLICIT_AGAINST = [
    r"\bi\s+(?:strongly\s+)?oppose\b",
    r"\bi\s+(?:strongly\s+)?disagree\s+with\b",
    r"\bi\s+am\s+(?:strongly\s+)?against\b",
    r"\bi['’]?m\s+(?:strongly\s+)?against\b",
    r"\bi\s+am\s+(?:strongly\s+)?opposed\s+to\b",
    r"\bi['’]?m\s+(?:strongly\s+)?opposed\s+to\b",
    r"\bi\s+do\s+not\s+support\b",
    r"\bi\s+don['’]?t\s+support\b",
    r"\bi\s+do\s+not\s+agree\s+with\b",
    r"\bi\s+don['’]?t\s+agree\s+with\b",
]


def apply_explicit_stance_override(argument, model_stance):
    """Return an explicit stance only for high-confidence declarations."""
    text = str(argument).lower().replace("’", "'")

    against = any(re.search(pattern, text) for pattern in _EXPLICIT_AGAINST)
    support = any(re.search(pattern, text) for pattern in _EXPLICIT_FOR)

    # If both are present, retain the ML prediction rather than guessing.
    if against and not support:
        return "-1", "AGAINST", "explicit stance cue"
    if support and not against:
        return "1", "FOR", "explicit stance cue"

    raw = str(model_stance)
    label = {"1": "FOR", "-1": "AGAINST", "0": "NEUTRAL"}.get(raw, raw)
    return raw, label, "V4 classifier"


class MindProofApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("MindProof — Argument Intelligence Lab")
        self.geometry("1400x900")
        self.minsize(1120, 760)
        self.mode = "dark"
        self.colors = DARK
        self.theme_name = "Purple"
        self.accent, self.accent2 = THEMES[self.theme_name]
        self.current = None
        self._build_shell()
        self.show_new_analysis()

    # ---------- shell ----------
    def _build_shell(self):
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.sidebar = ctk.CTkFrame(self, width=235, corner_radius=0, fg_color=self.colors["sidebar"])
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_propagate(False)
        self.content = ctk.CTkFrame(self, corner_radius=0, fg_color=self.colors["bg"])
        self.content.grid(row=0, column=1, sticky="nsew")
        self.content.grid_rowconfigure(1, weight=1)
        self.content.grid_columnconfigure(0, weight=1)
        self._build_sidebar()
        self._build_topbar()
        self._build_page_area()

    def _build_sidebar(self):
        s = self.sidebar
        ctk.CTkLabel(s, text="◉  MindProof", font=ctk.CTkFont(size=25, weight="bold"), text_color=self.colors["text"]).pack(anchor="w", padx=22, pady=(28, 2))
        ctk.CTkLabel(s, text="ARGUMENT INTELLIGENCE LAB", font=ctk.CTkFont(size=9, weight="bold"), text_color=self.accent).pack(anchor="w", padx=24, pady=(0, 28))

        self.nav = {}
        for key, label, cmd in [
            ("new", "＋  New Analysis", self.show_new_analysis),
            ("history", "◷  History", self.show_history),
            ("research", "▦  Research Dashboard", self.show_research),
            ("method", "ⓘ  Methodology", self.show_methodology),
        ]:
            b = ctk.CTkButton(s, text=label, anchor="w", height=45, corner_radius=11, fg_color="transparent", hover_color=self.colors["card2"], text_color=self.colors["muted"], font=ctk.CTkFont(size=14, weight="bold"), command=cmd)
            b.pack(fill="x", padx=13, pady=4)
            self.nav[key] = b

        ctk.CTkFrame(s, height=2, fg_color=self.colors["border"]).pack(fill="x", padx=18, pady=22)
        ctk.CTkLabel(s, text="APPEARANCE", font=ctk.CTkFont(size=10, weight="bold"), text_color=self.colors["dim"]).pack(anchor="w", padx=22, pady=(0, 8))
        self.mode_switch = ctk.CTkSegmentedButton(s, values=["🌙 Dark", "☀ Light"], height=34, command=self.set_mode)
        self.mode_switch.set(
            "☀ Light" if self.mode == "light" else "🌙 Dark"
        )
        self.mode_switch.pack(fill="x", padx=17, pady=(0, 18))

        ctk.CTkLabel(s, text="ACCENT THEME", font=ctk.CTkFont(size=10, weight="bold"), text_color=self.colors["dim"]).pack(anchor="w", padx=22, pady=(0, 8))
        self.theme_buttons = {}
        for name, (a, _) in THEMES.items():
            b = ctk.CTkButton(s, text=name, height=37, corner_radius=10, fg_color=a, hover_color=a, text_color="#FFFFFF", font=ctk.CTkFont(size=12, weight="bold"), command=lambda n=name: self.set_theme(n))
            b.pack(fill="x", padx=17, pady=3)
            self.theme_buttons[name] = b

        ctk.CTkFrame(s, height=2, fg_color=self.colors["border"]).pack(fill="x", padx=18, pady=22)
        ctk.CTkLabel(s, text="CURRENT ANALYSIS", font=ctk.CTkFont(size=10, weight="bold"), text_color=self.colors["dim"]).pack(anchor="w", padx=22)
        self.side_pdf = ctk.CTkButton(s, text="▣  Export PDF", height=40, corner_radius=10, fg_color=self.colors["card2"], hover_color=self.accent, text_color=self.colors["text"], command=self.export_pdf)
        self.side_pdf.pack(fill="x", padx=17, pady=(9, 4))
        self.side_json = ctk.CTkButton(s, text="⇩  Export JSON", height=40, corner_radius=10, fg_color=self.colors["card2"], hover_color=self.accent, text_color=self.colors["text"], command=self.export_json)
        self.side_json.pack(fill="x", padx=17, pady=4)
        self.side_csv = ctk.CTkButton(s, text="⇩  Export CSV", height=40, corner_radius=10, fg_color=self.colors["card2"], hover_color=self.accent, text_color=self.colors["text"], command=self.export_csv)
        self.side_csv.pack(fill="x", padx=17, pady=4)

    def _build_topbar(self):
        self.topbar = ctk.CTkFrame(self.content, height=74, corner_radius=0, fg_color=self.colors["bg"])
        self.topbar.grid(row=0, column=0, sticky="ew", padx=28, pady=(14, 0))
        self.topbar.grid_columnconfigure(0, weight=1)
        self.page_title = ctk.CTkLabel(self.topbar, text="New Analysis", font=ctk.CTkFont(size=25, weight="bold"), text_color=self.colors["text"])
        self.page_title.grid(row=0, column=0, sticky="w")
        self.page_subtitle = ctk.CTkLabel(self.topbar, text="Analyze arguments independently. Understand structure, not truth.", font=ctk.CTkFont(size=12), text_color=self.colors["muted"])
        self.page_subtitle.grid(row=1, column=0, sticky="w", pady=(2, 0))
        self.new_top = ctk.CTkButton(self.topbar, text="＋  New Analysis", width=145, height=38, corner_radius=10, fg_color=self.accent, hover_color=self.accent2, text_color="#FFFFFF", command=self.show_new_analysis)
        self.new_top.grid(row=0, column=1, rowspan=2, padx=(20, 0))

    def _build_page_area(self):
        self.page = ctk.CTkScrollableFrame(self.content, corner_radius=0, fg_color=self.colors["bg"], scrollbar_button_color=self.colors["border"])
        self.page.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 12))
        self.page.grid_columnconfigure(0, weight=1)

    def _clear_page(self):
        for w in self.page.winfo_children():
            w.destroy()

    def _activate_nav(self, key):
        for k, b in self.nav.items():
            b.configure(fg_color=self.colors["card2"] if k == key else "transparent", text_color=self.colors["text"] if k == key else self.colors["muted"])

    # ---------- styles ----------
    def set_mode(self, value):
        self.mode = "light" if "Light" in value else "dark"
        ctk.set_appearance_mode(self.mode)
        self.colors = LIGHT if self.mode == "light" else DARK
        self.after(50, self._rebuild_shell)

    def _rebuild_shell(self):
        for w in self.winfo_children():
            w.destroy()
        self._build_shell()
        if self.current:
            self.current()
        else:
            self.show_new_analysis()

    def set_theme(self, name):
        self.theme_name = name
        self.accent, self.accent2 = THEMES[name]
        self._rebuild_shell()

    def card(self, parent=None, **kwargs):
        parent = parent or self.page
        return ctk.CTkFrame(parent, fg_color=self.colors["card"], border_width=1, border_color=self.colors["border"], corner_radius=18, **kwargs)

    def label(self, parent, text, size=14, weight="normal", color=None, **kwargs):
        return ctk.CTkLabel(parent, text=text, font=ctk.CTkFont(size=size, weight=weight), text_color=color or self.colors["text"], **kwargs)

    # ---------- new analysis ----------
    def show_new_analysis(self):
        self.current = self.show_new_analysis
        self._activate_nav("new")
        self.page_title.configure(text="New Analysis")
        self.page_subtitle.configure(text="Two arguments in. A structured, explainable analysis out.")
        self._clear_page()

        hero = self.card()
        hero.grid(row=0, column=0, sticky="ew", pady=(8, 12))
        hero.grid_columnconfigure(0, weight=1)
        self.label(hero, "ARGUMENT INTELLIGENCE LAB", 10, "bold", self.accent).grid(row=0, column=0, sticky="w", padx=25, pady=(22, 3))
        self.label(hero, "Understand the argument.", 31, "bold").grid(row=1, column=0, sticky="w", padx=25)
        self.label(hero, "MindProof predicts argument quality and stance, then adds linguistic insights, possible fallacy indicators and improvement ideas.", 13, color=self.colors["muted"], wraplength=950, justify="left").grid(row=2, column=0, sticky="w", padx=25, pady=(5, 20))

        topic_card = self.card()
        topic_card.grid(row=1, column=0, sticky="ew", pady=7)
        self.label(topic_card, "DEBATE TOPIC", 11, "bold", self.accent).pack(anchor="w", padx=22, pady=(18, 7))
        self.topic = ctk.CTkEntry(topic_card, height=46, corner_radius=11, fg_color=self.colors["input"], border_color=self.colors["border"], text_color=self.colors["text"], font=ctk.CTkFont(size=15), placeholder_text="e.g. Should social media platforms be regulated more strictly?")
        self.topic.pack(fill="x", padx=22, pady=(0, 20))

        people = ctk.CTkFrame(self.page, fg_color="transparent")
        people.grid(row=2, column=0, sticky="ew", pady=7)
        people.grid_columnconfigure((0,1), weight=1, uniform="p")
        self.p1 = self._argument_card(people, 0, "01", "Person 1")
        self.p2 = self._argument_card(people, 1, "02", "Person 2")

        action = ctk.CTkFrame(self.page, fg_color="transparent")
        action.grid(row=3, column=0, pady=15)
        self.analyze_btn = ctk.CTkButton(action, text="✦  Analyze Arguments", width=220, height=48, corner_radius=14, fg_color=self.accent, hover_color=self.accent2, text_color="#FFFFFF", font=ctk.CTkFont(size=15, weight="bold"), command=self.analyze)
        self.analyze_btn.pack(side="left", padx=6)
        ctk.CTkButton(action, text="↻  Clear", width=120, height=48, corner_radius=14, fg_color=self.colors["card2"], hover_color=self.colors["border"], text_color=self.colors["text"], font=ctk.CTkFont(size=14, weight="bold"), command=self.clear_form).pack(side="left", padx=6)

        info = self.card()
        info.grid(row=4, column=0, sticky="ew", pady=7)
        self.label(info, "HOW MINDPROOF WORKS", 10, "bold", self.accent).pack(anchor="w", padx=22, pady=(18, 5))
        self.label(info, "Topic + Argument  →  TF-IDF + handcrafted features + semantic embeddings  →  Quality + Stance", 16, "bold").pack(anchor="w", padx=22, pady=(0, 5))
        self.label(info, "The quality score reflects the dataset-trained target; it is not a factual truth score or a verdict about who is correct.", 12, color=self.colors["muted"], wraplength=1000, justify="left").pack(anchor="w", padx=22, pady=(0, 18))

    def _argument_card(self, parent, col, number, title):
        card = self.card(parent)
        card.grid(row=0, column=col, sticky="nsew", padx=(0, 6) if col == 0 else (6,0))
        top = ctk.CTkFrame(card, fg_color=self.accent, height=5, corner_radius=3)
        top.pack(fill="x", padx=18, pady=(16,0))
        row = ctk.CTkFrame(card, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=(16, 10))
        self.label(row, number, 11, "bold", self.colors["dim"]).pack(side="left")
        self.label(row, title, 18, "bold", self.accent).pack(side="left", padx=10)
        name = ctk.CTkEntry(card, height=40, corner_radius=10, fg_color=self.colors["input"], border_color=self.colors["border"], font=ctk.CTkFont(size=14), placeholder_text="Name")
        name.pack(fill="x", padx=20, pady=(0, 10))
        text = ctk.CTkTextbox(card, height=190, corner_radius=11, fg_color=self.colors["input"], border_width=1, border_color=self.colors["border"], font=ctk.CTkFont(size=14), wrap="word")
        text.pack(fill="both", expand=True, padx=20, pady=(0, 8))
        counter = self.label(card, "0 characters", 10, color=self.colors["dim"])
        counter.pack(anchor="e", padx=22, pady=(0, 15))
        text.bind("<KeyRelease>", lambda e: counter.configure(text=f"{len(text.get('1.0','end-1c'))} characters"))
        return {"name": name, "text": text}

    def clear_form(self):
        self.topic.delete(0, "end")
        for p in (self.p1, self.p2):
            p["name"].delete(0, "end")
            p["text"].delete("1.0", "end")

    # ---------- analysis ----------
    def analyze(self):
        topic = self.topic.get().strip()
        n1, a1 = self.p1["name"].get().strip(), self.p1["text"].get("1.0", "end-1c").strip()
        n2, a2 = self.p2["name"].get().strip(), self.p2["text"].get("1.0", "end-1c").strip()
        if not topic or not a1 or not a2:
            messagebox.showwarning("Missing information", "Please enter the topic and both arguments.")
            return
        if not n1: n1 = "Person 1"
        if not n2: n2 = "Person 2"
        self.analyze_btn.configure(text="Analyzing…", state="disabled")
        self.update_idletasks()
        try:
            # Feed the same cleaned strings shown in the results screen to V4.
            cleaned_a1 = clean_text(a1)
            cleaned_a2 = clean_text(a2)
            results = analyze_two_arguments(topic, cleaned_a1, cleaned_a2)

            # V4 compatibility: depending on the predictor version,
            # analyze_two_arguments may return either:
            #   {"person1": {...}, "person2": {...}}
            # or ({...}, {...}) / [{...}, {...}]
            # Normalize all supported forms before the UI accesses them.
            if isinstance(results, dict):
                if "person1" in results and "person2" in results:
                    raw_results = {"person1": results["person1"], "person2": results["person2"]}
                elif 0 in results and 1 in results:
                    raw_results = {"person1": results[0], "person2": results[1]}
                else:
                    raise TypeError(
                        "V4 returned a dictionary, but it does not contain person1/person2 results."
                    )
            elif isinstance(results, (tuple, list)) and len(results) == 2:
                raw_results = {"person1": results[0], "person2": results[1]}
            else:
                raise TypeError(
                    f"Unexpected V4 prediction format: {type(results).__name__}. "
                    "Expected two person results."
                )

            if not isinstance(raw_results["person1"], dict) or not isinstance(raw_results["person2"], dict):
                raise TypeError(
                    "V4 returned invalid person results. "
                    f"Person 1={type(raw_results['person1']).__name__}, "
                    f"Person 2={type(raw_results['person2']).__name__}"
                )

            records = {}
            for key, name, arg in (("person1", n1, a1), ("person2", n2, a2)):
                r = dict(raw_results[key])
                r["name"] = name
                r["argument"] = arg  # original text entered by the user
                r["cleaned_argument"] = clean_text(arg)  # normalized text sent to V4
                r["profile"] = get_argument_profile(arg)
                r["fallacies"] = detect_fallacies(arg)
                r["explanation"] = explain_argument(r["profile"], float(r["quality"]))
                r["improvements"] = generate_improvement_suggestions(r["profile"], r["fallacies"])
                corrected_stance, corrected_label, stance_source = apply_explicit_stance_override(
                    arg, r.get("stance", "0")
                )
                r["stance"] = corrected_stance
                r["stance_label"] = corrected_label
                r["stance_source"] = stance_source
                records[key] = r
            self.current = self.show_results
            self.current_record = {"timestamp": datetime.now().isoformat(timespec="seconds"), "topic": topic, **records}
            save_history(self.current_record)
            self.show_results()
        except Exception as e:
            messagebox.showerror("Analysis error", f"{type(e).__name__}: {e}")
            print("\nMindProof analysis error:", repr(e))
        finally:
            # The results page rebuilds the main content and destroys the old
            # Analyze button. Do not call configure() on that destroyed Tk
            # widget, otherwise Tk raises: "invalid command name ...label".
            try:
                btn = getattr(self, "analyze_btn", None)
                if btn is not None and btn.winfo_exists():
                    btn.configure(text="✦  Analyze Arguments", state="normal")
            except Exception:
                # The button may already have been destroyed while the page
                # was being rebuilt. This is harmless and should not mask the
                # actual analysis result/error.
                pass

    # ---------- results ----------
    def show_results(self):
        self._activate_nav("new")
        self.page_title.configure(text="Analysis Results")
        self.page_subtitle.configure(text="A neutral, explainable view of the model outputs and linguistic indicators.")
        self._clear_page()
        rec = self.current_record
        p1, p2 = rec["person1"], rec["person2"]
        q1, q2 = float(p1["quality"]), float(p2["quality"])

        top = self.card(); top.grid(row=0, column=0, sticky="ew", pady=(8,12))
        self.label(top, "ANALYSIS", 10, "bold", self.accent).pack(anchor="w", padx=24, pady=(18,3))
        self.label(top, rec["topic"], 22, "bold").pack(anchor="w", padx=24)
        self.label(top, "Model: MindProof V4  •  Quality regression + stance classification", 11, color=self.colors["muted"]).pack(anchor="w", padx=24, pady=(4,18))

        compare = ctk.CTkFrame(self.page, fg_color="transparent")
        compare.grid(row=1, column=0, sticky="ew", pady=5); compare.grid_columnconfigure((0,1), weight=1, uniform="r")
        self._result_card(compare,0,p1); self._result_card(compare,1,p2)

        neutral = self.card(); neutral.grid(row=2,column=0,sticky="ew",pady=10)
        self.label(neutral,"NEUTRAL COMPARISON",10,"bold",self.accent).pack(anchor="w",padx=22,pady=(18,5))
        if abs(q1-q2) < 0.05:
            txt="The two predicted argument-quality scores are very close."
        else:
            higher = p1["name"] if q1 > q2 else p2["name"]
            txt=f"{higher} received a higher predicted argument-quality score by {abs(q1-q2):.1f} points."
        self.label(neutral,txt,15,"bold").pack(anchor="w",padx=22,pady=(0,15))
        self.label(neutral,"This comparison describes model output only; it does not determine factual correctness or a winner.",11,color=self.colors["muted"]).pack(anchor="w",padx=22,pady=(0,18))
        ex=ctk.CTkFrame(neutral,fg_color="transparent"); ex.pack(anchor="e",padx=18,pady=(0,16))
        for text,cmd in (("PDF",self.export_pdf),("JSON",self.export_json),("CSV",self.export_csv)):
            ctk.CTkButton(ex,text=f"⇩  {text}",width=105,height=35,corner_radius=9,fg_color=self.accent,hover_color=self.accent2,text_color="#fff",command=cmd).pack(side="left",padx=4)

        for idx, p in enumerate((p1,p2), start=3):
            self._detail_card(p, idx)

        note=self.card(); note.grid(row=5,column=0,sticky="ew",pady=10)
        self.label(note,"⚠  INTERPRETATION NOTE",12,"bold",self.accent).pack(anchor="w",padx=22,pady=(16,4))
        self.label(note,"Fallacy indicators are conservative text-pattern signals, not proof. Explainability and improvement suggestions are auxiliary linguistic analyses; the V4 prediction is the trained ML output.",11,color=self.colors["muted"],wraplength=1050,justify="left").pack(anchor="w",padx=22,pady=(0,18))

    def _result_card(self,parent,col,p):
        c=self.card(parent); c.grid(row=0,column=col,sticky="nsew",padx=(0,6) if col==0 else (6,0))
        self.label(c,p["name"],17,"bold",self.accent).pack(anchor="w",padx=22,pady=(20,2))
        self.label(c,"PREDICTED ARGUMENT QUALITY",10,"bold",self.colors["dim"]).pack(anchor="w",padx=22,pady=(6,0))
        self.label(c,f'{float(p["quality"]):.1f}',43,"bold").pack(anchor="w",padx=22,pady=(0,0))
        self.label(c,"/ 100",12,color=self.colors["muted"]).pack(anchor="w",padx=24,pady=(0,7))
        pb=ctk.CTkProgressBar(c,height=10,corner_radius=6,fg_color=self.colors["border"],progress_color=self.accent); pb.set(max(0,min(1,float(p["quality"])/100))); pb.pack(fill="x",padx=22,pady=(0,15))
        badge=ctk.CTkLabel(c,text=f'  {p["stance_label"]}  ',fg_color=self.accent,corner_radius=9,text_color="#fff",font=ctk.CTkFont(size=12,weight="bold")); badge.pack(anchor="w",padx=22,pady=(0,20))

    def _detail_card(self,p,row):
        c=self.card(); c.grid(row=row,column=0,sticky="ew",pady=7)
        self.label(c,f'{p["name"]}  ·  Analysis Details',18,"bold").pack(anchor="w",padx=22,pady=(18,10))
        grid=ctk.CTkFrame(c,fg_color="transparent"); grid.pack(fill="x",padx=18); grid.grid_columnconfigure((0,1),weight=1,uniform="d")
        self._mini_section(grid,0,0,"🔍  Explainability",self._bullets(p["explanation"]))
        self._mini_section(grid,0,1,"📊  Argument Profile",self._profile_text(p["profile"]))
        self._mini_section(grid,1,0,"⚠  Possible Fallacy Indicators",self._fallacy_text(p["fallacies"]))
        self._mini_section(grid,1,1,"💡  Improvement Suggestions",self._bullets(p["improvements"]))
        self.label(c,"Original argument (as entered)",11,"bold",self.colors["dim"]).pack(anchor="w",padx=22,pady=(15,4))
        self.label(c,p.get("argument", ""),12,color=self.colors["muted"],wraplength=1050,justify="left").pack(anchor="w",padx=22,pady=(0,10))
        self.label(c,"Cleaned text (sent to the V4 predictor)",11,"bold",self.colors["dim"]).pack(anchor="w",padx=22,pady=(0,4))
        self.label(c,p.get("cleaned_argument", clean_text(p.get("argument", ""))),12,color=self.colors["muted"],wraplength=1050,justify="left").pack(anchor="w",padx=22,pady=(0,20))

    def _mini_section(self,parent,row,col,title,text):
        f=ctk.CTkFrame(parent,fg_color=self.colors["card2"],corner_radius=13); f.grid(row=row,column=col,sticky="nsew",padx=5,pady=5)
        self.label(f,title,13,"bold").pack(anchor="w",padx=15,pady=(13,7))
        self.label(f,text,11,color=self.colors["muted"],justify="left",wraplength=480).pack(anchor="w",padx=15,pady=(0,14))

    def _bullets(self,x):
        if isinstance(x,list): return "\n".join("• "+str(v) for v in x)
        return str(x)
    def _profile_text(self,p):
        keys=[("word_count","Words"),("unique_ratio","Unique ratio"),("sentence_count","Sentences"),("average_sentence_length","Avg sentence length"),("reasoning_markers","Reasoning markers"),("evidence_markers","Evidence markers"),("example_markers","Example markers"),("contrast_markers","Contrast markers")]
        return "   •   ".join(f"{label}: {p.get(k,'—')}" for k,label in keys)
    def _fallacy_text(self,fs):
        if not fs: return "No rule-based indicator detected."
        return "\n".join(f"• {f.get('name','Indicator')} — {f.get('confidence','Possible')}" for f in fs)

    # ---------- history ----------
    def open_history_record(self, record):
        """Open a previously saved analysis exactly as it was stored."""
        self.current_record = record
        self.current = self.show_results
        self.show_results()

    def show_history(self):
        self.current=self.show_history
        self._activate_nav("history")
        self.page_title.configure(text="Analysis History")
        self.page_subtitle.configure(text="Click an argument or View Analysis to reopen the complete saved analysis.")
        self._clear_page()

        hist=load_history()
        if not hist:
            c=self.card()
            c.grid(row=0,column=0,sticky="ew",pady=10)
            self.label(c,"No analyses yet",22,"bold").pack(pady=35)
            return

        for i,r in enumerate(hist):
            c=self.card()
            c.grid(row=i,column=0,sticky="ew",pady=7)
            c.grid_columnconfigure(0, weight=1)
            c.grid_columnconfigure(1, weight=0)

            topic = r.get("topic", "Untitled")
            timestamp = r.get("timestamp", "")
            p1 = r.get("person1", {})
            p2 = r.get("person2", {})

            self.label(c, topic, 17, "bold", self.accent).grid(
                row=0, column=0, sticky="w", padx=20, pady=(17,2)
            )
            self.label(c, timestamp, 10, color=self.colors["dim"]).grid(
                row=1, column=0, sticky="w", padx=20, pady=(0,10)
            )

            # The actual saved arguments are shown in the history card.
            # Clicking either argument opens the complete saved result.
            for row_no, person in ((2, p1), (3, p2)):
                name = person.get("name", f"Person {row_no-1}")
                argument = person.get("argument", "")
                quality = person.get("quality", "—")
                stance = person.get("stance_label", "—")
                text = f'{name}:  {argument}'

                arg_label = self.label(
                    c, text, 12, color=self.colors["text"],
                    wraplength=850, justify="left"
                )
                arg_label.grid(row=row_no, column=0, sticky="ew", padx=20, pady=(3,4))
                arg_label.configure(cursor="hand2")
                arg_label.bind(
                    "<Button-1>",
                    lambda event, rec=r: self.open_history_record(rec)
                )
                arg_label.bind(
                    "<Enter>",
                    lambda event, w=arg_label: w.configure(text_color=self.accent)
                )
                arg_label.bind(
                    "<Leave>",
                    lambda event, w=arg_label: w.configure(text_color=self.colors["text"])
                )

                self.label(
                    c, f"Quality: {quality}   •   Stance: {stance}",
                    10, color=self.colors["muted"]
                ).grid(row=row_no, column=1, sticky="e", padx=(8,20), pady=3)

            view_btn = ctk.CTkButton(
                c, text="↗  View Full Analysis", width=170, height=34,
                corner_radius=9, fg_color=self.accent,
                hover_color=self.accent2, text_color="#FFFFFF",
                command=lambda rec=r: self.open_history_record(rec)
            )
            view_btn.grid(row=4, column=0, columnspan=2, sticky="w", padx=20, pady=(10,17))

    # ---------- research ----------
    def show_research(self):
        self.current=self.show_research; self._activate_nav("research"); self.page_title.configure(text="Research Dashboard"); self.page_subtitle.configure(text="Model comparison, ablation and evaluation evidence from your training run."); self._clear_page()
        try: data=json.loads(METRICS_FILE.read_text(encoding="utf-8"))
        except Exception: data={}
        models=[("V2",data.get("V2",{})),("V3",data.get("V3",{})),("V4",data.get("V4",{}))]
        row=0
        for name,d in models:
            c=self.card(); c.grid(row=row,column=0,sticky="ew",pady=6); row+=1
            q=d.get("quality_test",{}); s=d.get("stance_test",{})
            self.label(c,name,18,"bold",self.accent).pack(anchor="w",padx=20,pady=(15,2))
            self.label(c,f'Quality  ·  MAE {q.get("MAE","—"):.4f}   RMSE {q.get("RMSE","—"):.4f}   R² {q.get("R2","—"):.4f}    |    Stance  ·  Accuracy {s.get("Accuracy",0)*100:.2f}%   F1 {s.get("F1",0):.4f}',12,color=self.colors["muted"]).pack(anchor="w",padx=20,pady=(0,15))
        c=self.card(); c.grid(row=row,column=0,sticky="ew",pady=10)
        self.label(c,"Ablation Analysis",18,"bold").pack(anchor="w",padx=20,pady=(16,8))
        for a in data.get("Ablation",[]):
            self.label(c,f'{a.get("feature_set","-"):<28}  MAE {a.get("MAE",0):.4f}   RMSE {a.get("RMSE",0):.4f}   R² {a.get("R2",0):.4f}',11,color=self.colors["muted"]).pack(anchor="w",padx=20,pady=2)
        self.label(c,"Tree-based baselines are shown as independent evaluation models; V4 remains the production hybrid model.",11,color=self.colors["dim"]).pack(anchor="w",padx=20,pady=(12,16))

    # ---------- methodology ----------
    def show_methodology(self):
        self.current=self.show_methodology; self._activate_nav("method"); self.page_title.configure(text="Methodology"); self.page_subtitle.configure(text="How the MindProof pipeline works from dataset to prediction."); self._clear_page()
        sections=[
            ("01","Dataset","IBM Argument Quality Ranking 30K. Train, development and test sets are kept separate for model development and final evaluation."),
            ("02","Feature extraction","V4 combines TF-IDF text features, 16 handcrafted linguistic features and 384-dimensional MiniLM semantic embeddings."),
            ("03","Quality model","A regression model predicts the dataset's WA quality target, which is mapped to a 0–100 display score."),
            ("04","Stance model","A separate classifier predicts stance. In the dataset convention: +1 = FOR, −1 = AGAINST, 0 = NEUTRAL."),
            ("05","Explainability","Linguistic markers are summarized separately so the interface can explain observable patterns without claiming they prove argument quality."),
            ("06","Research evaluation","V2/V3/V4, Random Forest, XGBoost and ablation experiments are recorded in research/metrics.json."),
        ]
        for i,(n,t,d) in enumerate(sections):
            c=self.card(); c.grid(row=i,column=0,sticky="ew",pady=6)
            self.label(c,n,11,"bold",self.accent).pack(anchor="w",padx=22,pady=(16,2)); self.label(c,t,19,"bold").pack(anchor="w",padx=22); self.label(c,d,12,color=self.colors["muted"],wraplength=1050,justify="left").pack(anchor="w",padx=22,pady=(4,17))

    # ---------- exports ----------
    def _require_record(self):
        if not getattr(self,"current_record",None):
            messagebox.showinfo("No analysis", "Run an analysis first."); return False
        return True
    def export_json(self):
        if not self._require_record(): return
        path=filedialog.asksaveasfilename(defaultextension=".json",filetypes=[("JSON","*.json")],initialfile="mindproof_analysis.json")
        if path:
            Path(path).write_text(json.dumps(self.current_record,indent=2),encoding="utf-8"); messagebox.showinfo("Export complete",f"JSON saved to:\n{path}")
    def export_csv(self):
        if not self._require_record(): return
        path=filedialog.asksaveasfilename(defaultextension=".csv",filetypes=[("CSV","*.csv")],initialfile="mindproof_analysis.csv")
        if path:
            with open(path,"w",newline="",encoding="utf-8") as f:
                w=csv.writer(f); w.writerow(["Person","Name","Quality","Stance","Argument"])
                for k in ("person1","person2"):
                    p=self.current_record[k]; w.writerow([k,p["name"],p["quality"],p["stance_label"],p["argument"]])
            messagebox.showinfo("Export complete",f"CSV saved to:\n{path}")
    def export_pdf(self):
        if not self._require_record(): return
        if generate_pdf_report is None:
            messagebox.showerror("PDF unavailable","Install ReportLab first:\n\npy -m pip install reportlab")
            return
        path=filedialog.asksaveasfilename(defaultextension=".pdf",filetypes=[("PDF","*.pdf")],initialfile="mindproof_analysis.pdf")
        if not path: return
        r=self.current_record
        try:
            sig=inspect.signature(generate_pdf_report)
            params=list(sig.parameters)
            args=[r.get("topic"),r.get("person1"),r.get("person2"),path]
            if len(params)==4: generate_pdf_report(*args)
            elif len(params)==3: generate_pdf_report(r.get("topic"),r.get("person1"),path)
            elif len(params)==2: generate_pdf_report(r,path)
            else: generate_pdf_report(r)
            messagebox.showinfo("PDF exported",f"Professional PDF saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("PDF export error",f"{type(e).__name__}: {e}\n\nIf ReportLab is missing, run:\npy -m pip install reportlab")


if __name__ == "__main__":
    ensure_history()
    app=MindProofApp()
    app.mainloop()
