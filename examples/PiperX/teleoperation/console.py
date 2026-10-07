"""Live terminal dashboard for ``record.py`` (falls back to plain log lines when not on a TTY)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

PHASE_STYLE = {
    "idle": ("IDLE", "bold white on grey30"),
    "homing": ("HOMING", "bold black on yellow"),
    "ready": ("READY", "bold white on green4"),
    "recording": ("● REC", "bold white on red3"),
    "saving": ("SAVING", "bold white on blue"),
    "stopped": ("DONE", "bold white on blue"),
    "teleop": ("TELEOP", "bold white on green4"),
}
NEXT = {
    "idle": "A  move home (reset the scene first)",
    "homing": "wait for home ·  B  cancel",
    "ready": "A  start recording ·  Y  re-home ·  X  align frame",
    "recording": "A  save ·  B  discard",
    "saving": "saving ...",
    "stopped": "all episodes recorded ·  q  quit",
    "teleop": "hold grip to move ·  Y  home ·  X  align frame",
}
KEYS = "Space/→ A · Backspace/← B · h home · p print pose · q quit (saves) · Ctrl+C abort"


@dataclass
class ConsoleState:
    phase: str = "idle"
    elapsed_s: float | None = None
    frames: int = 0
    episode: int | None = None
    saved: int = 0
    target: int = 0
    headset_connected: bool = False
    headset_hz: float = 0.0
    headset_age_ms: float | None = None
    control_hz: float = 0.0
    tracking: bool = False
    deadman: bool = False
    gripper_closed: bool = False
    ik_ok: bool = True
    encoder_backlog: int = 0
    ee_mm: tuple[float, float, float] | None = None
    joints_deg: list[float] = field(default_factory=list)
    gripper_mm: float | None = None
    message: str = ""


class Dashboard:
    def __init__(self, title: str, url: str, dataset: str | None, task: str | None):
        self.console = Console()
        self.enabled = self.console.is_terminal
        self.title = title
        self.url = url
        self.dataset = dataset
        self.task = task
        self.state = ConsoleState()
        self._live: Live | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self.enabled:
            self._live = Live(get_renderable=self._render, console=self.console, refresh_per_second=8, transient=False)
            self._live.start()

    def stop(self) -> None:
        if self._live is not None:
            self._live.refresh()
            self._live.stop()
            self._live = None

    def update(self, **values: Any) -> None:
        for key, value in values.items():
            setattr(self.state, key, value)

    def log(self, text: str, style: str = "") -> None:
        stamp = time.strftime("%H:%M:%S")
        if self.enabled:
            self.console.print(Text.assemble((stamp + "  ", "dim"), (text, style)))
        else:
            print(f"{stamp}  {text}", flush=True)

    # ------------------------------------------------------------------ rendering
    def _render(self):
        s = self.state
        label, style = PHASE_STYLE.get(s.phase, (s.phase.upper(), "bold"))
        head = Text.assemble((f" {label} ", style), "  ")
        if s.elapsed_s is not None:
            minutes, seconds = divmod(int(s.elapsed_s), 60)
            head.append(f"{minutes}:{seconds:02d}  ", "bold red" if s.phase == "recording" else "")
            head.append(f"{s.frames} frames   ", "dim")
        if s.episode is not None:
            head.append(f"episode {s.episode}", "bold")
            head.append(f" · saved {s.saved}/{s.target}", "dim")

        rows = Table.grid(padding=(0, 2))
        rows.add_column(style="dim", no_wrap=True)
        rows.add_column()
        rows.add_row("next", Text(NEXT.get(s.phase, ""), "cyan"))

        if s.headset_connected:
            age = f" · {s.headset_age_ms:.0f} ms" if s.headset_age_ms is not None and s.headset_age_ms < 5000 else ""
            headset = Text(f"connected · {s.headset_hz:.0f} Hz{age}", "green")
        else:
            headset = Text(f"waiting — open {self.url}", "yellow")
        headset.append(f"      control {s.control_hz:.1f} Hz", "dim")
        rows.add_row("headset", headset)

        hand = Text()
        hand.append("tracked" if s.tracking else "not tracked", "green" if s.tracking else "yellow")
        hand.append(" · ")
        hand.append("GRIP" if s.deadman else "grip released", "bold green" if s.deadman else "dim")
        hand.append(" · ")
        hand.append("gripper closed" if s.gripper_closed else "gripper open")
        hand.append(" · ")
        hand.append("IK ok" if s.ik_ok else "OUT OF REACH", "green" if s.ik_ok else "bold yellow")
        if s.encoder_backlog > 30:
            hand.append(f" · encoder backlog {s.encoder_backlog}", "bold yellow")
        rows.add_row("hand", hand)

        if s.ee_mm is not None:
            x, y, z = s.ee_mm
            grip = f"      gripper {s.gripper_mm:.1f} mm" if s.gripper_mm is not None else ""
            rows.add_row("EE", f"x {x:7.1f}  y {y:7.1f}  z {z:7.1f} mm{grip}")
        if s.joints_deg:
            rows.add_row("joints", "  ".join(f"{v:7.1f}" for v in s.joints_deg) + "  deg")
        if self.dataset:
            task = f' · "{self.task}"' if self.task else ""
            rows.add_row("dataset", Text(f"{self.dataset}{task}", overflow="ellipsis", no_wrap=True))
        if s.message:
            rows.add_row("last", Text(s.message, "italic"))

        return Panel(
            Group(head, Text(""), rows),
            title=f"[bold]{self.title}[/] · {self.url}",
            title_align="left",
            subtitle=f"[dim]{KEYS}[/]",
            subtitle_align="left",
            border_style="red" if s.phase == "recording" else "grey50",
            padding=(0, 1),
        )
