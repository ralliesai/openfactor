from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Collapsible, DataTable, Footer, Header, Static


TOP_N = 8


def pct1(value):
    return "—" if value is None else f"{value * 100:.1f}%"


def signed_cell(value, bold=False):
    """Return a sign-colored percent cell."""
    if value is None:
        return Text("—", style="dim")
    text = Text(f"{value * 100:+.2f}%", style=("green" if value >= 0 else "red"))
    if bold:
        text.stylize("bold")
    return text


def signed(value):
    """Return a sign-colored signed-percent markup string."""
    if value is None:
        return "[dim]—[/]"
    return f"[{'green' if value >= 0 else 'red'}]{value * 100:+.2f}%[/]"


def expo_cell(value):
    """Return a signed exposure cell, red when short of the market."""
    return Text(f"{value:+.2f}", style="" if value >= 0 else "red")


def te_cell(value, bold=False):
    """Return a share-of-tracking-error cell; green when diversifying."""
    if value is None:
        return Text("—", style="dim")
    text = Text(f"{value * 100:+.1f}%", style=("bold" if bold else "") if value >= 0 else "green")
    return text


class OpenFactorTUI(App):
    """Interactive risk and return terminal for one portfolio."""

    CSS = """
    .subtitle { color: $text-muted; padding: 0 1; }
    #cards { height: auto; padding: 1 0; }
    .card { width: 1fr; height: 5; border: round $primary 40%; padding: 0 1; margin: 0 1 0 0; }
    .legend { color: $text-muted; padding: 0 1; }
    #columns { height: auto; }
    .col { width: 1fr; height: auto; }
    Collapsible { margin: 0 1 1 0; }
    DataTable { height: auto; margin: 0 1; }
    #horizons { height: auto; padding: 0 1 1 1; }
    Button { margin: 0 1 0 0; min-width: 12; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("e", "expand_all", "Expand all"),
        ("c", "collapse_all", "Collapse all"),
    ]

    def __init__(self, report):
        super().__init__()
        self.report = report
        self.horizon = len(report["horizons"]) - 1  # default to the longest window

    def compose(self) -> ComposeResult:
        active = sorted(self.report["active_rows"], key=lambda row: -abs(row["te_share"] or 0))
        tail = max(0, len(active) - TOP_N)
        yield Header(show_clock=False)
        with VerticalScroll():
            yield Static(self.header_line(), classes="subtitle")
            yield Horizontal(*self.cards(), id="cards")
            with Horizontal(id="columns"):
                with Vertical(classes="col"):
                    with Collapsible(title="Active risk — what's eating your tracking-error budget", collapsed=False):
                        yield DataTable(id="active", cursor_type="none", zebra_stripes=True)
                        with Collapsible(title=f"{tail} smaller factors", collapsed=True):
                            yield DataTable(id="active_tail", cursor_type="none", zebra_stripes=True)
                        yield Static("[dim]green = diversifying (reduces tracking error)[/]", classes="legend")
                    with Collapsible(title="Idiosyncratic risk — which names are the black box", collapsed=False):
                        yield Static(self.specific_summary(), classes="legend")
                        yield DataTable(id="specific", cursor_type="none", zebra_stripes=True)
                with Vertical(classes="col"):
                    with Collapsible(title="Return attribution", collapsed=False):
                        yield Horizontal(*self.horizon_buttons(), id="horizons")
                        yield Static(id="returns_summary", classes="legend")
                        yield DataTable(id="returns", cursor_type="none", zebra_stripes=True)
                    with Collapsible(title="Benchmark", collapsed=True):
                        yield Static(self.benchmark_text(), classes="legend")
                    with Collapsible(title="Tail risk & scenarios", collapsed=True):
                        yield Static(self.tail_text(), classes="legend")
        yield Footer()

    def on_mount(self):
        self.populate_active()
        self.populate_specific()
        self.populate_returns()

    # ---- headline -------------------------------------------------------
    def header_line(self):
        m = self.report["meta"]
        b = m["benchmark"]
        line = (f"[b]{m['universe']}[/] · as of {m['as_of_date']} · {m['tickers']} tickers · "
                f"{m['held']} held · benchmark: {b['name']} ({b['tagline']})")
        if m["missing"]:
            line += f"\n[yellow]dropped (not in universe): {', '.join(m['missing'])}[/]"
        return line

    def benchmark_text(self):
        b = self.report["meta"]["benchmark"]
        tilt = "—" if b["max_style_tilt"] is None else f"{b['max_style_tilt']:.2f}σ"
        return (
            f"[b]{b['name']}[/] · {b['tagline']}\n"
            f"{b['constituents']} constituents · top-10 weight {pct1(b['top10_weight'])} · "
            f"effective names {b['effective_names']:.0f}\n"
            f"[dim]Cap-weighted — the same methodology as standard large-cap indices. "
            f"It defines the market the style factors are measured against, so its own style tilts are "
            f"near zero (largest {tilt}) and its beta is 1.00. That makes it a faithful proxy for the "
            f"broad US large-cap market; active risk and beta are measured against it.[/]"
        )

    def cards(self):
        s = self.report["summary"]
        var = s["var"]["95%"]
        return [
            card("Total risk", pct1(s["total_risk"]), "annualized volatility"),
            card("Tracking error", pct1(s["tracking_error"]), "active risk vs benchmark"),
            card("1-day VaR (95%)", pct1(var["total_1d"]), f"active {pct1(var['active_1d'])}"),
            card("Predicted beta", "—" if s["beta"] is None else f"{s['beta']:.2f}", "to benchmark"),
            card("Idiosyncratic", pct1(s["specific_share_te"]), "of tracking error"),
            card("Info ratio", *self.ir_card()),
        ]

    def ir_card(self):
        """Return the info-ratio card value and sub-label.

        Example:
            a stored track record shows "realized, N days"; otherwise a backtest.
        """
        track = self.report.get("track")
        if track and track.get("ir") is not None:
            return f"{track['ir']:.2f}", f"realized, {track['days']} days"
        backtest = self.report["summary"]["ir"]["value"]
        return ("—" if backtest is None else f"{backtest:.2f}"), "backtest, not realized"

    # ---- tables ---------------------------------------------------------
    def populate_active(self):
        active = sorted(self.report["active_rows"], key=lambda row: -abs(row["te_share"] or 0))
        self.fill_active(self.query_one("#active", DataTable), active[:TOP_N], specific=True)
        self.fill_active(self.query_one("#active_tail", DataTable), active[TOP_N:], specific=False)

    def fill_active(self, table, rows, specific):
        table.add_columns("Factor", "Family", "Active Exp", "% of Tracking Error")
        for row in rows:
            table.add_row(row["label"], row["family"], expo_cell(row["active_exposure"]), te_cell(row["te_share"]))
        if specific:
            table.add_row(Text("Idiosyncratic (stock-picking)", style="bold"), "", "",
                          te_cell(self.report["specific_te_share"], bold=True))

    def populate_specific(self):
        table = self.query_one("#specific", DataTable)
        table.add_columns("Ticker", "Weight", "Idiosyncratic vol", "Share of idiosyncratic")
        for name in self.report["names"]["names"]:
            table.add_row(
                name["ticker"], pct1(name["weight"]),
                pct1(name["specific_vol"]), te_cell(name["share"]),
            )

    def populate_returns(self):
        h = self.horizon
        r = self.report
        rows = [row for row in r["active_rows"] if row.get("ret") and row["ret"][h] is not None]
        rows.sort(key=lambda row: -abs(row["ret"][h]))
        tail = max(0, len(rows) - TOP_N)
        badge = ("[green]your book's actual day[/]" if h == 0
                 else "[yellow]current-weights backtest — assumes you held today's book the whole window[/]")
        self.query_one("#returns_summary", Static).update(
            f"[b]{r['horizons'][h]}[/] · {r['horizon_dates'][h]}   {badge}\n"
            f"Benchmark {signed(r['benchmark_ret'][h])}  →  Portfolio {signed(r['portfolio_ret'][h])}"
            f"   =   Active (excess) {signed(r['active_ret'][h])}\n"
            f"[dim]rows below split the active return (top contributors; {tail} smaller factors folded)[/]"
        )
        table = self.query_one("#returns", DataTable)
        table.clear(columns=True)
        table.add_columns("Factor", "Family", "Contribution")
        for row in rows[:TOP_N]:
            table.add_row(row["label"], row["family"], signed_cell(row["ret"][h]))
        table.add_row(Text("Idiosyncratic (stock-picking)", style="bold"), "", signed_cell(r["specific_ret"][h]))
        table.add_row(Text("Active (excess)", style="bold"), "", signed_cell(r["active_ret"][h], bold=True))

    # ---- text panels ----------------------------------------------------
    def specific_summary(self):
        n = self.report["names"]
        eff = "—" if n["effective_names"] is None else f"{n['effective_names']:.1f}"
        top = n["names"][0]["ticker"] if n["names"] else "—"
        return (f"Idiosyncratic risk {pct1(n['total_specific'])} of the book · top name "
                f"[b]{top}[/] = {pct1(n['top_share'])} · effective names {eff}")

    def tail_text(self):
        s = self.report["summary"]
        lines = ["[b]Parametric VaR[/] (normal, one-day)"]
        for conf, value in s["var"].items():
            lines.append(f"  {conf}:  total {pct1(value['total_1d'])}   active {pct1(value['active_1d'])}")
        beta = "—" if s["beta"] is None else f"{s['beta']:.2f}"
        lines.append(f"[b]Predicted beta[/] to benchmark: {beta}")
        ir = s["ir"]
        value = "—" if ir["value"] is None else f"{ir['value']:.2f}"
        lines.append(f"[b]Information ratio[/] (backtest): {value}  (annualized active return ÷ tracking error)")
        lines.append(f"[dim]Backtest of current weights over the last {ir['days']} trading days — assumes you held "
                     "today's book the whole window, so it is not a skill track record.[/]")
        lines += self.track_lines()
        lines.append("[dim]Historical scenarios (2008, COVID, rates +100bp) need an external scenario set "
                     "that isn't in the published snapshot, so they're omitted rather than faked. "
                     "Days-to-liquidate needs volume data the snapshot doesn't carry.[/]")
        return "\n".join(lines)

    def track_lines(self):
        """Return Track-record lines when a --track file has accumulated days."""
        track = self.report.get("track")
        if not track or not track["days"]:
            return ["[dim]Pass --track <file> to store each day's result and build a REAL "
                    "track record (realized IR, hit rate) over time.[/]"]
        ir = "—" if track["ir"] is None else f"{track['ir']:.2f}"
        mean = "—" if track["mean"] is None else f"{track['mean'] * 100:+.3f}%"
        hit = "—" if track["hit_rate"] is None else f"{track['hit_rate'] * 100:.0f}%"
        cum = "—" if track["cumulative"] is None else f"{track['cumulative'] * 100:+.2f}%"
        return [
            f"[b]Track record[/] ({track['days']} day(s) stored)",
            f"  realized IR {ir} · mean daily active {mean} · hit rate {hit} · cumulative active {cum}",
            "[dim]This is your REAL accumulated record from stored days, not a backtest. "
            "More days → more reliable.[/]",
        ]

    # ---- interaction ----------------------------------------------------
    def horizon_buttons(self):
        return [Button(h, id=f"h{i}", variant=("primary" if i == self.horizon else "default"))
                for i, h in enumerate(self.report["horizons"])]

    def on_button_pressed(self, event):
        if event.button.id and event.button.id.startswith("h"):
            self.horizon = int(event.button.id[1:])
            for i in range(len(self.report["horizons"])):
                self.query_one(f"#h{i}", Button).variant = "primary" if i == self.horizon else "default"
            self.populate_returns()

    def action_expand_all(self):
        for widget in self.query(Collapsible):
            widget.collapsed = False

    def action_collapse_all(self):
        for widget in self.query(Collapsible):
            widget.collapsed = True


def card(title, value, sub):
    """Return one headline stat card."""
    return Static(f"[dim]{title}[/]\n[b]{value}[/]\n[dim]{sub}[/]", classes="card")
