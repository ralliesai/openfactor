from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Collapsible, DataTable, Footer, Header, Static


TOP_N = 8


def missing(value):
    return value is None or value != value


def pct1(value):
    return "—" if missing(value) else f"{value * 100:.1f}%"


def beta_value(value):
    return "—" if missing(value) else f"{value:.2f}"


def magnitude(value):
    return 0.0 if missing(value) else abs(float(value))


def signed_cell(value, bold=False):
    """Return a sign-colored percent cell."""
    if missing(value):
        return Text("—", style="dim")
    text = Text(f"{value * 100:+.2f}%", style=("green" if value >= 0 else "red"))
    if bold:
        text.stylize("bold")
    return text


def signed(value):
    """Return a sign-colored signed-percent markup string."""
    if missing(value):
        return "[dim]—[/]"
    return f"[{'green' if value >= 0 else 'red'}]{value * 100:+.2f}%[/]"


def expo_cell(value):
    """Return a signed exposure cell, red when short of the market."""
    if missing(value):
        return Text("—", style="dim")
    return Text(f"{value:+.2f}", style="" if value >= 0 else "red")


def vol_cell(value, bold=False):
    """Return an annualized volatility or risk contribution cell."""
    if missing(value):
        return Text("—", style="dim")
    text = Text(f"{value * 100:.1f}%")
    if bold:
        text.stylize("bold")
    return text


def contribution_cell(value):
    """Return a signed contribution-to-risk cell."""
    if missing(value):
        return Text("—", style="dim")
    return Text(f"{value * 100:+.1f}%", style="" if value >= 0 else "green")


def te_cell(value, bold=False):
    """Return a share-of-tracking-error cell; green when diversifying."""
    if missing(value):
        return Text("—", style="dim")
    text = Text(f"{value * 100:+.1f}%", style=("bold" if bold else "") if value >= 0 else "green")
    return text


def share_cell(value, bold=False):
    """Return an unsigned risk-share cell, except for diversifying negatives."""
    if missing(value):
        return Text("—", style="dim")
    text = Text(f"{value * 100:.1f}%" if value >= 0 else f"{value * 100:+.1f}%",
                style=("bold" if bold else "") if value >= 0 else "green")
    return text


def label_cell(row):
    """Return a table label styled by row type."""
    text = Text(str(row["label"]))
    if row["kind"] in ("section", "total"):
        text.stylize("bold")
    elif row["kind"] == "group":
        text.stylize("dim")
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
        self.horizon = 0  # default to 1 Day — your book's actual return

    def compose(self) -> ComposeResult:
        active = sorted(self.report["active_rows"], key=lambda row: -abs(row["te_share"] or 0))
        tail = max(0, len(active) - TOP_N)
        _, risk_tail = self.risk_row_sets()
        yield Header(show_clock=False)
        with VerticalScroll():
            yield Static(self.header_line(), classes="subtitle")
            yield Horizontal(*self.cards(), id="cards")
            with Horizontal(id="columns"):
                with Vertical(classes="col"):
                    with Collapsible(title="Portfolio risk — current snapshot", collapsed=False):
                        yield Static(self.risk_summary(), classes="legend")
                        yield DataTable(id="risk", cursor_type="none", zebra_stripes=True)
                        if risk_tail:
                            with Collapsible(title=f"{len(risk_tail)} smaller risk drivers", collapsed=True):
                                yield DataTable(id="risk_tail", cursor_type="none", zebra_stripes=True)
                    with Collapsible(title="Active risk — tracking-error drivers", collapsed=False):
                        yield DataTable(id="active", cursor_type="none", zebra_stripes=True)
                        with Collapsible(title=f"{tail} smaller factors", collapsed=True):
                            yield DataTable(id="active_tail", cursor_type="none", zebra_stripes=True)
                        yield Static("[dim]green rows reduce tracking error through covariance.[/]", classes="legend")
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
                    with Collapsible(title="Parametric loss & beta", collapsed=True):
                        yield Static(self.tail_text(), classes="legend")
                    with Collapsible(title="Footnotes", collapsed=True):
                        yield Static(self.footnotes_text(), classes="legend")
        yield Footer()

    def on_mount(self):
        self.populate_risk()
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
            f"[dim]Cap-weighted model benchmark: the same weighting approach as standard large-cap indices. "
            f"It defines the market the style factors are measured against, so its own style tilts are "
            f"near zero (largest {tilt}) and its ex-ante beta is 1.00. That makes it a faithful proxy for the "
            f"broad US large-cap market; active risk and beta are measured against it.[/]"
        )

    def cards(self):
        s = self.report["summary"]
        var = s["var"]["95%"]
        return [
            card("Total risk", pct1(s["total_risk"]), "annualized volatility"),
            card("Tracking error", pct1(s["tracking_error"]), "active risk vs benchmark"),
            card("1-day VaR (95%)", pct1(var["total_1d"]), f"active {pct1(var['active_1d'])}"),
            card("Ex-ante beta", beta_value(s["predicted_beta"]), self.beta_card_sub()),
            card("Idiosyncratic", pct1(s["specific_share_te"]), "of tracking error"),
            card("Info ratio", *self.ir_card()),
        ]

    def beta_card_sub(self):
        """Return the beta card sub-label without mixing forecast and realized beta."""
        track = self.report.get("track")
        if track and track.get("realized_beta") is not None:
            return f"realized {track['realized_beta']:.2f} / {track['days']}d"
        return "model forecast"

    def ir_card(self):
        """Return the info-ratio card value and sub-label.

        Example:
            a stored --track record shows "realized, N days"; without one the
            ratio stays blank rather than being backtested from today's weights.
        """
        track = self.report.get("track")
        if track and track.get("ir") is not None:
            return f"{track['ir']:.2f}", f"realized, {track['days']} days"
        if track and track.get("days"):
            return "—", f"needs more days ({track['days']})"
        return "—", "needs --track history"

    # ---- tables ---------------------------------------------------------
    def risk_row_sets(self):
        """Return top portfolio-risk rows and a collapsed factor tail."""
        rows = self.report["risk_rows"]
        factors = [row for row in rows if row["kind"] == "factor"]
        top_keys = {row["key"] for row in sorted(factors, key=lambda row: -magnitude(row["pct"]))[:TOP_N]}
        top = [row for row in rows if row["kind"] != "factor" or row["key"] in top_keys]
        tail = [row for row in factors if row["key"] not in top_keys]
        return top, sorted(tail, key=lambda row: -magnitude(row["pct"]))

    def populate_risk(self):
        top, tail = self.risk_row_sets()
        table = self.query_one("#risk", DataTable)
        self.fill_risk(table, top)
        if tail:
            self.fill_risk(self.query_one("#risk_tail", DataTable), tail)

    def fill_risk(self, table, rows):
        table.add_columns("Source", "Exposure", "Vol", "% Total")
        for row in rows:
            table.add_row(
                label_cell(row),
                expo_cell(row["exposure"]),
                vol_cell(row["volatility"], bold=row["kind"] in ("section", "total")),
                share_cell(row["pct"], bold=row["kind"] in ("section", "total")),
            )

    def populate_active(self):
        active = sorted(self.report["active_rows"], key=lambda row: -abs(row["te_share"] or 0))
        self.fill_active(self.query_one("#active", DataTable), active[:TOP_N], specific=True)
        self.fill_active(self.query_one("#active_tail", DataTable), active[TOP_N:], specific=False)

    def fill_active(self, table, rows, specific):
        table.add_columns("Factor", "Family", "Active", "Factor Vol", "TE Contrib", "% TE")
        for row in rows:
            table.add_row(
                row["label"],
                row["family"],
                expo_cell(row["active_exposure"]),
                vol_cell(row["factor_volatility"]),
                contribution_cell(row["te_contribution"]),
                te_cell(row["te_share"]),
            )
        if specific:
            table.add_row(
                Text("Idiosyncratic residual", style="bold"),
                "",
                "",
                "",
                vol_cell(self.report["specific_te_contribution"], bold=True),
                te_cell(self.report["specific_te_share"], bold=True),
            )

    def populate_specific(self):
        table = self.query_one("#specific", DataTable)
        table.add_columns("Ticker", "Weight", "Idiosyncratic vol", "Share of idiosyncratic")
        for name in self.report["names"]["names"]:
            table.add_row(
                name["ticker"], pct1(name["weight"]),
                pct1(name["specific_vol"]), te_cell(name["share"]),
            )

    def populate_returns(self):
        if self.horizon == "track":
            self.populate_realized()
            return
        h = self.horizon
        r = self.report
        rows = [row for row in r["active_rows"] if row.get("ret") and row["ret"][h] is not None]
        rows.sort(key=lambda row: -abs(row["ret"][h]))
        tail = max(0, len(rows) - TOP_N)
        badge = ("[green]your book's actual day[/]" if h == 0
                 else "[yellow]current-weights backtest — assumes you held today's book the whole window[/]")
        self.query_one("#returns_summary", Static).update(
            f"[b]{r['horizons'][h]}[/] · {r['horizon_dates'][h]}   {badge}\n"
            f"Model benchmark {signed(r['benchmark_ret'][h])}  →  Portfolio {signed(r['portfolio_ret'][h])}"
            f"   =   Active (excess) {signed(r['active_ret'][h])}\n"
            f"[dim]rows below split the active return (top contributors; {tail} smaller factors folded)[/]"
        )
        table = self.query_one("#returns", DataTable)
        table.clear(columns=True)
        table.add_columns("Factor", "Family", "Contribution")
        for row in rows[:TOP_N]:
            table.add_row(row["label"], row["family"], signed_cell(row["ret"][h]))
        table.add_row(Text("Idiosyncratic residual", style="bold"), "", signed_cell(r["specific_ret"][h]))
        table.add_row(Text("Active (excess)", style="bold"), "", signed_cell(r["active_ret"][h], bold=True))

    def populate_realized(self):
        """Render attribution summed over the real daily holdings from --track."""
        r = self.report
        real = r["realized"]
        lookup = {row["factor"]: (row["label"], row["family"]) for row in r["active_rows"]}
        items = sorted(real["factor"].items(), key=lambda kv: -abs(kv[1]))
        tail = max(0, len(items) - TOP_N)
        self.query_one("#returns_summary", Static).update(
            f"[b]Realized · {real['days']} trading day(s)[/] · {real['date_range']}   "
            "[green]your actual book, summed day by day[/]\n"
            f"Model benchmark {signed(real['benchmark'])}  →  Portfolio {signed(real['portfolio'])}"
            f"   =   Active (excess) {signed(real['active'])}\n"
            f"[dim]each day's real holdings — not today's weights run backward; "
            f"top contributors ({tail} smaller factors folded)[/]"
        )
        table = self.query_one("#returns", DataTable)
        table.clear(columns=True)
        table.add_columns("Factor", "Family", "Contribution")
        for factor, value in items[:TOP_N]:
            label, fam = lookup.get(factor, (factor, "—"))
            table.add_row(label, fam, signed_cell(value))
        table.add_row(Text("Idiosyncratic residual", style="bold"), "", signed_cell(real["specific"]))
        table.add_row(Text("Active (excess)", style="bold"), "", signed_cell(real["active"], bold=True))

    # ---- text panels ----------------------------------------------------
    def risk_summary(self):
        s = self.report["summary"]
        return (
            f"Common factor {pct1(s['factor_share'])} of total variance · "
            f"idiosyncratic {pct1(s['specific_share_total'])} · "
            f"active tracking error {pct1(s['tracking_error'])}"
        )

    def specific_summary(self):
        n = self.report["names"]
        eff = "—" if n["effective_names"] is None else f"{n['effective_names']:.1f}"
        top = n["names"][0]["ticker"] if n["names"] else "—"
        return (f"Idiosyncratic risk {pct1(n['total_specific'])} of the book · top name "
                f"[b]{top}[/] = {pct1(n['top_share'])} · effective names {eff}")

    def tail_text(self):
        s = self.report["summary"]
        lines = ["[b]Parametric loss estimate[/] (normal, one-day)"]
        for conf, value in s["var"].items():
            lines.append(f"  {conf}:  total {pct1(value['total_1d'])}   active {pct1(value['active_1d'])}")
        lines.append(f"[b]Ex-ante beta[/] to benchmark: {beta_value(s['predicted_beta'])}")
        lines += self.track_lines()
        lines.append("[dim]Historical and macro scenarios are not shown because the published snapshot "
                     "does not carry an external shock library. They are omitted rather than faked.[/]")
        return "\n".join(lines)

    def track_lines(self):
        """Return Track-record lines when a --track file has accumulated days."""
        track = self.report.get("track")
        if not track or not track["days"]:
            return ["[dim]Pass --track <file> to store each day's result and build a REAL "
                    "track record (realized IR, hit rate) over time.[/]"]
        ir = "—" if track["ir"] is None else f"{track['ir']:.2f}"
        rb = "—" if track.get("realized_beta") is None else f"{track['realized_beta']:.2f}"
        mean = "—" if track["mean"] is None else f"{track['mean'] * 100:+.3f}%"
        hit = "—" if track["hit_rate"] is None else f"{track['hit_rate'] * 100:.0f}%"
        cum = "—" if track["cumulative"] is None else f"{track['cumulative'] * 100:+.2f}%"
        return [
            f"[b]Track record[/] ({track['days']} day(s) stored)",
            f"  realized beta {rb} · realized IR {ir} · mean daily active {mean}",
            f"  hit rate {hit} · cumulative active {cum}",
            "[dim]This is your REAL accumulated record from stored days, not a backtest. "
            "More days → more reliable.[/]",
        ]

    def footnotes_text(self):
        return "\n".join([
            "[1] Ex-ante risk and beta use today's holdings, current exposures, and the model covariance. "
            "They are forecasts, not realized history.",
            "[2] Realized beta appears only with --track history and is computed from stored portfolio "
            "returns versus stored model-benchmark returns.",
            "[3] Tracking error is active risk: portfolio minus the cap-weighted model benchmark.",
            "[4] TE contribution is variance contribution divided by total tracking error; negative values "
            "are diversifiers in the current covariance matrix.",
            "[5] Idiosyncratic risk is stock-level residual risk after common factors. It is not automatically "
            "stock-picking skill.",
            "[6] The 1-week attribution uses today's holdings run backward. The Realized button, when present, "
            "sums the holdings actually stored by --track.",
        ])

    # ---- interaction ----------------------------------------------------
    def horizon_buttons(self):
        buttons = [Button(h, id=f"h{i}", variant=("primary" if i == self.horizon else "default"))
                   for i, h in enumerate(self.report["horizons"])]
        realized = self.report.get("realized")
        if realized:
            buttons.append(Button(f"Realized · {realized['days']}d", id="htrack",
                                  variant=("success" if self.horizon == "track" else "default")))
        return buttons

    def on_button_pressed(self, event):
        bid = event.button.id
        if not bid or not bid.startswith("h"):
            return
        self.horizon = "track" if bid == "htrack" else int(bid[1:])
        for i in range(len(self.report["horizons"])):
            self.query_one(f"#h{i}", Button).variant = "primary" if i == self.horizon else "default"
        if self.report.get("realized"):
            self.query_one("#htrack", Button).variant = "success" if self.horizon == "track" else "default"
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
