import asyncio

from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Collapsible, DataTable, Footer, Header, Input, Markdown, Static

from openfactor.llm.report_chat import ReportChat


TOP_N = 8
CHAT_PLACEHOLDER = "Ask about beta, TE, attribution..."
CHAT_WAITING_FRAMES = ("Thinking", "Thinking.", "Thinking..", "Thinking...")


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


def ratio_cell(value, bold=False):
    """Return a signed ratio cell for return attribution shares."""
    if missing(value):
        return Text("—", style="dim")
    text = Text(f"{value * 100:+.1f}%", style=("green" if value >= 0 else "red"))
    if bold:
        text.stylize("bold")
    return text


def idiosyncratic_share_cell(value, contribution):
    """Return a name-residual share colored by contribution direction."""
    if missing(value):
        return Text("—", style="dim")
    return Text(f"{value * 100:+.1f}%", style=("green" if contribution >= 0 else "red"))


def label_cell(row):
    """Return a table label styled by row type."""
    text = Text(str(row["label"]))
    if row["kind"] in ("section", "total"):
        text.stylize("bold")
    elif row["kind"] == "group":
        text.stylize("dim")
    return text


def return_label_cell(row):
    """Return a return-attribution label styled by row type."""
    text = Text(str(row["label"]))
    if row["kind"] in ("section", "total"):
        text.stylize("bold")
    return text


class OpenFactorApp(App):
    """Interactive risk and return terminal for one portfolio."""

    TITLE = "OpenFactor"

    CSS = """
    .subtitle { color: $text-muted; padding: 0 1; }
    #cards { height: auto; padding: 1 0; }
    .card { width: 1fr; height: 5; border: round $primary 40%; padding: 0 1; margin: 0 1 0 0; }
    .legend { color: $text-muted; padding: 0 1; }
    #main { height: 1fr; }
    #report_area { width: 1fr; height: 1fr; }
    #report_scroll { width: 1fr; height: 1fr; scrollbar-size-horizontal: 1; scrollbar-size-vertical: 0; }
    #columns { height: auto; }
    .col { width: 1fr; height: auto; }
    #chat_sidebar { width: 52; height: 1fr; border-left: vkey $foreground 25%; padding: 0 1 0 2; }
    #chat_title { height: auto; padding: 0 0 1 0; }
    #chat_log { height: 1fr; padding: 0; overflow-y: auto; overflow-x: hidden; scrollbar-size-horizontal: 0; scrollbar-size-vertical: 1; }
    #chat_input { height: 3; border: none; padding: 0 1; }
    #chat_input:focus { border: none; background-tint: $foreground 5%; }
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

    def __init__(self, report, snapshot=None):
        super().__init__()
        self.report = report
        self.snapshot = snapshot
        self.horizon = 0  # default to 1 Day — your book's actual return
        self.chat = ReportChat.from_env(report, snapshot=snapshot)
        self.chat_history = []
        self.chat_transcript = ""
        self.chat_wait_timer = None
        self.chat_wait_frame = 0

    def compose(self) -> ComposeResult:
        active = sorted(self.report["active_rows"], key=lambda row: -abs(row["te_share"] or 0))
        tail = max(0, len(active) - TOP_N)
        _, risk_tail = self.risk_row_sets()
        yield Header(show_clock=False)
        with Horizontal(id="main"):
            with Vertical(id="report_area"):
                yield Static(self.header_line(), classes="subtitle")
                yield Horizontal(*self.cards(), id="cards")
                with VerticalScroll(id="report_scroll"):
                    with Horizontal(id="columns"):
                        with Vertical(classes="col"):
                            with Collapsible(title="Portfolio risk — current report", collapsed=False):
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
                            with Collapsible(title="Idiosyncratic risk — stock-level residuals", collapsed=False):
                                yield Static(self.idiosyncratic_summary(), classes="legend")
                                yield DataTable(id="idiosyncratic", cursor_type="none", zebra_stripes=True)
                        with Vertical(classes="col"):
                            with Collapsible(title="Active return attribution", collapsed=False):
                                yield Horizontal(*self.horizon_buttons(), id="horizons")
                                yield Static(id="returns_summary", classes="legend")
                                yield Static("[b]Active return reconciliation[/]", classes="legend")
                                yield DataTable(id="returns_recon", cursor_type="none", zebra_stripes=True)
                                yield Static("[b]Top active return contributors[/]", classes="legend")
                                yield DataTable(id="returns_factors", cursor_type="none", zebra_stripes=True)
                            with Collapsible(title="Idiosyncratic return — name drivers", collapsed=False):
                                yield DataTable(id="idiosyncratic_returns", cursor_type="none", zebra_stripes=True)
                            with Collapsible(title="Benchmark", collapsed=True):
                                yield Static(self.benchmark_text(), classes="legend")
                            with Collapsible(title="Parametric loss & beta", collapsed=True):
                                yield Static(self.tail_text(), classes="legend")
                            with Collapsible(title="Footnotes", collapsed=True):
                                yield Static(self.footnotes_text(), classes="legend")
            if self.chat:
                with Vertical(id="chat_sidebar"):
                    yield Static("[b]Ask OpenFactor[/]\n[dim]Questions use this report.[/]", id="chat_title")
                    yield Markdown("", id="chat_log", open_links=False)
                    yield Input(placeholder=CHAT_PLACEHOLDER, id="chat_input")
        yield Footer()

    async def on_mount(self):
        self.populate_risk()
        self.populate_active()
        self.populate_idiosyncratic()
        self.populate_idiosyncratic_returns()
        self.populate_returns()
        if self.chat:
            await self.append_chat("OpenFactor", "Ask about this report, beta hedges, tracking error, or attribution.")

    async def _on_exit_app(self):
        try:
            if self.chat:
                self.chat.close()
        finally:
            await super()._on_exit_app()

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
        if b.get("kind") in {"index", "missing"}:
            tilt = "—" if b["risk_max_style_tilt"] is None else f"{b['risk_max_style_tilt']:.2f}σ"
            source = b["return_source"] if b.get("kind") == "index" else f"{b['return_source']} missing"
            return (
                f"[b]{b['name']}[/] · {b['tagline']}\n"
                f"{b['ticker']} returns from {source} · risk proxy {b['risk_name']}\n"
                f"[dim]Return attribution is benchmarked to the public index return. Ex-ante tracking error "
                f"and beta still use the model risk proxy ({b['risk_constituents']} constituents, "
                f"top-10 weight {pct1(b['risk_top10_weight'])}, effective names "
                f"{b['risk_effective_names']:.0f}, largest style tilt {tilt}) until index look-through "
                f"or index factor exposures are published.[/]"
            )
        tilt = "—" if b["max_style_tilt"] is None else f"{b['max_style_tilt']:.2f}σ"
        return (
            f"[b]{b['name']}[/] · {b['tagline']}\n"
            f"{b['constituents']} constituents · top-10 weight {pct1(b['top10_weight'])} · "
            f"effective names {b['effective_names']:.0f}\n"
            f"[dim]Cap-weighted model risk proxy: the same weighting approach as standard large-cap indices. "
            f"It defines the market the style factors are measured against, so its own style tilts are "
            f"near zero (largest {tilt}) and its ex-ante beta is 1.00. That makes it a faithful proxy for the "
            f"broad US large-cap market; active risk and beta are measured against it.[/]"
        )

    def cards(self):
        s = self.report["summary"]
        var = s["var"]["95%"]
        return [
            card("Total risk", pct1(s["total_risk"]), "annualized volatility"),
            card("Tracking error", pct1(s["tracking_error"]), s["tracking_error_label"]),
            card("1-day VaR (95%)", pct1(var["total_1d"]), f"active {pct1(var['active_1d'])}"),
            card("Ex-ante beta", beta_value(s["predicted_beta"]), self.beta_card_sub()),
            card("Idiosyncratic", pct1(s["idiosyncratic_share_of_tracking_error"]), "of tracking error"),
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
        self.fill_active(self.query_one("#active", DataTable), active[:TOP_N], include_idiosyncratic=True)
        self.fill_active(self.query_one("#active_tail", DataTable), active[TOP_N:], include_idiosyncratic=False)

    def fill_active(self, table, rows, include_idiosyncratic):
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
        if include_idiosyncratic:
            table.add_row(
                Text("Idiosyncratic risk", style="bold"),
                "",
                "",
                "",
                vol_cell(self.report["idiosyncratic_te_contribution"], bold=True),
                te_cell(self.report["idiosyncratic_te_share"], bold=True),
            )

    def populate_idiosyncratic(self):
        table = self.query_one("#idiosyncratic", DataTable)
        table.add_columns("Ticker", "Weight", "Idiosyncratic vol", "Share of idiosyncratic")
        for name in self.report["idiosyncratic_risk_by_name"]["rows"]:
            table.add_row(
                name["ticker"], pct1(name["weight"]),
                pct1(name["idiosyncratic_vol"]), te_cell(name["idiosyncratic_variance_share"]),
            )

    def populate_idiosyncratic_returns(self):
        table = self.query_one("#idiosyncratic_returns", DataTable)
        table.clear(columns=True)
        table.add_columns("Ticker", idiosyncratic_weight_label(self.horizon), "Contribution", "% Idio")
        rows = self.idiosyncratic_return_rows()
        if not rows:
            table.add_row("—", Text("—", style="dim"), Text("—", style="dim"), Text("—", style="dim"))
            return
        for row in rows[:TOP_N]:
            table.add_row(
                row["ticker"],
                pct1(row["weight"]),
                signed_cell(row["contribution"]),
                idiosyncratic_share_cell(row["share"], row["contribution"]),
            )

    def idiosyncratic_return_rows(self):
        """Return idiosyncratic return rows for the selected return horizon."""
        if realized_horizon(self.horizon):
            return realized_window(self.report, self.horizon).get("idiosyncratic_by_name") or []
        return self.report.get("idiosyncratic_return_by_name") or []

    def populate_returns(self):
        if realized_horizon(self.horizon):
            self.populate_realized(self.horizon)
            return
        h = self.horizon
        r = self.report
        recon, factors, tail = self.return_rows(h)
        self.query_one("#returns_summary", Static).update(
            f"[b]{r['horizons'][h]}[/] · {r['horizon_dates'][h]}   [green]your book's actual day[/]\n"
            f"{r['meta']['benchmark']['name']} return {signed(r['benchmark_ret'][h])}"
            f"  +  Active return {signed(r['active_ret'][h])}"
            f"   =   Portfolio {signed(r['portfolio_ret'][h])}\n"
            f"[dim]the table reconciles active return; ranked factor contributors are separate "
            f"({tail} smaller factors folded)[/]"
        )
        self.fill_return_recon(self.query_one("#returns_recon", DataTable), recon)
        self.fill_return_factors(self.query_one("#returns_factors", DataTable), factors)

    def return_rows(self, horizon):
        """Return institutional-style active return attribution rows."""
        r = self.report
        active = r["active_ret"][horizon]
        recon = []
        for name in ["Style", "Sector", "Industry"]:
            value = family_value(r, name, horizon)
            te = family_te_share(r["active_rows"], name)
            if not missing(value) and abs(value) > 1e-9:
                recon.append(
                    return_row(
                        f"{name} factors",
                        "Factor block",
                        value,
                        share_of(value, active),
                        te,
                        "section",
                    )
                )

        factors = [row for row in r["active_rows"] if row.get("ret") and not missing(row["ret"][horizon])]
        factors.sort(key=lambda row: -magnitude(row["ret"][horizon]))
        top = []
        for row in factors[:TOP_N]:
            top.append(
                return_row(
                    row["label"],
                    row["family"],
                    row["ret"][horizon],
                    share_of(row["ret"][horizon], active),
                    row["te_share"],
                    "factor",
                )
            )

        recon += [
            return_row(
                "Idiosyncratic return",
                "Idiosyncratic",
                r["idiosyncratic_return"][horizon],
                share_of(r["idiosyncratic_return"][horizon], active),
                r["idiosyncratic_te_share"],
                "section",
            ),
            return_row("Active return", "Total", active, share_of(active, active), None, "total"),
        ]
        return recon, top, max(0, len(factors) - TOP_N)

    def fill_return_recon(self, table, rows):
        """Render the rows that reconcile active return."""
        table.clear(columns=True)
        table.add_columns("Source", "Contribution", "% Active")
        for row in rows:
            table.add_row(
                return_label_cell(row),
                signed_cell(row["contribution"], bold=row["kind"] == "total"),
                ratio_cell(row["active_share"], bold=row["kind"] == "total"),
            )

    def fill_return_factors(self, table, rows):
        """Render ranked factor details without implying parent-child hierarchy."""
        table.clear(columns=True)
        table.add_columns("Group", "Factor", "Contribution", "% Active", "TE Share")
        for row in rows:
            table.add_row(
                row["family"],
                row["label"],
                signed_cell(row["contribution"]),
                ratio_cell(row["active_share"]),
                te_cell(row["te_share"]),
            )

    def populate_realized(self, key):
        """Render attribution summed over the real daily holdings from --track."""
        r = self.report
        real = realized_window(r, key)
        lookup = {row["factor"]: (row["label"], row["family"]) for row in r["active_rows"]}
        items = sorted(real["factor"].items(), key=lambda kv: -abs(kv[1]))
        tail = max(0, len(items) - TOP_N)
        title = real.get("label") or f"Realized · {real['days']} trading day(s)"
        self.query_one("#returns_summary", Static).update(
            f"[b]{title}[/] · {real['date_range']}   "
            "[green]your actual book, summed day by day[/]\n"
            f"{r['meta']['benchmark']['name']} return {signed(real['benchmark'])}"
            f"  +  Active return {signed(real['active'])}"
            f"   =   Portfolio {signed(real['portfolio'])}\n"
            f"[dim]each day's real holdings — not today's weights run backward; "
            f"the table reconciles active return; ranked factor contributors are separate "
            f"({tail} smaller factors folded)[/]"
        )
        recon = []
        families = realized_family_values(real["factor"], lookup)
        for name, value in families.items():
            te = family_te_share(r["active_rows"], name)
            if abs(value) > 1e-9:
                recon.append(
                    return_row(
                        f"{name} factors",
                        "Factor block",
                        value,
                        share_of(value, real["active"]),
                        te,
                        "section",
                    )
                )
        factors = []
        for factor, value in items[:TOP_N]:
            label, fam = lookup.get(factor, (factor, "—"))
            te = next((row["te_share"] for row in r["active_rows"] if row["factor"] == factor), None)
            factors.append(
                return_row(label, fam, value, share_of(value, real["active"]), te, "factor")
            )
        idiosyncratic = active_residual(real["active"], families, real["idiosyncratic"])
        recon += [
            return_row(
                "Idiosyncratic return",
                "Idiosyncratic",
                idiosyncratic,
                share_of(idiosyncratic, real["active"]),
                r["idiosyncratic_te_share"],
                "section",
            ),
            return_row(
                "Active return",
                "Total",
                real["active"],
                share_of(real["active"], real["active"]),
                None,
                "total",
            ),
        ]
        self.fill_return_recon(self.query_one("#returns_recon", DataTable), recon)
        self.fill_return_factors(self.query_one("#returns_factors", DataTable), factors)

    # ---- text panels ----------------------------------------------------
    def risk_summary(self):
        s = self.report["summary"]
        return (
            f"Common factor {pct1(s['factor_share'])} of total variance · "
            f"idiosyncratic {pct1(s['idiosyncratic_share_of_total_variance'])} · "
            f"active tracking error {pct1(s['tracking_error'])}"
        )

    def idiosyncratic_summary(self):
        n = self.report["idiosyncratic_risk_by_name"]
        eff = "—" if n["effective_names"] is None else f"{n['effective_names']:.1f}"
        top = n["rows"][0]["ticker"] if n["rows"] else "—"
        return (f"Idiosyncratic risk {pct1(n['total_idiosyncratic_risk'])} of the book · top name "
                f"[b]{top}[/] = {pct1(n['top_name_share'])} · effective names {eff}")

    def tail_text(self):
        s = self.report["summary"]
        lines = ["[b]Parametric loss estimate[/] (normal, one-day)"]
        for conf, value in s["var"].items():
            lines.append(f"  {conf}:  total {pct1(value['total_1d'])}   active {pct1(value['active_1d'])}")
        target = "model risk proxy" if self.report["meta"]["benchmark"].get("kind") in {"index", "missing"} else "benchmark"
        lines.append(f"[b]Ex-ante beta[/] to {target}: {beta_value(s['predicted_beta'])}")
        lines += self.track_lines()
        lines.append("[dim]Historical and macro scenarios are not shown because this report "
                     "does not carry an external shock library. They are omitted rather than faked.[/]")
        return "\n".join(lines)

    def track_lines(self):
        """Return track-record lines when a --track folder has accumulated days."""
        track = self.report.get("track")
        if not track or not track["days"]:
            return ["[dim]Pass --track <folder> to store each day's result and build a REAL "
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
            "returns versus stored benchmark returns.",
            "[3] Tracking error is active risk from today's holdings versus the model risk proxy.",
            "[4] TE contribution is variance contribution divided by total tracking error; negative values "
            "are diversifiers in the current covariance matrix.",
            "[5] Idiosyncratic risk is stock-level residual risk after common factors. It is not automatically "
            "stock-picking skill.",
            "[6] Active return contribution is lagged exposure times realized factor return. % Active can exceed "
            "100% when positive and negative factor effects offset.",
            "[7] Multi-day return buttons, when present, sum the holdings actually stored by --track.",
        ])

    # ---- interaction ----------------------------------------------------
    def horizon_buttons(self):
        buttons = [Button(h, id=f"h{i}", variant=("primary" if i == self.horizon else "default"))
                   for i, h in enumerate(self.report["horizons"])]
        for key, window in realized_windows(self.report).items():
            buttons.append(
                Button(
                    window["label"],
                    id=f"hrealized_{key}",
                    variant=("primary" if self.horizon == realized_key(key) else "default"),
                )
            )
        return buttons

    def on_button_pressed(self, event):
        bid = event.button.id
        if not bid or not bid.startswith("h"):
            return
        if bid.startswith("hrealized_"):
            self.horizon = realized_key(bid.removeprefix("hrealized_"))
        else:
            self.horizon = int(bid[1:])
        for i in range(len(self.report["horizons"])):
            self.query_one(f"#h{i}", Button).variant = "primary" if i == self.horizon else "default"
        for key in realized_windows(self.report):
            button = self.query_one(f"#hrealized_{key}", Button)
            button.variant = "primary" if self.horizon == realized_key(key) else "default"
        self.populate_returns()
        self.populate_idiosyncratic_returns()

    async def on_input_submitted(self, event):
        if event.input.id != "chat_input" or not self.chat:
            return
        question = event.value.strip()
        event.input.value = ""
        if not question:
            return
        event.input.disabled = True
        self.start_chat_waiting(event.input)
        await self.append_chat("You", question)
        try:
            answer = await asyncio.to_thread(self.chat.answer, question, self.chat_history)
        except Exception as error:
            detail = str(error) or type(error).__name__  # timeouts stringify empty
            await self.append_chat("OpenFactor", f"Chat error: {detail}")
        else:
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": answer})
            await self.append_chat("OpenFactor", answer)
        finally:
            self.stop_chat_waiting(event.input)

    async def append_chat(self, author, body):
        log = self.query_one("#chat_log", Markdown)
        self.chat_transcript = f"{self.chat_transcript}{chat_fragment(author, body)}"
        await log.update(self.chat_transcript)
        log.scroll_end(animate=False)

    def start_chat_waiting(self, chat_input):
        if self.chat_wait_timer is not None:
            self.chat_wait_timer.stop()
        self.chat_wait_frame = 0
        self.update_chat_waiting(chat_input)
        self.chat_wait_timer = self.set_interval(0.35, lambda: self.update_chat_waiting(chat_input))

    def update_chat_waiting(self, chat_input):
        chat_input.placeholder = CHAT_WAITING_FRAMES[self.chat_wait_frame % len(CHAT_WAITING_FRAMES)]
        self.chat_wait_frame += 1

    def stop_chat_waiting(self, chat_input):
        if self.chat_wait_timer is not None:
            self.chat_wait_timer.stop()
            self.chat_wait_timer = None
        chat_input.placeholder = CHAT_PLACEHOLDER
        chat_input.disabled = False
        chat_input.focus()

    def action_expand_all(self):
        for widget in self.query(Collapsible):
            widget.collapsed = False

    def action_collapse_all(self):
        for widget in self.query(Collapsible):
            widget.collapsed = True


def card(title, value, sub):
    """Return one headline stat card."""
    return Static(f"[dim]{title}[/]\n[b]{value}[/]\n[dim]{sub}[/]", classes="card")


def return_row(label, family, contribution, active_share, te_share, kind):
    """Return one row for the return attribution table."""
    return {
        "label": label,
        "family": family,
        "contribution": contribution,
        "active_share": active_share,
        "te_share": te_share,
        "kind": kind,
    }


def idiosyncratic_weight_label(horizon):
    """Return the weight column label for the selected return horizon."""
    return "Avg Weight" if realized_horizon(horizon) else "Weight"


def realized_windows(report):
    """Return the available multi-day realized return windows."""
    return report.get("realized_windows") or {}


def realized_key(key):
    """Return the app-state key for one realized window."""
    return f"realized:{key}"


def realized_horizon(horizon):
    """Return True when the selected horizon is a realized return window."""
    return isinstance(horizon, str) and horizon.startswith("realized:")


def realized_window(report, horizon):
    """Return the selected realized return window."""
    key = horizon.removeprefix("realized:")
    return realized_windows(report)[key]


def chat_fragment(author, body):
    """Return one Markdown chat turn."""
    if author == "You":
        quote = "\n".join(f"> {line}" if line else ">" for line in str(body).splitlines())
        return f"\n\n**You**\n\n{quote}\n"
    return f"\n\n**{author}**\n\n{str(body).strip()}\n"


def family_value(report, name, horizon):
    """Return one factor-block return contribution."""
    values = report.get("family_ret", {}).get(name)
    if values is None:
        return None
    return values[horizon]


def family_te_share(active_rows, name):
    """Return a factor block's share of tracking error."""
    values = [row["te_share"] for row in active_rows if row["family"] == name and not missing(row["te_share"])]
    return sum(values) if values else None


def realized_family_values(factors, lookup):
    """Return realized factor contributions summed by family."""
    totals = {"Style": 0.0, "Sector": 0.0, "Industry": 0.0}
    for factor, value in factors.items():
        _, family = lookup.get(factor, (factor, None))
        if family in totals and not missing(value):
            totals[family] += value
    return totals


def active_residual(active, families, fallback):
    """Return active return not explained by factor blocks."""
    if missing(active):
        return fallback
    return active - sum(families.values())


def share_of(value, total):
    """Return value divided by total, or None for a zero denominator."""
    if missing(value) or missing(total) or total == 0:
        return None
    return value / total
