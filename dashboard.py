"""
Dash dashboard: score cards, strike profiles, expiry breakdown, overlay.

Layer 1: Headline scores (GEX, VEX, CEX, GEX+)
Layer 2: Strike-level bar charts per Greek
Layer 3: Expiry-bucket stacked bars
Layer 4: Combined overlay
Layer 5 (v1.4): Positioning — OI+Volume calls vs puts for /GC and /CL
"""

import logging
from datetime import date, timedelta
from urllib.parse import parse_qs

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from config import PRODUCTS, EXPIRY_BUCKET_LABELS
from archive import (
    list_archived_products, list_archived_dates,
    load_scores, load_strike_profiles, load_expiry_breakdown,
    load_metadata, get_archive_availability, load_score_history,
    archive_results, load_positioning, list_positioning_dates,
)
from aggregation import interpret_scores
from positioning import current_week_dates, sum_positioning_across_expiries

REGIME_COLORS = {
    "STABILIZING":  "#26a69a",
    "FRAGILE":      "#ffa726",
    "DESTABILIZED": "#ef5350",
    "NEUTRAL":      "#888",
}

REGIME_BLURBS = {
    "STABILIZING":  "Dealer hedging dampens moves",
    "FRAGILE":      "Stabilizing on one channel, destabilizing on the other",
    "DESTABILIZED": "Dealer hedging amplifies moves — crash-prone",
    "NEUTRAL":      "Options market not actively driving spot",
}

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# COLOR SCHEME
# ══════════════════════════════════════════════════════════════

COLORS = {
    "bg": "#0f1117",
    "card": "#1a1d29",
    "card_border": "#2a2d39",
    "text": "#e0e0e0",
    "text_dim": "#888",
    "positive": "#26a69a",
    "negative": "#ef5350",
    "neutral": "#ffa726",
    "accent": "#42a5f5",
    "grid": "#1e2130",
    "bucket_near": "#ef5350",
    "bucket_short": "#ffa726",
    "bucket_medium": "#42a5f5",
    "bucket_long": "#26a69a",
}

BUCKET_COLORS = {
    "near_term": COLORS["bucket_near"],
    "short_term": COLORS["bucket_short"],
    "medium_term": COLORS["bucket_medium"],
    "long_term": COLORS["bucket_long"],
}

PLOT_LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor=COLORS["bg"],
    font=dict(color=COLORS["text"], family="Arial, sans-serif"),
    xaxis=dict(gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"]),
    yaxis=dict(gridcolor=COLORS["grid"], zerolinecolor="#555"),
    margin=dict(l=60, r=30, t=50, b=50),
    hovermode="x unified",
)


# ══════════════════════════════════════════════════════════════
# SCORE CARD COMPONENT
# ══════════════════════════════════════════════════════════════

def make_score_card(title, value, unit, subtitle=""):
    """Create a single score card component."""
    if isinstance(value, (int, float)):
        color = COLORS["positive"] if value > 0 else COLORS["negative"] if value < 0 else COLORS["neutral"]
        formatted = f"${value:,.0f}"
    else:
        color = COLORS["text_dim"]
        formatted = str(value)

    return dbc.Card(
        dbc.CardBody([
            html.Div(title, style={
                "fontSize": "13px", "color": COLORS["text_dim"],
                "textTransform": "uppercase", "letterSpacing": "1px",
                "marginBottom": "8px",
            }),
            html.Div(formatted, style={
                "fontSize": "28px", "fontWeight": "700", "color": color,
                "lineHeight": "1.1",
            }),
            html.Div(unit, style={
                "fontSize": "12px", "color": COLORS["text_dim"],
                "marginTop": "4px",
            }),
            html.Div(subtitle, style={
                "fontSize": "11px", "color": COLORS["text_dim"],
                "marginTop": "6px",
            }) if subtitle else None,
        ]),
        style={
            "backgroundColor": COLORS["card"],
            "border": f"1px solid {COLORS['card_border']}",
            "borderRadius": "8px",
            "minHeight": "130px",
        },
    )


# ══════════════════════════════════════════════════════════════
# CHART BUILDERS
# ══════════════════════════════════════════════════════════════

def build_strike_profile_chart(
    profiles: pd.DataFrame, greek: str, underlying: float, title: str,
    flip_strikes: list = None, greek_label: str = None,
    compare_profiles: pd.DataFrame = None, compare_flips: list = None,
    compare_label: str = None,
) -> go.Figure:
    """Build a bar chart for a single Greek's strike profile."""
    if profiles is None or profiles.empty:
        fig = go.Figure()
        fig.update_layout(**PLOT_LAYOUT, title=title)
        return fig

    values = profiles[greek].values
    colors = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in values]
    label = greek_label or greek.upper()

    fig = go.Figure(data=[
        go.Bar(
            x=profiles["strike"],
            y=values,
            marker_color=colors,
            hovertemplate="Strike: %{x:.0f}<br>Exposure: $%{y:,.0f}<extra></extra>",
            name="Current",
        )
    ])

    # Comparison overlay (semi-transparent)
    if compare_profiles is not None and not compare_profiles.empty and greek in compare_profiles.columns:
        cmp_values = compare_profiles[greek].values
        cmp_colors = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in cmp_values]
        fig.add_trace(go.Bar(
            x=compare_profiles["strike"],
            y=cmp_values,
            marker_color=cmp_colors,
            opacity=0.35,
            name=f"Compare: {compare_label or 'prev'}",
            hovertemplate="Strike: %{x:.0f}<br>Compare: $%{y:,.0f}<extra></extra>",
        ))
        fig.update_layout(barmode="overlay")

    # Add underlying price reference line
    fig.add_vline(
        x=underlying, line_dash="dash", line_color=COLORS["accent"],
        line_width=2, annotation_text=f"Spot: {underlying:.0f}",
        annotation_font_color=COLORS["accent"],
    )

    # Add zero line
    fig.add_hline(y=0, line_color="#555", line_width=1)

    # Flip strike lines (current date, red dashed)
    if flip_strikes:
        for fs in flip_strikes:
            fig.add_vline(
                x=fs, line_dash="dash", line_color=COLORS["negative"],
                line_width=2,
                annotation_text=f"{label} Flip: ${fs:.2f}",
                annotation_position="top",
                annotation_font_color=COLORS["negative"],
            )

    # Compare flip lines (muted)
    if compare_flips:
        for fs in compare_flips:
            fig.add_vline(
                x=fs, line_dash="dot", line_color=COLORS["text_dim"],
                line_width=1,
                annotation_text=f"Cmp Flip: ${fs:.2f}",
                annotation_position="bottom",
                annotation_font_color=COLORS["text_dim"],
            )

    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Strike",
        yaxis_title="$ Exposure",
        showlegend=bool(compare_profiles is not None),
        height=350,
    )

    return fig


def build_expiry_breakdown_chart(breakdown: pd.DataFrame, greek: str, underlying: float, title: str) -> go.Figure:
    """Build a stacked bar chart showing expiry bucket breakdown."""
    if breakdown.empty:
        fig = go.Figure()
        fig.update_layout(**PLOT_LAYOUT, title=title)
        return fig

    fig = go.Figure()

    for bucket in ["near_term", "short_term", "medium_term", "long_term"]:
        bucket_data = breakdown[breakdown["expiry_bucket"] == bucket]
        if bucket_data.empty:
            continue
        fig.add_trace(go.Bar(
            x=bucket_data["strike"],
            y=bucket_data[greek],
            name=EXPIRY_BUCKET_LABELS.get(bucket, bucket),
            marker_color=BUCKET_COLORS.get(bucket, COLORS["accent"]),
            hovertemplate="Strike: %{x:.0f}<br>%{fullData.name}: $%{y:,.0f}<extra></extra>",
        ))

    fig.add_vline(
        x=underlying, line_dash="dash", line_color=COLORS["accent"],
        line_width=2,
    )
    fig.add_hline(y=0, line_color="#555", line_width=1)

    fig.update_layout(
        **PLOT_LAYOUT,
        barmode="relative",
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Strike",
        yaxis_title="$ Exposure",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ),
        height=380,
    )

    return fig


def build_overlay_chart(
    profiles: pd.DataFrame, underlying: float,
    gex_flips: list = None, vex_flips: list = None, cex_flips: list = None,
) -> go.Figure:
    """Build combined overlay of GEX, VEX, CEX on same strike axis."""
    if profiles is None or profiles.empty:
        fig = go.Figure()
        fig.update_layout(**PLOT_LAYOUT, title="Combined Overlay")
        return fig

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # GEX and VEX on primary y-axis (same units: $/pt)
    fig.add_trace(go.Bar(
        x=profiles["strike"], y=profiles["gex"],
        name="GEX ($/pt)", marker_color=COLORS["positive"],
        opacity=0.7,
        hovertemplate="GEX: $%{y:,.0f}<extra></extra>",
    ), secondary_y=False)

    fig.add_trace(go.Bar(
        x=profiles["strike"], y=profiles["vex"],
        name="VEX ($/pt)", marker_color=COLORS["accent"],
        opacity=0.7,
        hovertemplate="VEX: $%{y:,.0f}<extra></extra>",
    ), secondary_y=False)

    # CEX on secondary y-axis (different units: $/day)
    fig.add_trace(go.Scatter(
        x=profiles["strike"], y=profiles["cex"],
        name="CEX ($/day)", mode="lines",
        line=dict(color=COLORS["neutral"], width=2),
        hovertemplate="CEX: $%{y:,.0f}<extra></extra>",
    ), secondary_y=True)

    fig.add_vline(
        x=underlying, line_dash="dash", line_color=COLORS["accent"],
        line_width=2, annotation_text=f"Spot: {underlying:.0f}",
        annotation_font_color=COLORS["accent"],
    )

    # Flip lines: GEX green, VEX blue, CEX amber
    flip_specs = [
        (gex_flips, COLORS["positive"], "GEX"),
        (vex_flips, COLORS["accent"], "VEX"),
        (cex_flips, COLORS["neutral"], "CEX"),
    ]
    for flips, color, label in flip_specs:
        if not flips:
            continue
        for fs in flips:
            fig.add_vline(
                x=fs, line_dash="dash", line_color=color, line_width=2,
                annotation_text=f"{label} Flip: ${fs:.2f}",
                annotation_position="top",
                annotation_font_color=color,
            )

    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text="Combined Greek Exposure Overlay", font=dict(size=16)),
        barmode="group",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=11),
        ),
        height=420,
    )
    fig.update_yaxes(title_text="GEX / VEX ($/pt)", secondary_y=False, gridcolor=COLORS["grid"])
    fig.update_yaxes(title_text="CEX ($/day)", secondary_y=True, gridcolor=COLORS["grid"])

    return fig


# ══════════════════════════════════════════════════════════════
# POSITIONING CHART (v1.4)
# ══════════════════════════════════════════════════════════════

POSITIONING_PRODUCT_OPTIONS = [
    {"label": "/GC", "value": "GC"},
    {"label": "/CL", "value": "CL"},
]


def _positioning_empty_figure(message: str) -> go.Figure:
    """Empty figure with a centered message in the chart area."""
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], family="Arial, sans-serif"),
        margin=dict(l=60, r=30, t=30, b=50),
        height=500,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(
            text=message, xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS["text_dim"]),
        )],
    )
    return fig


def build_positioning_chart(df_sum: pd.DataFrame) -> go.Figure:
    """
    Single Plotly line chart, two traces (Calls green, Puts red).
    Per the v1.4 spec: no subplots, no fills, no markers (unless a single
    point), no zero line, no annotations, standard hover, standard legend.
    """
    if df_sum is None or df_sum.empty:
        return _positioning_empty_figure("No data for selected expiries")

    mode = "markers" if len(df_sum) == 1 else "lines"
    x = df_sum["snapshot_time"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=df_sum["calls"], name="Calls", mode=mode,
        line=dict(color="#26a69a", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=df_sum["puts"], name="Puts", mode=mode,
        line=dict(color="#ef5350", width=2),
    ))

    fig.update_layout(
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(color=COLORS["text"], family="Arial, sans-serif"),
        margin=dict(l=60, r=30, t=30, b=50),
        hovermode="x unified",
        height=500,
        showlegend=True,
        xaxis=dict(
            gridcolor=COLORS["grid"], zerolinecolor=COLORS["grid"],
            tickformat="%H:%M", type="category",
        ),
        yaxis=dict(
            gridcolor=COLORS["grid"], rangemode="tozero",
            title="OI + Volume",
        ),
    )
    return fig


def _format_week_label(monday: date) -> str:
    return f"Week of Mon {monday.isoformat()}"


def _weeks_from_dates(dates: list[date], show_all: bool) -> list[date]:
    """Distinct Mondays of weeks containing any archived date."""
    if not dates:
        return []
    mondays = sorted({d - timedelta(days=d.weekday()) for d in dates}, reverse=True)
    if show_all:
        return mondays
    # Default: only the current week (Mon-Fri block of latest archived date)
    return mondays[:1]


# ══════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════

def create_app() -> dash.Dash:
    """Create and configure the Dash application."""
    import os
    base = os.environ.get("URL_BASE_PATHNAME", "/")
    if not base.startswith("/"):
        base = "/" + base
    if not base.endswith("/"):
        base = base + "/"
    dash_kwargs = dict(
        external_stylesheets=[dbc.themes.DARKLY],
        title="Greek Exposure Engine",
        suppress_callback_exceptions=True,
    )
    if base != "/":
        dash_kwargs["url_base_pathname"] = base
    app = dash.Dash(__name__, **dash_kwargs)

    app.layout = dbc.Container([
        # ── URL for deep-linking (?tab=positioning&date=YYYY-MM-DD) ──
        dcc.Location(id="url", refresh=False),

        # ── Header ──
        dbc.Row([
            dbc.Col([
                html.H2("Greek Exposure Engine", style={
                    "color": COLORS["text"], "marginBottom": "4px", "fontWeight": "600",
                }),
                html.Div("Gamma · Vanna · Charm", style={
                    "color": COLORS["text_dim"], "fontSize": "14px",
                }),
            ], width=4),
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        html.Label("Product", style={"fontSize": "12px", "color": COLORS["text_dim"]}),
                        dcc.Dropdown(
                            id="product-select",
                            options=[{"label": s, "value": s} for s in PRODUCTS.keys()],
                            value="SPY",
                            clearable=False,
                            style={"backgroundColor": COLORS["card"]},
                        ),
                    ], width=3),
                    dbc.Col([
                        html.Label("Date", style={"fontSize": "12px", "color": COLORS["text_dim"]}),
                        dcc.Dropdown(
                            id="date-select",
                            options=[],
                            value=None,
                            clearable=False,
                            placeholder="Select date...",
                            style={"backgroundColor": COLORS["card"]},
                        ),
                    ], width=3),
                    dbc.Col([
                        html.Label("Compare", style={"fontSize": "12px", "color": COLORS["text_dim"]}),
                        dbc.Switch(
                            id="compare-toggle", value=False,
                            style={"marginTop": "6px"},
                        ),
                    ], width=2),
                    dbc.Col([
                        html.Label("Compare Date", style={"fontSize": "12px", "color": COLORS["text_dim"]}),
                        dcc.Dropdown(
                            id="compare-date-select",
                            options=[],
                            value=None,
                            clearable=True,
                            placeholder="—",
                            disabled=True,
                            style={"backgroundColor": COLORS["card"]},
                        ),
                    ], width=2),
                    dbc.Col([
                        html.Label(" ", style={"fontSize": "12px"}),
                        dcc.Loading(
                            id="refresh-loading", type="dot",
                            children=dbc.Button(
                                "Refresh", id="refresh-btn",
                                color="primary", size="sm",
                                style={"width": "100%", "marginTop": "4px"},
                            ),
                        ),
                    ], width=2),
                ]),
            ], width=8),
        ], className="mb-4 mt-3"),

        html.Hr(style={"borderColor": COLORS["card_border"]}),

        # ── Toast for refresh errors ──
        dbc.Toast(
            id="refresh-toast", header="Refresh", icon="danger",
            is_open=False, dismissable=True, duration=8000,
            style={
                "position": "fixed", "top": 70, "right": 20,
                "minWidth": 300, "zIndex": 9999,
            },
        ),

        # ── Regime Banner ──
        html.Div(id="regime-banner", className="mb-3"),

        # ── Score Cards ──
        dbc.Row(id="score-cards", className="mb-4"),

        # ── View Tabs ──
        dbc.Tabs([
            dbc.Tab(label="Strike Profiles", tab_id="tab-profiles", children=[
                dbc.Row([
                    dbc.Col(dcc.Graph(id="gex-profile"), width=12),
                ], className="mt-3"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="vex-profile"), width=6),
                    dbc.Col(dcc.Graph(id="cex-profile"), width=6),
                ]),
            ]),
            dbc.Tab(label="Expiry Breakdown", tab_id="tab-expiry", children=[
                dbc.Row([
                    dbc.Col(dcc.Graph(id="gex-expiry"), width=12),
                ], className="mt-3"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="vex-expiry"), width=6),
                    dbc.Col(dcc.Graph(id="cex-expiry"), width=6),
                ]),
            ]),
            dbc.Tab(label="Combined Overlay", tab_id="tab-overlay", children=[
                dbc.Row([
                    dbc.Col(dcc.Graph(id="overlay-chart"), width=12),
                ], className="mt-3"),
            ]),
            dbc.Tab(label="Time Series", tab_id="tab-timeseries", children=[
                dbc.Row([
                    dbc.Col(dcc.Graph(id="timeseries-scores"), width=12),
                ], className="mt-3"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="timeseries-flips"), width=12),
                ]),
            ]),
            dbc.Tab(label="Positioning", tab_id="tab-positioning", children=[
                dbc.Row([
                    dbc.Col([
                        html.Label("Product", style={"fontSize": "12px", "color": COLORS["text_dim"]}),
                        dcc.RadioItems(
                            id="pos-product",
                            options=POSITIONING_PRODUCT_OPTIONS,
                            value="GC",
                            inline=True,
                            inputStyle={"marginRight": "6px"},
                            labelStyle={"marginRight": "16px", "color": COLORS["text"]},
                        ),
                    ], width=2),
                    dbc.Col([
                        html.Label("Show all history", style={"fontSize": "12px", "color": COLORS["text_dim"]}),
                        dbc.Switch(id="pos-show-all", value=False, style={"marginTop": "4px"}),
                    ], width=2),
                    dbc.Col([
                        html.Label("Week", style={"fontSize": "12px", "color": COLORS["text_dim"]}),
                        dcc.Dropdown(
                            id="pos-week", options=[], value=None, clearable=False,
                            style={"backgroundColor": COLORS["card"]},
                        ),
                    ], width=2),
                    dbc.Col([
                        html.Label("Trading date", style={"fontSize": "12px", "color": COLORS["text_dim"]}),
                        dcc.Dropdown(
                            id="pos-date", options=[], value=None, clearable=False,
                            style={"backgroundColor": COLORS["card"]},
                        ),
                    ], width=2),
                    dbc.Col([
                        html.Label("Expiries", style={"fontSize": "12px", "color": COLORS["text_dim"]}),
                        dcc.Dropdown(
                            id="pos-expiries", options=[], value=[], multi=True,
                            style={"backgroundColor": COLORS["card"]},
                        ),
                    ], width=3),
                    dbc.Col([
                        html.Label(" ", style={"fontSize": "12px"}),
                        dbc.Button(
                            "Select all", id="pos-select-all", color="secondary",
                            size="sm", style={"width": "100%", "marginTop": "4px"},
                        ),
                    ], width=1),
                ], className="mt-3"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="positioning-chart"), width=12),
                ]),
                dcc.Store(id="pos-url-date"),
            ]),
        ], id="view-tabs", active_tab="tab-profiles",
           style={"marginBottom": "20px"}),

        # ── Metadata Footer ──
        html.Div(id="metadata-footer", style={
            "fontSize": "11px", "color": COLORS["text_dim"],
            "textAlign": "center", "padding": "20px 0",
        }),

        # ── Hidden stores ──
        dcc.Store(id="current-data"),
        dcc.Store(id="refresh-tick", data=0),

    ], fluid=True, style={
        "backgroundColor": COLORS["bg"],
        "minHeight": "100vh",
        "padding": "0 20px",
    })

    # ══════════════════════════════════════════════════════════
    # CALLBACKS
    # ══════════════════════════════════════════════════════════

    @app.callback(
        Output("date-select", "options"),
        Output("date-select", "value"),
        Output("compare-date-select", "options"),
        Input("product-select", "value"),
        Input("refresh-tick", "data"),
    )
    def update_dates(product, _tick):
        """Populate date dropdowns from archive."""
        if not product:
            return [], None, []
        dates = list_archived_dates(product)
        options = [{"label": d.isoformat(), "value": d.isoformat()} for d in reversed(dates)]
        value = options[0]["value"] if options else None
        return options, value, options

    @app.callback(
        Output("compare-date-select", "disabled"),
        Output("compare-date-select", "value"),
        Input("compare-toggle", "value"),
    )
    def toggle_compare(enabled):
        return (not enabled), (None if not enabled else dash.no_update)

    @app.callback(
        Output("refresh-tick", "data"),
        Output("refresh-toast", "is_open"),
        Output("refresh-toast", "children"),
        Output("refresh-toast", "icon"),
        Input("refresh-btn", "n_clicks"),
        State("product-select", "value"),
        State("refresh-tick", "data"),
        prevent_initial_call=True,
    )
    def trigger_refresh(n_clicks, product, tick):
        """Run a live pipeline for the currently selected product."""
        if not n_clicks or not product:
            return dash.no_update, False, "", "danger"
        try:
            from pipeline import run_pipeline
            res = run_pipeline(symbol=product, snapshot_date=date.today())
            archive_results(res)
            msg = (
                f"Refreshed {product} @ ${res['underlying_price']:.2f} — "
                f"GEX ${res['scores']['gex']:,.0f}/pt"
            )
            return (tick or 0) + 1, True, msg, "success"
        except Exception as e:
            logger.error(f"Live refresh failed: {e}")
            return dash.no_update, True, f"Refresh failed: {type(e).__name__}: {e}", "danger"

    @app.callback(
        Output("current-data", "data"),
        Input("date-select", "value"),
        Input("compare-date-select", "value"),
        Input("compare-toggle", "value"),
        Input("refresh-tick", "data"),
        State("product-select", "value"),
        prevent_initial_call=False,
    )
    def load_data(date_str, compare_date_str, compare_on, _tick, product):
        """Load archived data for selected product/date (and compare date)."""
        if not product or not date_str:
            return None

        try:
            snapshot_date = date.fromisoformat(date_str)
            avail = get_archive_availability(product, snapshot_date)

            if not avail["tier2_scores"]:
                return None

            scores = load_scores(product, snapshot_date)
            meta = load_metadata(product, snapshot_date)

            data = {
                "product": product,
                "date": date_str,
                "scores": scores,
                "metadata": meta,
                "underlying_price": meta.get("underlying_price", 0),
            }

            if compare_on and compare_date_str and compare_date_str != date_str:
                try:
                    cmp_date = date.fromisoformat(compare_date_str)
                    cmp_scores = load_scores(product, cmp_date)
                    cmp_meta = load_metadata(product, cmp_date)
                    data["compare"] = {
                        "date": compare_date_str,
                        "scores": cmp_scores,
                        "metadata": cmp_meta,
                        "underlying_price": cmp_meta.get("underlying_price", 0),
                    }
                except Exception as ce:
                    logger.warning(f"Compare load failed: {ce}")

            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

    @app.callback(
        Output("regime-banner", "children"),
        Output("score-cards", "children"),
        Input("current-data", "data"),
    )
    def update_score_cards(data):
        """Render the regime banner and headline score cards."""
        if not data or not data.get("scores"):
            empty_banner = html.Div(
                "No data loaded",
                style={
                    "padding": "10px 16px", "borderRadius": "6px",
                    "backgroundColor": COLORS["card"],
                    "color": COLORS["text_dim"],
                    "border": f"1px solid {COLORS['card_border']}",
                    "fontSize": "13px",
                },
            )
            cards = [dbc.Col(make_score_card(name, "—", "", "No data loaded"), width=3)
                     for name in ["GEX", "VEX", "CEX", "GEX+"]]
            return empty_banner, cards

        s = data["scores"]
        cmp = (data.get("compare") or {}).get("scores")
        labels = interpret_scores(s)

        def fmt_flip(flips):
            if not flips:
                return ""
            if len(flips) == 1:
                return f"Flip at ${flips[0]:.2f}"
            return "Flips: " + ", ".join(f"${f:.0f}" for f in flips[:3])

        def delta_line(cur, key):
            if not cmp:
                return ""
            d = cur - cmp.get(key, 0)
            sign = "+" if d >= 0 else ""
            return f"Δ {sign}${d:,.0f}"

        # Subtitle = regime-aware label · flip strike · Δ vs compare date
        def subtitle(label, flips_key, score_key):
            return " · ".join([x for x in [
                label,
                fmt_flip(s.get(flips_key)),
                delta_line(s[score_key], score_key),
            ] if x])

        gex_sub = subtitle(labels["gex"], "gex_flip", "gex")
        vex_sub = subtitle(labels["vex"], "vex_flip", "vex")
        cex_sub = subtitle(labels["cex"], "cex_flip", "cex")
        gpl_sub = " · ".join([x for x in [
            labels["gex_plus"], delta_line(s["gex_plus"], "gex_plus"),
        ] if x])

        regime = labels["regime"]
        banner = html.Div(
            [
                html.Span(regime, style={
                    "fontSize": "20px", "fontWeight": "700",
                    "letterSpacing": "2px", "marginRight": "16px",
                }),
                html.Span(REGIME_BLURBS.get(regime, ""), style={
                    "fontSize": "13px", "opacity": 0.85,
                }),
            ],
            style={
                "padding": "12px 18px",
                "borderRadius": "6px",
                "backgroundColor": REGIME_COLORS.get(regime, COLORS["text_dim"]),
                "color": "#0f1117" if regime != "DESTABILIZED" else "#fff",
                "border": "none",
                "textAlign": "center",
            },
        )

        cards = [
            dbc.Col(make_score_card("GEX", s["gex"], "$ / point", gex_sub), width=3),
            dbc.Col(make_score_card("VEX", s["vex"], "$ / point", vex_sub), width=3),
            dbc.Col(make_score_card("CEX", s["cex"], "$ / day", cex_sub), width=3),
            dbc.Col(make_score_card("GEX+", s["gex_plus"], "$ / point", gpl_sub), width=3),
        ]
        return banner, cards

    @app.callback(
        Output("gex-profile", "figure"),
        Output("vex-profile", "figure"),
        Output("cex-profile", "figure"),
        Input("current-data", "data"),
    )
    def update_profiles(data):
        """Update strike profile charts."""
        empty = go.Figure()
        empty.update_layout(**PLOT_LAYOUT, height=350)

        if not data:
            return empty, empty, empty

        try:
            product = data["product"]
            snapshot_date = date.fromisoformat(data["date"])
            underlying = data["underlying_price"]
            profiles = load_strike_profiles(product, snapshot_date)
            scores = data.get("scores") or {}

            cmp = data.get("compare")
            cmp_profiles = None
            cmp_label = None
            cmp_gex = cmp_vex = cmp_cex = None
            if cmp:
                try:
                    cmp_profiles = load_strike_profiles(product, date.fromisoformat(cmp["date"]))
                    cmp_label = cmp["date"]
                    cmp_scores = cmp.get("scores") or {}
                    cmp_gex = cmp_scores.get("gex_flip") or []
                    cmp_vex = cmp_scores.get("vex_flip") or []
                    cmp_cex = cmp_scores.get("cex_flip") or []
                except Exception as ce:
                    logger.warning(f"Compare load failed: {ce}")
                    cmp_profiles = None

            gex_fig = build_strike_profile_chart(
                profiles, "gex", underlying, "GEX Strike Profile ($/pt)",
                flip_strikes=scores.get("gex_flip"), greek_label="GEX",
                compare_profiles=cmp_profiles, compare_flips=cmp_gex,
                compare_label=cmp_label,
            )
            vex_fig = build_strike_profile_chart(
                profiles, "vex", underlying, "VEX Strike Profile ($/pt)",
                flip_strikes=scores.get("vex_flip"), greek_label="VEX",
                compare_profiles=cmp_profiles, compare_flips=cmp_vex,
                compare_label=cmp_label,
            )
            cex_fig = build_strike_profile_chart(
                profiles, "cex", underlying, "CEX Strike Profile ($/day)",
                flip_strikes=scores.get("cex_flip"), greek_label="CEX",
                compare_profiles=cmp_profiles, compare_flips=cmp_cex,
                compare_label=cmp_label,
            )

            return gex_fig, vex_fig, cex_fig
        except Exception as e:
            logger.error(f"Error building profiles: {e}")
            return empty, empty, empty

    @app.callback(
        Output("gex-expiry", "figure"),
        Output("vex-expiry", "figure"),
        Output("cex-expiry", "figure"),
        Input("current-data", "data"),
    )
    def update_expiry(data):
        """Update expiry breakdown charts."""
        empty = go.Figure()
        empty.update_layout(**PLOT_LAYOUT, height=380)

        if not data:
            return empty, empty, empty

        try:
            product = data["product"]
            snapshot_date = date.fromisoformat(data["date"])
            underlying = data["underlying_price"]
            breakdown = load_expiry_breakdown(product, snapshot_date)

            gex_fig = build_expiry_breakdown_chart(breakdown, "gex", underlying, "GEX by Expiry Bucket")
            vex_fig = build_expiry_breakdown_chart(breakdown, "vex", underlying, "VEX by Expiry Bucket")
            cex_fig = build_expiry_breakdown_chart(breakdown, "cex", underlying, "CEX by Expiry Bucket")

            return gex_fig, vex_fig, cex_fig
        except Exception as e:
            logger.error(f"Error building expiry charts: {e}")
            return empty, empty, empty

    @app.callback(
        Output("overlay-chart", "figure"),
        Input("current-data", "data"),
    )
    def update_overlay(data):
        """Update combined overlay chart."""
        empty = go.Figure()
        empty.update_layout(**PLOT_LAYOUT, height=420)

        if not data:
            return empty

        try:
            product = data["product"]
            snapshot_date = date.fromisoformat(data["date"])
            underlying = data["underlying_price"]
            profiles = load_strike_profiles(product, snapshot_date)
            scores = data.get("scores") or {}
            return build_overlay_chart(
                profiles, underlying,
                gex_flips=scores.get("gex_flip"),
                vex_flips=scores.get("vex_flip"),
                cex_flips=scores.get("cex_flip"),
            )
        except Exception as e:
            logger.error(f"Error building overlay: {e}")
            return empty

    @app.callback(
        Output("timeseries-scores", "figure"),
        Output("timeseries-flips", "figure"),
        Input("product-select", "value"),
        Input("refresh-tick", "data"),
    )
    def update_timeseries(product, _tick):
        """Render historical score and flip-strike line charts."""
        empty = go.Figure()
        empty.update_layout(**PLOT_LAYOUT, height=380)
        if not product:
            return empty, empty

        try:
            history = load_score_history(product)
        except Exception as e:
            logger.error(f"Score history load failed: {e}")
            return empty, empty

        if history.empty:
            empty.update_layout(title="No archived snapshots yet")
            return empty, empty

        x = pd.to_datetime(history["date"])

        # ── Scores chart ──
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # GEX+ background bands (green when > 0, red when < 0)
        for i in range(len(history) - 1):
            x0, x1 = x.iloc[i], x.iloc[i + 1]
            v = history["gex_plus"].iloc[i]
            fig.add_vrect(
                x0=x0, x1=x1,
                fillcolor=COLORS["positive"] if v > 0 else COLORS["negative"],
                opacity=0.06, layer="below", line_width=0,
            )

        fig.add_trace(go.Scatter(
            x=x, y=history["gex"], name="GEX ($/pt)", mode="lines+markers",
            line=dict(color=COLORS["positive"], width=2),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=x, y=history["vex"], name="VEX ($/pt)", mode="lines+markers",
            line=dict(color=COLORS["accent"], width=2),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=x, y=history["gex_plus"], name="GEX+ ($/pt)", mode="lines",
            line=dict(color=COLORS["text"], width=2, dash="dot"),
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=x, y=history["cex"], name="CEX ($/day)", mode="lines+markers",
            line=dict(color=COLORS["neutral"], width=1.5),
        ), secondary_y=True)
        if history["underlying_price"].notna().any():
            fig.add_trace(go.Scatter(
                x=x, y=history["underlying_price"],
                name="Underlying", mode="lines",
                line=dict(color=COLORS["text_dim"], width=1, dash="dash"),
                hovertemplate="Underlying: %{y:.2f}<extra></extra>",
            ), secondary_y=True)

        fig.update_layout(
            **PLOT_LAYOUT,
            title=dict(text=f"{product} — Score History", font=dict(size=16)),
            xaxis=dict(rangeslider=dict(visible=True), type="date",
                        gridcolor=COLORS["grid"]),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, font=dict(size=11),
            ),
            height=420,
        )
        fig.update_yaxes(title_text="GEX / VEX / GEX+ ($/pt)", secondary_y=False, gridcolor=COLORS["grid"])
        fig.update_yaxes(title_text="CEX / Underlying", secondary_y=True, gridcolor=COLORS["grid"])

        # ── Flip strike chart ──
        flips_fig = go.Figure()

        def first_flip(lst):
            if isinstance(lst, list) and len(lst) > 0:
                return lst[0]
            return None

        gex_first = history["gex_flip"].apply(first_flip)
        vex_first = history["vex_flip"].apply(first_flip)
        cex_first = history["cex_flip"].apply(first_flip)

        if history["underlying_price"].notna().any():
            flips_fig.add_trace(go.Scatter(
                x=x, y=history["underlying_price"], name="Underlying",
                mode="lines", line=dict(color=COLORS["text_dim"], width=1.5),
            ))
        flips_fig.add_trace(go.Scatter(
            x=x, y=gex_first, name="GEX flip", mode="lines+markers",
            line=dict(color=COLORS["positive"], width=2),
        ))
        flips_fig.add_trace(go.Scatter(
            x=x, y=vex_first, name="VEX flip", mode="lines+markers",
            line=dict(color=COLORS["accent"], width=2),
        ))
        flips_fig.add_trace(go.Scatter(
            x=x, y=cex_first, name="CEX flip", mode="lines+markers",
            line=dict(color=COLORS["neutral"], width=2),
        ))
        flips_fig.update_layout(
            **PLOT_LAYOUT,
            title=dict(text=f"{product} — Flip Strikes vs. Underlying", font=dict(size=14)),
            xaxis=dict(type="date", gridcolor=COLORS["grid"]),
            yaxis=dict(title="Strike / Price", gridcolor=COLORS["grid"]),
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                xanchor="right", x=1, font=dict(size=11),
            ),
            height=300,
        )

        return fig, flips_fig

    # ══════════════════════════════════════════════════════════
    # POSITIONING TAB CALLBACKS (v1.4)
    # ══════════════════════════════════════════════════════════

    @app.callback(
        Output("view-tabs", "active_tab"),
        Output("pos-url-date", "data"),
        Input("url", "search"),
    )
    def parse_url(search):
        """
        Read ?tab=positioning&date=YYYY-MM-DD on initial load.
        URL date is a one-shot hint; user clicks afterwards take over.
        """
        if not search:
            return dash.no_update, None
        qs = parse_qs(search.lstrip("?"))
        active = dash.no_update
        if qs.get("tab", [None])[0] == "positioning":
            active = "tab-positioning"
        url_date = None
        d = qs.get("date", [None])[0]
        if d:
            try:
                url_date = date.fromisoformat(d).isoformat()
            except ValueError:
                url_date = None
        return active, url_date

    @app.callback(
        Output("pos-week", "options"),
        Output("pos-week", "value"),
        Input("pos-product", "value"),
        Input("pos-show-all", "value"),
        State("pos-url-date", "data"),
    )
    def update_pos_weeks(product, show_all, url_date):
        if not product:
            return [], None
        dates = list_positioning_dates(product)
        if not dates:
            return [], None
        mondays = _weeks_from_dates(dates, bool(show_all))
        options = [{"label": _format_week_label(m), "value": m.isoformat()} for m in mondays]

        chosen = mondays[0].isoformat() if mondays else None
        if url_date:
            try:
                d = date.fromisoformat(url_date)
                if d in dates:
                    monday = (d - timedelta(days=d.weekday())).isoformat()
                    if any(opt["value"] == monday for opt in options):
                        chosen = monday
            except ValueError:
                pass
        return options, chosen

    @app.callback(
        Output("pos-date", "options"),
        Output("pos-date", "value"),
        Input("pos-product", "value"),
        Input("pos-week", "value"),
        State("pos-url-date", "data"),
    )
    def update_pos_dates(product, week, url_date):
        if not product or not week:
            return [], None
        try:
            monday = date.fromisoformat(week)
        except ValueError:
            return [], None
        archived = set(list_positioning_dates(product))
        week_days = [monday + timedelta(days=i) for i in range(5)]
        options = []
        for d in week_days:
            has_data = d in archived
            options.append({
                "label": f"{d.strftime('%a')} {d.isoformat()}" + ("" if has_data else "  (no data)"),
                "value": d.isoformat(),
                "disabled": not has_data,
            })
        in_week = [d for d in week_days if d in archived]
        chosen = in_week[-1].isoformat() if in_week else None
        if url_date:
            try:
                d = date.fromisoformat(url_date)
                if d in archived and monday <= d < monday + timedelta(days=7):
                    chosen = d.isoformat()
            except ValueError:
                pass
        return options, chosen

    @app.callback(
        Output("pos-expiries", "options"),
        Output("pos-expiries", "value"),
        Input("pos-product", "value"),
        Input("pos-date", "value"),
    )
    def update_pos_expiries(product, date_str):
        if not product or not date_str:
            return [], []
        try:
            snapshot_date = date.fromisoformat(date_str)
        except ValueError:
            return [], []
        df = load_positioning(product, snapshot_date)
        if df.empty:
            return [], []
        expiries = sorted({pd.Timestamp(d).date() for d in df["expiry"]})
        options = [
            {"label": f"{e.strftime('%a')} {e.isoformat()}", "value": e.isoformat()}
            for e in expiries
        ]
        # Default: nearest expiry on or after the trading date (else the earliest).
        forward = [e for e in expiries if e >= snapshot_date]
        nearest = forward[0] if forward else expiries[0]
        return options, [nearest.isoformat()]

    @app.callback(
        Output("pos-expiries", "value", allow_duplicate=True),
        Input("pos-select-all", "n_clicks"),
        State("pos-expiries", "options"),
        prevent_initial_call=True,
    )
    def pos_select_all(n_clicks, options):
        if not n_clicks or not options:
            return dash.no_update
        return [opt["value"] for opt in options if not opt.get("disabled")]

    @app.callback(
        Output("positioning-chart", "figure"),
        Input("pos-product", "value"),
        Input("pos-date", "value"),
        Input("pos-expiries", "value"),
    )
    def update_pos_chart(product, date_str, selected):
        if not product or not date_str:
            return _positioning_empty_figure("No positioning data available for this date")
        try:
            snapshot_date = date.fromisoformat(date_str)
        except ValueError:
            return _positioning_empty_figure("No positioning data available for this date")

        df = load_positioning(product, snapshot_date)
        if df.empty:
            return _positioning_empty_figure("No positioning data available for this date")
        if not selected:
            return _positioning_empty_figure("Select at least one expiry")

        try:
            selected_dates = [date.fromisoformat(s) for s in selected]
        except (TypeError, ValueError):
            return _positioning_empty_figure("Select at least one expiry")

        df_sum = sum_positioning_across_expiries(df, selected_dates)
        if df_sum.empty:
            return _positioning_empty_figure("No data for selected expiries")
        return build_positioning_chart(df_sum)

    @app.callback(
        Output("metadata-footer", "children"),
        Input("current-data", "data"),
    )
    def update_footer(data):
        """Update metadata footer."""
        if not data or not data.get("metadata"):
            return "No data loaded. Run the pipeline to generate snapshots."

        m = data["metadata"]
        parts = [
            f"Engine v{m.get('engine_version', '?')}",
            f"r = {m.get('risk_free_rate', 0):.2%}",
            f"q = {m.get('dividend_yield', 0):.3f}" if m.get('dividend_yield') is not None else None,
            f"Vol multiplier = {m.get('vol_spot_multiplier', '?')}x",
            f"Contracts: {m.get('iv_log', {}).get('converged', '?')}",
            f"IV failures: {m.get('iv_log', {}).get('failed', '?')}",
        ]
        return " · ".join([p for p in parts if p])

    return app


# ══════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    app = create_app()
    app.run(debug=True, host="0.0.0.0", port=8050)
