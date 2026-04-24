"""
Dash dashboard: score cards, strike profiles, expiry breakdown, overlay.

Layer 1: Headline scores (GEX, VEX, CEX, GEX+)
Layer 2: Strike-level bar charts per Greek
Layer 3: Expiry-bucket stacked bars
Layer 4: Combined overlay
"""

import logging
from datetime import date

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
    load_metadata, get_archive_availability,
)

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

def build_strike_profile_chart(profiles: pd.DataFrame, greek: str, underlying: float, title: str) -> go.Figure:
    """Build a bar chart for a single Greek's strike profile."""
    if profiles.empty:
        fig = go.Figure()
        fig.update_layout(**PLOT_LAYOUT, title=title)
        return fig

    values = profiles[greek].values
    colors = [COLORS["positive"] if v > 0 else COLORS["negative"] for v in values]

    fig = go.Figure(data=[
        go.Bar(
            x=profiles["strike"],
            y=values,
            marker_color=colors,
            hovertemplate="Strike: %{x:.0f}<br>Exposure: $%{y:,.0f}<extra></extra>",
        )
    ])

    # Add underlying price reference line
    fig.add_vline(
        x=underlying, line_dash="dash", line_color=COLORS["accent"],
        line_width=2, annotation_text=f"Spot: {underlying:.0f}",
        annotation_font_color=COLORS["accent"],
    )

    # Add zero line
    fig.add_hline(y=0, line_color="#555", line_width=1)

    fig.update_layout(
        **PLOT_LAYOUT,
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Strike",
        yaxis_title="$ Exposure",
        showlegend=False,
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


def build_overlay_chart(profiles: pd.DataFrame, underlying: float) -> go.Figure:
    """Build combined overlay of GEX, VEX, CEX on same strike axis."""
    if profiles.empty:
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
# APP LAYOUT
# ══════════════════════════════════════════════════════════════

def create_app() -> dash.Dash:
    """Create and configure the Dash application."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title="Greek Exposure Engine",
        suppress_callback_exceptions=True,
    )

    app.layout = dbc.Container([
        # ── Header ──
        dbc.Row([
            dbc.Col([
                html.H2("Greek Exposure Engine", style={
                    "color": COLORS["text"], "marginBottom": "4px", "fontWeight": "600",
                }),
                html.Div("Gamma · Vanna · Charm", style={
                    "color": COLORS["text_dim"], "fontSize": "14px",
                }),
            ], width=6),
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
                    ], width=4),
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
                    ], width=5),
                    dbc.Col([
                        html.Label(" ", style={"fontSize": "12px"}),
                        dbc.Button(
                            "Refresh", id="refresh-btn",
                            color="primary", size="sm",
                            style={"width": "100%", "marginTop": "4px"},
                        ),
                    ], width=3),
                ]),
            ], width=6),
        ], className="mb-4 mt-3"),

        html.Hr(style={"borderColor": COLORS["card_border"]}),

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
        ], id="view-tabs", active_tab="tab-profiles",
           style={"marginBottom": "20px"}),

        # ── Metadata Footer ──
        html.Div(id="metadata-footer", style={
            "fontSize": "11px", "color": COLORS["text_dim"],
            "textAlign": "center", "padding": "20px 0",
        }),

        # ── Hidden store for current data ──
        dcc.Store(id="current-data"),

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
        Input("product-select", "value"),
    )
    def update_dates(product):
        """Populate date dropdown from archive."""
        if not product:
            return [], None
        dates = list_archived_dates(product)
        options = [{"label": d.isoformat(), "value": d.isoformat()} for d in reversed(dates)]
        value = options[0]["value"] if options else None
        return options, value

    @app.callback(
        Output("current-data", "data"),
        Input("date-select", "value"),
        Input("refresh-btn", "n_clicks"),
        State("product-select", "value"),
        prevent_initial_call=False,
    )
    def load_data(date_str, n_clicks, product):
        """Load archived data for selected product/date."""
        if not product or not date_str:
            return None

        try:
            snapshot_date = date.fromisoformat(date_str)
            avail = get_archive_availability(product, snapshot_date)

            if not avail["tier2_scores"]:
                return None

            scores = load_scores(product, snapshot_date)
            meta = load_metadata(product, snapshot_date)

            return {
                "product": product,
                "date": date_str,
                "scores": scores,
                "metadata": meta,
                "underlying_price": meta.get("underlying_price", 0),
            }
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None

    @app.callback(
        Output("score-cards", "children"),
        Input("current-data", "data"),
    )
    def update_score_cards(data):
        """Update the headline score cards."""
        if not data or not data.get("scores"):
            return [dbc.Col(make_score_card(name, "—", "", "No data loaded"), width=3)
                    for name in ["GEX", "VEX", "CEX", "GEX+"]]

        s = data["scores"]
        return [
            dbc.Col(make_score_card("GEX", s["gex"], "$ / point",
                    "Stabilizing" if s["gex"] > 0 else "Destabilizing"), width=3),
            dbc.Col(make_score_card("VEX", s["vex"], "$ / point",
                    "Stabilizing" if s["vex"] > 0 else "Crash risk"), width=3),
            dbc.Col(make_score_card("CEX", s["cex"], "$ / day",
                    "Net selling" if s["cex"] > 0 else "Net buying"), width=3),
            dbc.Col(make_score_card("GEX+", s["gex_plus"], "$ / point",
                    "Combined liquidity"), width=3),
        ]

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

            gex_fig = build_strike_profile_chart(profiles, "gex", underlying, "GEX Strike Profile ($/pt)")
            vex_fig = build_strike_profile_chart(profiles, "vex", underlying, "VEX Strike Profile ($/pt)")
            cex_fig = build_strike_profile_chart(profiles, "cex", underlying, "CEX Strike Profile ($/day)")

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
            return build_overlay_chart(profiles, underlying)
        except Exception as e:
            logger.error(f"Error building overlay: {e}")
            return empty

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
