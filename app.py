import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# Page config
st.set_page_config(
    page_title="Resistivity Measurement Lab",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for scientific look
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .metric-card-dark {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .info-box {
        background: linear-gradient(135deg, #e8f4f8 0%, #d5e8ed 100%);
        border-left: 4px solid #3498db;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .formula-box {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
        font-family: 'Times New Roman', serif;
        font-size: 1.1rem;
    }
    .probe-diagram {
        background: #fafafa;
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 1.5rem;
        text-align: center;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
    }
</style>
""", unsafe_allow_html=True)


def compute_resistivity_two_probe(V, I, L, A, R_contact=0, R_wires=0):
    """Calculate resistivity using two-probe method"""
    if I == 0:
        return None, None, None

    R_measured = V / I
    R_sample = R_measured - (2 * R_contact + R_wires)

    if L == 0 or A == 0:
        return None, None, None

    rho = R_sample * (A / L)

    return R_measured, R_sample, rho


def compute_resistivity_four_probe(V, I, s, t=None, is_thin_film=False):
    """Calculate resistivity using four-probe method"""
    if I == 0:
        return None, None

    if is_thin_film and t:
        # Thin film formula
        rho = (math.pi / math.log(2)) * t * (V / I)
    else:
        # Bulk formula
        rho = 2 * math.pi * s * (V / I)

    return V / I, rho


def linear_fit(x, y):
    """Perform linear regression and return slope, intercept, R²"""
    if len(x) < 2:
        return 1, 0, 1.0

    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    sum_x2 = sum(xi**2 for xi in x)

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2) if (n * sum_x2 - sum_x**2) != 0 else 1
    intercept = (sum_y - slope * sum_x) / n

    # Calculate R²
    y_mean = sum_y / n
    ss_res = sum((yi - (slope * xi + intercept))**2 for xi, yi in zip(x, y))
    ss_tot = sum((yi - y_mean)**2 for yi in y)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0

    return slope, intercept, r_squared


def plot_v_i_graph(voltages, currents, method):
    """Create interactive V-I plot with linear fit"""
    if len(voltages) < 2:
        return None

    slope, intercept, r_squared = linear_fit(currents, voltages)

    # Sort for line plotting
    sorted_pairs = sorted(zip(currents, voltages))
    sorted_currents = [x[0] for x in sorted_pairs]
    sorted_voltages = [x[1] for x in sorted_pairs]

    # Fit line
    fit_currents = np.linspace(min(currents) * 0.9, max(currents) * 1.1, 100)
    fit_voltages = slope * fit_currents + intercept

    fig = go.Figure()

    # Data points
    fig.add_trace(go.Scatter(
        x=list(currents),
        y=list(voltages),
        mode='markers',
        marker=dict(size=12, color='#3498db', symbol='circle',
                     line=dict(width=2, color='white')),
        name='Data Points',
        text=[f'I={i:.4f}A, V={v:.4f}V' for i, v in zip(currents, voltages)],
        hoverinfo='text'
    ))

    # Linear fit
    fig.add_trace(go.Scatter(
        x=fit_currents,
        y=fit_voltages,
        mode='lines',
        line=dict(color='#e74c3c', width=2, dash='dash'),
        name=f'Linear Fit (R²={r_squared:.4f})'
    ))

    fig.update_layout(
        title=dict(
            text=f'V-I Characteristic - {method} Method',
            font=dict(size=16, color='#2c3e50')
        ),
        xaxis_title='Current (A)',
        yaxis_title='Voltage (V)',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            gridcolor='#ecf0f1',
            zeroline=True,
            zerolinecolor='#bdc3c7'
        ),
        yaxis=dict(
            gridcolor='#ecf0f1',
            zeroline=True,
            zerolinecolor='#bdc3c7'
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=400
    )

    return fig


def draw_two_probe_diagram():
    """Draw two-probe measurement diagram"""
    fig = go.Figure()

    # Sample bar
    fig.add_shape(type="rect", x0=1, y0=2, x1=9, y1=3,
                  fillcolor='#bdc3c7', line=dict(color='#7f8c8d', width=2),
                  name="Sample")

    # Probe 1 (left)
    fig.add_shape(type="rect", x0=0.3, y0=2.2, x1=1.1, y1=2.8,
                  fillcolor='#3498db', line=dict(color='#2980b9', width=2),
                  name="Probe 1")

    # Probe 2 (right)
fig.add_shape(type="rect", x0=8.9, y0=2.2, x1=9.7, y1=2.8)
fillcolor='#3498db', line=dict(color='#2980b9', width=2),
                  name="Probe 2")

    # Current arrows
    fig.add_annotation(x=0.7, y=3.5, text="I", font=dict(size=16, color='#e74c3c'),
                      showarrow=True, arrowhead=2, ax=0, ay=-20)
    fig.add_annotation(x=9.3, y=3.5, text="I", font=dict(size=16, color='#e74c3c'),
                      showarrow=True, arrowhead=2, ax=0, ay=-20)

    # Voltage labels
    fig.add_annotation(x=0.7, y=1.5, text="V", font=dict(size=14, color='#9b59b6'),
                      showarrow=True, arrowhead=2, ax=0, ay=20)
    fig.add_annotation(x=9.3, y=1.5, text="V", font=dict(size=14, color='#9b59b6'),
                      showarrow=True, arrowhead=2, ax=0, ay=20)

    # Labels
    fig.add_annotation(x=5, y=4.2, text="CURRENT + VOLTAGE",
                      font=dict(size=14, color='#2c3e50', family='Arial'),
                      showarrow=False)
    fig.add_annotation(x=5, y=1.0, text="Contact Points",
                      font=dict(size=12, color='#7f8c8d'), showarrow=False)

    fig.update_layout(
        showlegend=False,
        xaxis=dict(range=[-0.5, 10.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, 5], showgrid=False, zeroline=False, showticklabels=False),
        height=300,
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=40, b=20)
    )

    return fig


def draw_four_probe_diagram():
    """Draw four-probe measurement diagram"""
    fig = go.Figure()

    # Sample bar
    fig.add_shape(type="rect", x0=0.5, y0=2, x1=9.5, y1=3,
                  fillcolor='#bdc3c7', line=dict(color='#7f8c8d', width=2),
                  name="Sample")

    # Probe 1 (outer left - current)
    fig.add_shape(type="rect", x0=0.1, y0=2.2, x1=0.9, y1=2.8,
                  fillcolor='#e74c3c', line=dict(color='#c0392b', width=2))

    # Probe 2 (inner left - voltage)
    fig.add_shape(type="rect", x0=2.3, y0=2.2, x1=3.1, y1=2.8,
                  fillcolor='#3498db', line=dict(color='#2980b9', width=2))

    # Probe 3 (inner right - voltage)
    fig.add_shape(type="rect", x0=6.9, y0=2.2, x1=7.7, y1=2.8,
                  fillcolor='#3498db', line=dict(color='#2980b9', width=2))

    # Probe 4 (outer right - current)
    fig.add_shape(type="rect", x0=9.1, y0=2.2, x1=9.9, y0=2.8,
                  fillcolor='#e74c3c', line=dict(color='#c0392b', width=2))

    # Current flow lines
    current_x = [0.5, 2.5, 7.5, 9.5]
    for cx in current_x:
        fig.add_annotation(
            x=cx, y=3.5, text="→",
            font=dict(size=20, color='#e74c3c'),
            showarrow=False
        )

    # Voltage measurement bracket
    fig.add_shape(type="line", x0=2.7, y0=1.3, x1=7.3, y1=1.3,
                  line=dict(color='#9b59b6', width=2))
    fig.add_annotation(x=5, y=0.8, text="V (Voltage Measurement)",
                      font=dict(size=12, color='#9b59b6'), showarrow=False)

    # Current injection
    fig.add_annotation(x=0.5, y=4.2, text="I (Current Source)",
                      font=dict(size=12, color='#e74c3c'), showarrow=False)
    fig.add_annotation(x=9.5, y=4.2, text="I (Current Sink)",
                      font=dict(size=12, color='#e74c3c'), showarrow=False)

    # Labels
    fig.add_annotation(x=5, y=4.8, text="CURRENT PROBES (Outer)",
                      font=dict(size=12, color='#e74c3c'), showarrow=False)
    fig.add_annotation(x=5, y=5.3, text="VOLTAGE PROBES (Inner)",
                      font=dict(size=12, color='#3498db'), showarrow=False)

    # Probe spacing annotations
    fig.add_annotation(x=1.5, y=2.5, text="s", font=dict(size=10, color='#7f8c8d'), showarrow=False)
    fig.add_annotation(x=4.5, y=2.5, text="s", font=dict(size=10, color='#7f8c8d'), showarrow=False)
    fig.add_annotation(x=7.5, y=2.5, text="s", font=dict(size=10, color='#7f8c8d'), showarrow=False)

    fig.update_layout(
        showlegend=False,
        xaxis=dict(range=[-0.5, 10.5], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(range=[0, 6], showgrid=False, zeroline=False, showticklabels=False),
        height=350,
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=20, b=20)
    )

    return fig


def draw_comparison_diagram():
    """Draw comparison between two-probe and four-probe errors"""
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=("Two-Probe (Higher Error)", "Four-Probe (Lower Error)"),
                       horizontal_spacing=0.15)

    # Left: Two-probe
    fig.add_shape(type="rect", x0=1, y0=2, x1=9, y1=3,
                  fillcolor='#bdc3c7', line=dict(color='#7f8c8d', width=2),
                  row=1, col=1)
    fig.add_shape(type="rect", x0=0.3, y0=2.2, x1=1.1, y1=2.8,
                  fillcolor='#3498db', row=1, col=1)
    fig.add_shape(type="rect", x0=8.9, y0=2.2, x1=9.7, y0=2.8,
                  fillcolor='#3498db', row=1, col=1)

    # Error indicators for two-probe
    fig.add_annotation(x=0.7, y=1.5, text="⚠ Contact R",
                      font=dict(size=10, color='#e74c3c'), showarrow=False, row=1, col=1)
    fig.add_annotation(x=9.3, y=1.5, text="⚠ Contact R",
                      font=dict(size=10, color='#e74c3c'), showarrow=False, row=1, col=1)

    fig.add_annotation(x=5, y=4, text="R_total = R_contact + R_wires + R_sample",
                      font=dict(size=10, color='#c0392b'), showarrow=False, row=1, col=1)

    # Right: Four-probe
    fig.add_shape(type="rect", x0=0.5, y0=2, x1=9.5, y1=3,
                  fillcolor='#bdc3c7', line=dict(color='#7f8c8d', width=2),
                  row=1, col=2)
    fig.add_shape(type="rect", x0=0.1, y0=2.2, x1=0.9, y0=2.8,
                  fillcolor='#2ecc71', row=1, col=2)
    fig.add_shape(type="rect", x0=2.3, y0=2.2, x1=3.1, y0=2.8,
                  fillcolor='#3498db', row=1, col=2)
    fig.add_shape(type="rect", x0=6.9, y0=2.2, x1=7.7, y0=2.8,
                  fillcolor='#3498db', row=1, col=2)
    fig.add_shape(type="rect", x0=9.1, y0=2.2, x1=9.9, y0=2.8,
                  fillcolor='#2ecc71', row=1, col=2)

    fig.add_annotation(x=5, y=4, text="Only V measured, I doesn't flow through V meter",
                      font=dict(size=10, color='#27ae60'), showarrow=False, row=1, col=2)

    fig.update_layout(
        showlegend=False,
        height=300,
        plot_bgcolor='white'
    )

    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)

    return fig


# ============== MAIN APP ==============

def main():
    # Header
    st.markdown('<p class="main-header">⚡ Resistivity Measurement Laboratory</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Two-Probe & Four-Probe Methods for Electrical Resistivity Measurement</p>', unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown("## 📊 Measurement Configuration")

    method = st.sidebar.radio(
        "Measurement Method",
        ["Two-Probe Method", "Four-Probe Method"],
        help="Select the measurement technique"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 Sample Parameters")

    sample_type = st.sidebar.selectbox(
        "Sample Type",
        ["Bulk Material", "Thin Film"],
        help="Select sample geometry"
    )

    is_thin_film = sample_type == "Thin Film"

    L = st.sidebar.number_input("Length L (cm)", value=2.5, min_value=0.001, format="%.3f")
    A = st.sidebar.number_input("Cross-sectional Area A (cm²)", value=0.1, min_value=0.001, format="%.4f")

    s = st.sidebar.number_input("Probe Spacing s (cm)", value=0.5, min_value=0.001, format="%.3f")

    t = None
    if is_thin_film:
        t = st.sidebar.number_input("Thickness t (cm)", value=0.01, min_value=0.0001, format="%.5f")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚡ Contact & Wire Resistances")

    R_contact = st.sidebar.number_input("Contact Resistance R_c (Ω)", value=0.5, min_value=0.0, format="%.4f")
    R_wires = st.sidebar.number_input("Wire Resistance R_w (Ω)", value=0.1, min_value=0.0, format="%.4f")

    # Main content
    tabs = st.tabs(["📝 Single Measurement", "📈 Multiple Readings", "🔬 Method Comparison"])

    with tabs[0]:
        st.markdown("### Single Point Measurement")

        col1, col2 = st.columns([1, 1])

        with col1:
            I = st.number_input("Current I (A)", value=0.1, min_value=0.0001, format="%.5f", key="single_I")
            V = st.number_input("Voltage V (V)", value=0.5, min_value=0.0, format="%.5f", key="single_V")

            calculate = st.button("🔬 Calculate Resistivity", use_container_width=True)

        with col2:
            if calculate and I > 0:
                if method == "Two-Probe Method":
                    R_measured, R_sample, rho = compute_resistivity_two_probe(V, I, L, A, R_contact, R_wires)

                    if rho is not None:
                        st.markdown("#### Results")

                        res_col1, res_col2, res_col3 = st.columns(3)
                        with res_col1:
                            st.metric("Measured Resistance", f"{R_measured:.4f} Ω",
                                     delta=f"- {2*R_contact + R_wires:.4f} Ω contact/wire")
                        with res_col2:
                            st.metric("Sample Resistance", f"{R_sample:.4f} Ω")
                        with res_col3:
                            st.metric("Resistivity ρ", f"{rho:.4f} Ω·cm")

                        st.markdown("#### Formula Used")
                        st.latex(r" \rho = \frac{R_{sample} \cdot A}{L} = \frac{(V/I - 2R_c - R_w) \cdot A}{L} ")

                        # Draw diagram
                        fig = draw_two_probe_diagram()
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Invalid measurement values")

                else:  # Four-probe
                    R_voltage, rho = compute_resistivity_four_probe(V, I, s, t, is_thin_film)

                    if rho is not None:
                        st.markdown("#### Results")

                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.metric("Voltage/Current Ratio", f"{R_voltage:.4f} Ω")
                        with res_col2:
                            st.metric("Resistivity ρ", f"{rho:.4f} Ω·cm")

                        st.markdown("#### Formula Used")
                        if is_thin_film:
                            st.latex(r" \rho = \frac{\pi}{\ln(2)} \cdot t \cdot \frac{V}{I} ")
                            st.caption("Thin film formula (correction for finite sample)")
                        else:
                            st.latex(r" \rho = 2\pi s \cdot \frac{V}{I} ")
                            st.caption("Bulk material formula")

                        fig = draw_four_probe_diagram()
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.error("Invalid measurement values")
            else:
                st.info("👆 Enter current and voltage values, then click Calculate")

    with tabs[1]:
        st.markdown("### Multiple Readings - V-I Curve Analysis")

        # Data input section
        st.markdown("#### Enter Measurement Data")

        col_headers = ["#", "Current (A)", "Voltage (V)"]
        default_data = [
            [0.05, 0.25],
            [0.10, 0.50],
            [0.15, 0.75],
            [0.20, 1.00],
            [0.25, 1.25],
            [0.30, 1.50],
        ]

        # Initialize session state
        if 'data_rows' not in st.session_state:
            st.session_state.data_rows = default_data

        num_rows = st.number_input("Number of data points", min_value=2, max_value=20, value=len(st.session_state.data_rows))

        # Ensure we have enough rows
        while len(st.session_state.data_rows) < num_rows:
            st.session_state.data_rows.append([0.1, 0.5])

        # Editable data table
        new_data = []
        cols = st.columns([0.5, 1, 1] + [0.3] * (num_rows - 3) if num_rows > 3 else [0.5, 1, 1])

        with st.container():
            for i in range(num_rows):
                col_idx = i % 3
                row_idx = i // 3

                if i % 3 == 0:
                    cols = st.columns([0.5, 1, 1, 0.5, 0.5])

                with cols[0]:
                    st.write(f"#{i+1}")
                with cols[1]:
                    current_val = st.number_input(f"I", value=st.session_state.data_rows[i][0],
                                                 format="%.5f", key=f"I_{i}", label_visibility="collapsed")
                with cols[2]:
                    voltage_val = st.number_input(f"V", value=st.session_state.data_rows[i][1],
                                                  format="%.5f", key=f"V_{i}", label_visibility="collapsed")

                new_data.append([current_val, voltage_val])

        # Update session state
        st.session_state.data_rows = new_data[:num_rows]

        if st.button("📊 Generate V-I Graph", use_container_width=True):
            currents = [row[0] for row in st.session_state.data_rows]
            voltages = [row[1] for row in st.session_state.data_rows]

            slope, intercept, r_squared = linear_fit(currents, voltages)

            # Display graph
            fig = plot_v_i_graph(voltages, currents, method)
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            # Calculate resistivity
            I_avg = sum(currents) / len(currents)
            V_avg = sum(voltages) / len(voltages)

            st.markdown("#### Calculated Results")

            res_cols = st.columns(4)
            with res_cols[0]:
                st.metric("Slope (R)", f"{slope:.4f} Ω",
                         help="Resistance from linear fit")
            with res_cols[1]:
                st.metric("R² Value", f"{r_squared:.4f}",
                         help="Coefficient of determination")
            with res_cols[2]:
                st.metric("Avg Current", f"{I_avg:.5f} A")
            with res_cols[3]:
                st.metric("Avg Voltage", f"{V_avg:.5f} V")

            # Calculate final resistivity
            if method == "Two-Probe Method":
                R_sample = slope - (2 * R_contact + R_wires)
                rho = R_sample * (A / L)
            else:
                if is_thin_film and t:
                    rho = (math.pi / math.log(2)) * t * (V_avg / I_avg)
                else:
                    rho = 2 * math.pi * s * (V_avg / I_avg)

            st.markdown("#### Final Resistivity")
            st.metric("ρ", f"{rho:.4f} Ω·cm",
                     help="Final calculated resistivity based on average readings")

    with tabs[2]:
        st.markdown("### Method Comparison & Theory")

        show_comparison = st.checkbox("Show Ideal vs Real Measurement Comparison", value=True)

        if show_comparison:
            fig = draw_comparison_diagram()
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("### 📚 Theoretical Background")

        theory_col1, theory_col2 = st.columns(2)

        with theory_col1:
            st.markdown("#### Two-Probe Method")
            st.markdown("""
            **Principle:** Current and voltage are measured using the same two probes.

            **Advantages:**
            - Simple setup
            - Fewer contact points
            - Lower cost

            **Disadvantages:**
            - Contact resistance included in measurement
            - Wire resistance affects accuracy
            - Less accurate for low-resistivity materials
            - Systematic error from voltage drop at contacts
            """)

            st.latex(r" R_{measured} = R_{sample} + 2R_{contact} + R_{wires} ")

        with theory_col2:
            st.markdown("#### Four-Probe Method")
            st.markdown("""
            **Principle:** Current is injected through outer probes, voltage is measured across inner probes.

            **Advantages:**
            - Eliminates contact resistance
            - No current through voltage leads
            - High accuracy for all resistivity ranges
            - Ideal for semiconductor measurements

            **Disadvantages:**
            - More complex setup
            - More contact points to align
            - Higher cost
            """)

            if is_thin_film:
                st.latex(r" \rho = \frac{\pi}{\ln(2)} \cdot t \cdot \frac{V}{I} ")
            else:
                st.latex(r" \rho = 2\pi s \cdot \frac{V}{I} ")

        st.markdown("---")

        # Info box
        st.markdown("""
        <div class="info-box">
        <h4>💡 Why Four-Probe is More Accurate</h4>
        <p>In the four-probe method, the <strong>voltmeter has very high impedance</strong>, so almost <strong>no current flows</strong> through the voltage probes.
        This means the voltage measured is the <strong>actual voltage drop</strong> across the sample, without any contribution from contact resistance.</p>
        <p>In the two-probe method, the same probes carry both current and voltage, so the measured voltage includes unwanted drops at the contact points.</p>
        </div>
        """, unsafe_allow_html=True)

        # Accuracy comparison
        st.markdown("#### Typical Accuracy Comparison")

        accuracy_data = {
            "Method": ["Two-Probe", "Four-Probe"],
            "Contact Error": ["2-10 Ω typical", "≈ 0 Ω"],
            "Wire Error": ["0.1-1 Ω typical", "Negligible"],
            "Best Use": ["High-R", "All ranges"],
            "Relative Error": ["5-20%", "0.1-1%"]
        }

        st.table({
            "Parameter": ["Contact Error", "Wire Error", "Best Application", "Relative Error"],
            "Two-Probe": ["2-10 Ω", "0.1-1 Ω", "High resistance", "5-20%"],
            "Four-Probe": ["≈ 0 Ω", "Negligible", "All ranges", "0.1-1%"]
        })

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #7f8c8d; font-size: 0.9rem;">
    <p>⚡ Resistivity Measurement Lab | Educational Tool for Physics & Engineering</p>
    <p>Formulas: IEEE Standard 442-1981 | ASTM F84</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()