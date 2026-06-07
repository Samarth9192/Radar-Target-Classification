import streamlit as st
import numpy as np
import pandas as pd
import pickle
import json
import matplotlib.pyplot as plt
import shap
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Radar Target Classification",
    page_icon="📡",
    layout="wide"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0a0a0a; }
[data-testid="stSidebar"] { background-color: #111111; }
h1, h2, h3 { color: #e8e8e8; }
.target-box {
    border-radius: 10px;
    padding: 18px 22px;
    text-align: center;
    margin: 8px 0;
}
.bird-box    { background: rgba(78,157,224,0.12); border: 1px solid #4e9de0; }
.drone-box   { background: rgba(241,165,53,0.12); border: 1px solid #f1a535; }
.aircraft-box{ background: rgba(80,200,120,0.12); border: 1px solid #50c878; }
.stealth-box { background: rgba(255,107,107,0.12);border: 1px solid #ff6b6b; }
.info-box    { background: #1a1a1a; border: 1px solid #2a2a2a;
               border-radius: 10px; padding: 14px 18px; }
</style>
""", unsafe_allow_html=True)

CLASS_NAMES  = {0: 'Bird', 1: 'Drone', 2: 'Aircraft', 3: 'Stealth UAV'}
CLASS_COLORS = {0: '#4e9de0', 1: '#f1a535', 2: '#50c878', 3: '#ff6b6b'}
CLASS_EMOJI  = {0: '🐦', 1: '🚁', 2: '✈️', 3: '👻'}
CLASS_BOX    = {0: 'bird-box', 1: 'drone-box', 2: 'aircraft-box', 3: 'stealth-box'}


@st.cache_resource
def load_all():
    with open('rf_model_rc.pkl', 'rb') as f:
        rf = pickle.load(f)
    with open('scaler_rc.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names_rc.json') as f:
        feature_names = json.load(f)
    with open('radar_samples.json') as f:
        samples = json.load(f)
    with open('radar_signals.json') as f:
        signals = json.load(f)
    return rf, scaler, feature_names, samples, signals

rf, scaler, feature_names, samples, signals = load_all()


def predict(features_raw):
    row = np.array(features_raw).reshape(1, -1)
    row_scaled = scaler.transform(row)
    pred  = rf.predict(row_scaled)[0]
    probs = rf.predict_proba(row_scaled)[0]
    return int(pred), probs, row_scaled


def plot_signal(amp, vel, color, title):
    fig, axes = plt.subplots(2, 1, figsize=(9, 4))
    fig.patch.set_facecolor('#1a1a1a')
    for ax in axes:
        ax.set_facecolor('#111')
        ax.tick_params(colors='#888')
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[0].plot(amp, color=color, linewidth=1.5)
    axes[0].set_ylabel('Amplitude', color='#888', fontsize=9)
    axes[0].set_title(title, color='#e8e8e8', fontsize=11)

    axes[1].plot(vel, color='#888', linewidth=1.2)
    axes[1].set_ylabel('Velocity', color='#888', fontsize=9)
    axes[1].set_xlabel('Timestep', color='#888', fontsize=9)

    plt.tight_layout()
    return fig


def plot_confidence(probs):
    fig, ax = plt.subplots(figsize=(6, 3))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#111')
    colors = [CLASS_COLORS[i] for i in range(4)]
    bars = ax.barh(
        [CLASS_NAMES[i] for i in range(4)],
        probs * 100, color=colors, alpha=0.8
    )
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height()/2,
                f'{prob*100:.1f}%', va='center',
                color='#e8e8e8', fontsize=10)
    ax.set_xlabel('Confidence (%)', color='#888')
    ax.set_xlim(0, 115)
    ax.tick_params(colors='#888')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Confidence per Class', color='#e8e8e8', fontsize=11)
    plt.tight_layout()
    return fig


def plot_shap(row_scaled_df):
    with st.spinner("Computing SHAP..."):
        explainer = shap.TreeExplainer(rf)
        sv = explainer.shap_values(row_scaled_df)
        pred_class = int(rf.predict(row_scaled_df)[0])
        # Normalize shap_values to a 1D array for the predicted class
        if isinstance(sv, list):
            # list of arrays (one per class): each array shape (n_samples, n_features)
            shap_vals = sv[pred_class][0]
        else:
            # sv is a numpy array. Possible shapes:
            # (n_samples, n_features, n_classes) -> pick [:, :, pred_class]
            # (n_samples, n_features) -> pick [0]
            # (n_features, n_classes) -> pick [:, pred_class]
            arr = np.array(sv)
            if arr.ndim == 3:
                shap_vals = arr[0, :, pred_class]
            elif arr.ndim == 2:
                # prefer (1, n_features)
                if arr.shape[0] == 1:
                    shap_vals = arr[0]
                # or (n_features, n_classes)
                elif arr.shape[1] == len(CLASS_NAMES):
                    shap_vals = arr[:, pred_class]
                else:
                    shap_vals = arr[0]
            else:
                shap_vals = arr.flatten()

        # Normalize expected/base value for the predicted class
        ev = explainer.expected_value
        if isinstance(ev, np.ndarray):
            if ev.ndim == 1:
                base_val = ev[pred_class]
            elif ev.ndim == 2:
                base_val = ev[0][pred_class]
            else:
                base_val = float(np.ravel(ev)[pred_class])
        elif isinstance(ev, (list, tuple)):
            base_val = ev[pred_class]
        else:
            base_val = ev
        exp = shap.Explanation(
            values=np.array(shap_vals),
            base_values=float(base_val),
            data=row_scaled_df.values[0],
            feature_names=feature_names
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor('#1a1a1a')
        # Use the newer plotting API which expects a single explanation
        shap.plots.waterfall(exp, max_display=12, show=False)
        # Ensure text in the SHAP plot is white for visibility on dark background
        for a in fig.axes:
            a.set_facecolor('#111')
            a.tick_params(colors='#fff')
            a.set_title(a.get_title(), color='#fff')
            a.set_xlabel(a.get_xlabel(), color='#fff')
            a.set_ylabel(a.get_ylabel(), color='#fff')
            for lbl in a.get_xticklabels() + a.get_yticklabels():
                lbl.set_color('#fff')
        for t in fig.texts:
            t.set_color('#fff')
        plt.tight_layout()
        return fig


# ── SIDEBAR
st.sidebar.title("📡 Radar Classifier")
st.sidebar.caption("Random Forest · 16 features · 100% accuracy")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", [
    "🎯 Live Classifier",
    "📡 Signal Patterns",
    "🔬 Robustness Analysis",
    "📊 Model Intelligence",
])



# PAGE 1 — LIVE CLASSIFIER

if page == "🎯 Live Classifier":
    st.title("🎯 Live Radar Target Classifier")
    st.markdown("Simulate radar returns from real dataset and classify airborne targets.")
    st.markdown("---")

    tab1, tab2 = st.tabs(["🎲 Random Simulator", "🧪 Manual Input"])

    # ── TAB 1 — RANDOM SIMULATOR
    with tab1:
        col_a, col_b = st.columns([1, 1])
        with col_a:
            target_filter = st.selectbox(
                "Target type to simulate",
                ["Any", "Bird", "Drone", "Aircraft", "Stealth UAV"]
            )
        with col_b:
            st.markdown("<br>", unsafe_allow_html=True)
            pick = st.button("📡 Pick Real Radar Return",
                             use_container_width=True)

        if pick or 'sim_label' not in st.session_state:
            label_map = {"Any": None, "Bird": 0, "Drone": 1,
                         "Aircraft": 2, "Stealth UAV": 3}
            chosen = label_map[target_filter]
            if chosen is None:
                chosen = int(np.random.randint(0, 4))

            # pick random real sample from dataset
            class_samples = samples[str(chosen)]
            class_signals = signals[str(chosen)]
            idx = np.random.randint(0, len(class_samples))
            sig_idx = np.random.randint(0, len(class_signals))

            st.session_state['sim_label']   = chosen
            st.session_state['sim_feats']   = class_samples[idx]
            st.session_state['sim_amp']     = class_signals[sig_idx]

        true    = st.session_state['sim_label']
        feats   = st.session_state['sim_feats']
        amp_sig = np.array(st.session_state['sim_amp'])

        pred, probs, row_scaled = predict(feats)
        row_df = pd.DataFrame(row_scaled, columns=feature_names)

        st.markdown("---")

        # approximate velocity from amplitude gradient
        vel_approx = np.gradient(amp_sig)
        fig = plot_signal(
            amp_sig, vel_approx,
            CLASS_COLORS[true],
            f"Real Radar Return — True: {CLASS_EMOJI[true]} {CLASS_NAMES[true]}"
        )
        st.pyplot(fig)
        plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="target-box {CLASS_BOX[pred]}">
                <div style="font-size:42px">{CLASS_EMOJI[pred]}</div>
                <div style="font-size:22px;font-weight:700;
                            color:#e8e8e8;margin-top:6px">
                    {CLASS_NAMES[pred]}
                </div>
                <div style="font-size:14px;color:#888;margin-top:4px">
                    Predicted Target
                </div>
                <div style="font-size:18px;color:{CLASS_COLORS[pred]};
                            margin-top:6px;font-weight:600">
                    {probs[pred]*100:.1f}% confidence
                </div>
            </div>""", unsafe_allow_html=True)

            if pred == true:
                st.success("✅ Correct prediction")
            else:
                st.error(f"❌ Misclassified — actual was "
                         f"{CLASS_EMOJI[true]} {CLASS_NAMES[true]}")

        with col2:
            fig2 = plot_confidence(probs)
            st.pyplot(fig2)
            plt.close()

        st.markdown("---")
        st.subheader("Why this prediction?")
        fig3 = plot_shap(row_df)
        st.pyplot(fig3)
        plt.close()
        st.caption("Red = pushes toward this class · Blue = pushes away")

    # ── TAB 2 — MANUAL INPUT
    with tab2:
        st.markdown("Adjust radar signal characteristics. Prediction updates live.")
        st.markdown("---")

        col_l, col_r = st.columns([1, 1])

        with col_l:
            st.subheader("Signal Parameters")

            amp_std  = st.slider("Amplitude Std — signal variability (Signal fluctuation level)",
                                 0.05, 2.0, 0.6, 0.05)
            amp_mean = st.slider("Amplitude Mean — signal level",
                                 -0.5, 1.5, 0.1, 0.05)
            dom_freq = st.slider("Dominant Frequency — FFT peak (Rotor/engine signature)",
                                 0.0, 49.0, 10.0, 1.0)
            autocorr = st.slider("Autocorrelation lag-5 (Motion consistency)",
                                 -1.0, 1.0, 0.3, 0.05)

            st.markdown("---")
            st.caption("""
**Input guide:**
- 🐦 Bird → std > 1.0, autocorr < 0.2, freq ~3
- 🚁 Drone → std ~0.7, autocorr > 0.7, freq ~10
- ✈️ Aircraft → std < 0.2, autocorr > 0.9, mean > 0.3
- 👻 Stealth → std ~0.4, autocorr ~0, freq scattered
            """)

        amp_var   = amp_std ** 2
        amp_range = amp_std * 3.5
        amp_min   = amp_mean - amp_range / 2
        amp_max   = amp_mean + amp_range / 2
        amp_energy = amp_var * 100

        manual_feats = [
            amp_mean, amp_std, amp_var, amp_min, amp_max,
            amp_range, amp_energy,
            0.0, amp_std * 0.3, amp_var * 0.09,
            -amp_std * 0.5, amp_std * 0.5,
            amp_std * 1.0, amp_energy * 0.09,
            dom_freq, autocorr
        ]

        pred_m, probs_m, row_m = predict(manual_feats)
        row_m_df = pd.DataFrame(row_m, columns=feature_names)

        with col_r:
            st.subheader("Prediction")
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="target-box {CLASS_BOX[pred_m]}">
                <div style="font-size:42px">{CLASS_EMOJI[pred_m]}</div>
                <div style="font-size:24px;font-weight:700;
                            color:#e8e8e8;margin-top:6px">
                    {CLASS_NAMES[pred_m]}
                </div>
                <div style="font-size:18px;color:{CLASS_COLORS[pred_m]};
                            margin-top:6px;font-weight:600">
                    {probs_m[pred_m]*100:.1f}% confidence
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            fig_m = plot_confidence(probs_m)
            st.pyplot(fig_m)
            plt.close()

        st.markdown("---")
        st.subheader("Why this prediction?")
        fig_shap_m = plot_shap(row_m_df)
        st.pyplot(fig_shap_m)
        plt.close()



# PAGE 2 — SIGNAL PATTERNS

elif page == "📡 Signal Patterns":
    st.title("📡 Radar Signal Patterns")
    st.markdown("Real micro-Doppler signatures from the dataset.")
    st.markdown("---")

    descriptions = {
        0: "Irregular flapping motion creates chaotic amplitude spikes. "
           "High variance, no dominant frequency. Most unpredictable radar return.",
        1: "Rotor blades spinning at fixed RPM produce a strong periodic signal. "
           "Clear dominant frequency in FFT. High autocorrelation — the signal repeats.",
        2: "Smooth linear approach creates a steadily rising amplitude trend. "
           "Near-zero variance. Highest prediction confidence of all classes.",
        3: "Low radar cross-section design keeps amplitude minimal. "
           "Noise-dominant signal. Hardest to classify under interference."
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor('#1a1a1a')
    axes = axes.flatten()

    for label in [0, 1, 2, 3]:
        amp_sig = np.array(signals[str(label)][0])
        ax = axes[label]
        ax.set_facecolor('#111')
        ax.tick_params(colors='#888')
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.plot(amp_sig, color=CLASS_COLORS[label], linewidth=2)
        ax.set_title(f"{CLASS_EMOJI[label]} {CLASS_NAMES[label]}",
                     color='#e8e8e8', fontsize=13)
        ax.set_xlabel('Timestep', color='#888', fontsize=9)
        ax.set_ylabel('Amplitude', color='#888', fontsize=9)

    plt.suptitle('Real Micro-Doppler Amplitude Signatures from Dataset',
                 color='#e8e8e8', fontsize=14)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Target Characteristics")
    col1, col2 = st.columns(2)
    for i, label in enumerate([0, 1, 2, 3]):
        col = col1 if i % 2 == 0 else col2
        with col:
            st.markdown(f"""
            <div class="target-box {CLASS_BOX[label]}"
                 style="text-align:left;margin-bottom:12px">
                <div style="font-size:20px;font-weight:700;
                            color:{CLASS_COLORS[label]}">
                    {CLASS_EMOJI[label]} {CLASS_NAMES[label]}
                </div>
                <div style="font-size:13px;color:#aaa;
                            margin-top:6px;line-height:1.6">
                    {descriptions[label]}
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Feature Statistics per Class — Real Data")
    stats_data = []
    for label in [0, 1, 2, 3]:
        arr = np.array(samples[str(label)])
        stats_data.append({
            'Target':   f"{CLASS_EMOJI[label]} {CLASS_NAMES[label]}",
            'Amp Mean': round(float(np.mean(arr[:, 0])), 3),
            'Amp Std':  round(float(np.mean(arr[:, 1])), 3),
            'Amp Var':  round(float(np.mean(arr[:, 2])), 3),
            'Dom Freq': round(float(np.mean(arr[:, 14])), 1),
            'Autocorr': round(float(np.mean(arr[:, 15])), 3),
        })
    st.dataframe(pd.DataFrame(stats_data),
                 use_container_width=True, hide_index=True)
    st.caption("Aircraft has near-zero variance — easiest to classify. "
               "Drone has consistent dominant frequency — rotor signature. "
               "Bird has highest std — irregular flapping.")



# PAGE 3 — ROBUSTNESS ANALYSIS

elif page == "🔬 Robustness Analysis":
    st.title("🔬 Robustness Under Radar Noise")
    st.markdown("""
All models achieve 100% on clean data.
This section simulates real-world radar interference by adding Gaussian noise
and tests how classification degrades.
    """)
    st.markdown("---")

    rob_df = pd.DataFrame({
        'Noise Level':          [0.0,   0.1,   0.3,   0.5],
        'Logistic Regression':  [100.0, 100.0, 100.0, 99.11],
        'Random Forest':        [100.0, 100.0, 99.82, 94.46],
        'XGBoost':              [100.0, 98.93, 91.61, 83.21],
        'SVM':                  [100.0, 100.0, 100.0, 99.46],
    })

    noise_val = st.slider("Select Noise Level", 0.0, 0.5, 0.0, 0.1)
    closest   = (rob_df['Noise Level'] - noise_val).abs().argmin()
    row       = rob_df.iloc[closest]

    c1, c2, c3, c4 = st.columns(4)
    for col, model in zip([c1, c2, c3, c4],
                          ['Logistic Regression', 'Random Forest',
                           'XGBoost', 'SVM']):
        delta = f"{row[model]-100:.2f}%" if noise_val > 0 else None
        col.metric(model, f"{row[model]:.2f}%",
                   delta=delta, delta_color="inverse")

    st.markdown("---")

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#111')
    model_colors = {
        'Logistic Regression': '#4e9de0',
        'Random Forest':       '#50c878',
        'XGBoost':             '#f1a535',
        'SVM':                 '#ff6b6b'
    }
    for model, color in model_colors.items():
        ax.plot(rob_df['Noise Level'], rob_df[model],
                marker='o', linewidth=2.5, label=model, color=color)
    ax.axvline(x=noise_val, color='#555', linestyle='--',
               alpha=0.6, label='Selected noise')
    ax.set_xlabel('Noise Level (Gaussian std)', color='#888')
    ax.set_ylabel('Accuracy (%)', color='#888')
    ax.set_title('Model Robustness Under Noise', color='#e8e8e8')
    ax.legend(facecolor='#1a1a1a', labelcolor='#e8e8e8')
    ax.tick_params(colors='#888')
    ax.spines['bottom'].set_color('#333')
    ax.spines['left'].set_color('#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(75, 102)
    ax.grid(alpha=0.15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("Per-Class Accuracy at Noise 0.5")
    per_class = pd.DataFrame({
        'Class':                ['🐦 Bird', '🚁 Drone', '✈️ Aircraft', '👻 Stealth UAV'],
        'Logistic Regression':  ['100%', '97.9%', '100%', '98.6%'],
        'Random Forest':        ['99.3%', '81.4%', '99.3%', '97.9%'],
        'XGBoost':              ['92.9%', '57.9%', '99.3%', '82.9%'],
        'SVM':                  ['100%',  '98.6%', '100%',  '99.3%'],
    })
    st.dataframe(per_class, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Key Finding")
    st.markdown("""
**Drone degrades fastest under noise across all models.**

XGBoost Drone accuracy collapses to **57.9%** at noise 0.5 — barely above random (25% baseline).
Aircraft stays at **99.3%** across all models under all noise levels.

**Why?** Aircraft's linear amplitude trend survives Gaussian noise — the upward slope
is preserved even when individual timestep values are perturbed. Drone's periodic rotor
signature is frequency-based — noise corrupts the periodicity making it resemble Bird's
irregular pattern.

This finding suggests real-world radar systems processing drone signatures need
noise-robust signal preprocessing before model inference.
    """)



# PAGE 4 — MODEL INTELLIGENCE

elif page == "📊 Model Intelligence":
    st.title("📊 Model Intelligence")
    st.markdown("---")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Training Samples", "2,800")
    c2.metric("Classes", "4")
    c3.metric("Features", "16")
    c4.metric("Accuracy", "100%")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model Comparison")
        results = pd.DataFrame({
            'Model': ['Random Forest', 'SVM', 'Logistic Regression',
                      'Decision Tree', 'XGBoost'],
            'Accuracy':  ['100%', '100%', '100%', '100%', '100%'],
            'CV Score':  ['1.000 ±0.000', '1.000 ±0.000',
                          '1.000 ±0.000', '0.9996 ±0.0007', '1.000 ±0.000'],
            'Noise 0.5': ['94.46%', '99.46%', '99.11%', '—', '83.21%'],
        })
        st.dataframe(results, use_container_width=True, hide_index=True)
        st.caption("RF deployed for highest prediction confidence (99.91%). "
                   "SVM most robust under noise (99.46% at 0.5).")

    with col2:
        st.subheader("Prediction Confidence — RF")
        conf_data = pd.DataFrame({
            'Class': ['🐦 Bird', '🚁 Drone', '✈️ Aircraft', '👻 Stealth UAV'],
            'RF Confidence':  ['99.91%', '99.98%', '100.00%', '99.96%'],
            'SVM Confidence': ['99.65%', '99.60%', '99.54%', '99.52%'],
        })
        st.dataframe(conf_data, use_container_width=True, hide_index=True)

    st.markdown("---")
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Feature Importance (RF)")
        feat_imp = {
            'amp_var': 0.153, 'amp_std': 0.149, 'amp_energy': 0.138,
            'amp_range': 0.095, 'vel_var': 0.081, 'amp_min': 0.072,
            'vel_std': 0.065, 'vel_energy': 0.058, 'amp_max': 0.042,
            'amp_mean': 0.038, 'vel_min': 0.031, 'vel_range': 0.030,
            'amp_dominant_freq': 0.022, 'vel_max': 0.018,
            'amp_autocorr_lag5': 0.005, 'vel_mean': 0.000
        }
        fig, ax = plt.subplots(figsize=(6, 6))
        fig.patch.set_facecolor('#1a1a1a')
        ax.set_facecolor('#111')
        items = sorted(feat_imp.items(), key=lambda x: x[1])
        bar_colors = ['#50c878' if k.startswith('amp') else '#4e9de0'
                      for k, _ in items]
        ax.barh([k for k, _ in items], [v for _, v in items],
                color=bar_colors)
        ax.set_xlabel('Importance', color='#888')
        ax.tick_params(colors='#888', labelsize=8)
        ax.set_title('Green = Amplitude · Blue = Velocity',
                     color='#888', fontsize=9)
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader("Amplitude vs Velocity")
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor('#1a1a1a')
        ax.pie([67.4, 32.6],
               labels=['Amplitude\n67.4%', 'Velocity\n32.6%'],
               colors=['#50c878', '#4e9de0'],
               autopct='%1.1f%%',
               textprops={'color': '#e8e8e8'})
        ax.set_title('Feature Group Importance', color='#e8e8e8')
        fig.patch.set_facecolor('#1a1a1a')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("""
        <div class="info-box" style="margin-top:12px">
            <div style="color:#888;font-size:12px;line-height:1.6">
            Amplitude features dominate — consistent with radar physics.
            Velocity adds 32.6% supporting information but amplitude return
            is the primary target discriminator at this level.
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("Feature Engineering")
    st.markdown("""
Raw dataset: **2,800 samples × 300 columns** (100 timesteps × 3 signals)

Instead of 300 raw columns — **16 meaningful features** extracted across 3 layers:

- **Statistical (14):** Mean, std, variance, min, max, range, energy for amplitude + velocity
- **Frequency (1):** FFT dominant frequency — captures drone rotor periodicity
- **Temporal (1):** Autocorrelation lag-5 — captures signal self-similarity

Energy column verified as abs(amplitude) — dropped as mathematically redundant.
    """)
