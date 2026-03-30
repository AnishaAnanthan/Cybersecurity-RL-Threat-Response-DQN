from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st
from stable_baselines3 import DQN

# Ensure sibling package imports work when this file is run directly (e.g., via Streamlit).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from env.cyber_env import CyberSecurityEnv


ACTION_LABELS = {
    0: "Allow",
    1: "Block",
    2: "Monitor",
    3: "Alert",
}

ACTION_THEME = {
    0: {"chip": "chip-allow", "tag": "Safe"},
    1: {"chip": "chip-block", "tag": "Strict"},
    2: {"chip": "chip-monitor", "tag": "Watch"},
    3: {"chip": "chip-alert", "tag": "Warning"},
}


def apply_custom_styles():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --bg-start: #f3f7f9;
            --bg-end: #eaf2ef;
            --ink: #0f2f3a;
            --subtle: #48626a;
            --panel: #ffffffcc;
            --line: #bfd1d4;
            --allow: #18815f;
            --block: #c24739;
            --monitor: #945300;
            --alert: #876d00;
        }

        .stApp {
            font-family: 'Space Grotesk', sans-serif;
            background:
                radial-gradient(circle at 15% 20%, #c8e6df55 0, transparent 35%),
                radial-gradient(circle at 85% 10%, #f5d6bb55 0, transparent 32%),
                linear-gradient(145deg, var(--bg-start), var(--bg-end));
            color: var(--ink);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        .hero {
            border: 1px solid var(--line);
            background: var(--panel);
            border-radius: 16px;
            padding: 1.2rem 1.3rem;
            margin-bottom: 1rem;
            backdrop-filter: blur(4px);
        }

        .hero h1 {
            margin: 0;
            font-size: 2rem;
            letter-spacing: 0.2px;
        }

        .hero p {
            margin: 0.35rem 0 0 0;
            color: var(--subtle);
            font-size: 0.98rem;
        }

        .section-label {
            font-size: 0.84rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--subtle);
            margin-top: 0.35rem;
            margin-bottom: 0.35rem;
            font-weight: 600;
        }

        .action-card {
            border-radius: 14px;
            border: 1px solid var(--line);
            background: var(--panel);
            padding: 0.9rem 1rem;
            margin-top: 0.35rem;
            margin-bottom: 0.6rem;
        }

        .action-row {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            flex-wrap: wrap;
        }

        .chip {
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 700;
            border: 1px solid transparent;
        }

        .chip-allow { background: #e7f7f1; color: var(--allow); border-color: #9fdac7; }
        .chip-block { background: #fdebe8; color: var(--block); border-color: #f1b2aa; }
        .chip-monitor { background: #fff3df; color: var(--monitor); border-color: #e8cc9e; }
        .chip-alert { background: #fff8dd; color: var(--alert); border-color: #e8d79c; }

        .metric-card {
            border: 1px solid var(--line);
            border-radius: 12px;
            background: #ffffffd8;
            padding: 0.55rem 0.75rem;
            margin-bottom: 0.5rem;
        }

        .metric-title {
            font-size: 0.74rem;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--subtle);
            margin-bottom: 0.15rem;
        }

        .metric-value {
            font-size: 1.06rem;
            font-weight: 700;
            font-family: 'IBM Plex Mono', monospace;
        }

        .stButton > button {
            border-radius: 10px;
            border: 1px solid #9fb4b9;
            font-weight: 700;
            padding-top: 0.55rem;
            padding-bottom: 0.55rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_env_and_model():
    data_path = PROJECT_ROOT / "dataset" / "data.csv"
    model_path = PROJECT_ROOT / "dqn_cyber_model"
    zip_model_path = PROJECT_ROOT / "dqn_cyber_model.zip"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    if not model_path.exists() and not zip_model_path.exists():
        raise FileNotFoundError(
            "Model not found. Train first with: python -m train.train_dqn"
        )

    env = CyberSecurityEnv(str(data_path))
    model = DQN.load(str(model_path if model_path.exists() else zip_model_path))
    return env, model


def compute_reward_details(observation: np.ndarray, action: int, env: CyberSecurityEnv):
    traffic_intensity = float(np.mean(observation))
    q1 = float(env.intensity_q1)
    q2 = float(env.intensity_q2)
    q3 = float(env.intensity_q3)

    if traffic_intensity < q1:
        severity_band = 0
    elif traffic_intensity < q2:
        severity_band = 1
    elif traffic_intensity < q3:
        severity_band = 2
    else:
        severity_band = 3

    action_risk = {0: 0, 3: 1, 2: 2, 1: 3}[int(action)]
    distance = abs(action_risk - severity_band)
    reward = 30 - 15 * distance

    if severity_band == 3 and int(action) == 0:
        reward -= 20
    if severity_band == 0 and int(action) == 1:
        reward -= 10

    band_names = {
        0: "low",
        1: "mild",
        2: "high",
        3: "critical",
    }
    reason = (
        f"Input severity band is {band_names[severity_band]} (band {severity_band}) and "
        f"action risk distance is {distance}."
    )

    return reward, traffic_intensity, severity_band, reason


def severity_name(severity_band: int):
    return {
        0: "Low",
        1: "Mild",
        2: "High",
        3: "Critical",
    }.get(severity_band, f"Band {severity_band}")


def render_metric_card(title: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="Cyber RL Demo", page_icon="", layout="wide")
    apply_custom_styles()
    st.markdown(
        """
        <div class="hero">
            <h1>CyberSecurity RL Demo</h1>
            <p>Provide normalized traffic features, predict a response action, and inspect reward reasoning step by step.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        env, model = load_env_and_model()
    except Exception as exc:
        st.error(str(exc))
        return

    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "input_obs" not in st.session_state:
        default_obs, _ = env.reset()
        st.session_state.input_obs = default_obs.tolist()
    if "show_details" not in st.session_state:
        st.session_state.show_details = False

    feature_names = env.feature_names
    left_col, right_col = st.columns([1.35, 1], gap="large")

    with left_col:
        st.markdown('<div class="section-label">Input Panel</div>', unsafe_allow_html=True)
        st.markdown("Use normalized values between 0 and 1 for each feature.")

        with st.form("prediction_form"):
            cols = st.columns(2)
            input_values = []
            for idx, feature_name in enumerate(feature_names):
                with cols[idx % 2]:
                    value = st.number_input(
                        label=feature_name,
                        min_value=0.0,
                        max_value=1.0,
                        value=float(st.session_state.input_obs[idx]),
                        step=0.01,
                        format="%.4f",
                        key=f"feature_{idx}",
                    )
                    input_values.append(value)

            predict_clicked = st.form_submit_button("Predict Action", use_container_width=True)

        if predict_clicked:
            obs = np.array(input_values, dtype=np.float32)
            action, _ = model.predict(obs, deterministic=True)
            action_int = int(action)
            st.session_state.input_obs = obs.tolist()
            st.session_state.prediction = {
                "obs": obs,
                "action": action_int,
                "action_label": ACTION_LABELS.get(action_int, f"Action {action_int}"),
            }
            st.session_state.show_details = False

    with right_col:
        st.markdown('<div class="section-label">Model Output</div>', unsafe_allow_html=True)

        if st.session_state.prediction is None:
            st.info("Enter values and click Predict Action to see model output.")
        else:
            action = st.session_state.prediction["action"]
            action_label = st.session_state.prediction["action_label"]
            action_theme = ACTION_THEME.get(action, {"chip": "chip-alert", "tag": "Output"})

            st.markdown(
                f"""
                <div class="action-card">
                    <div class="action-row">
                        <strong>Predicted Action:</strong>
                        <span class="chip {action_theme['chip']}">{action_label} (Action {action})</span>
                        <span class="chip {action_theme['chip']}">{action_theme['tag']}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if st.button("View Reward and Prediction Steps", use_container_width=True):
                st.session_state.show_details = True

            if st.session_state.show_details:
                obs = st.session_state.prediction["obs"]
                reward, traffic_intensity, severity_band, reason = compute_reward_details(
                    obs, action, env
                )

                m1, m2 = st.columns(2)
                with m1:
                    render_metric_card("Reward", str(reward))
                    render_metric_card("Severity", severity_name(severity_band))
                with m2:
                    render_metric_card("Traffic Intensity", f"{traffic_intensity:.4f}")
                    render_metric_card("Action", f"{action_label} ({action})")

                with st.expander("Prediction reasoning", expanded=True):
                    st.write("1. User values are converted into a 10-feature observation vector.")
                    st.write("2. Trained DQN predicts one action from this vector.")
                    st.write(f"3. Mean intensity is computed as {traffic_intensity:.4f}.")
                    st.write(
                        "4. Severity band is assigned using dataset quantile thresholds "
                        f"{float(env.intensity_q1):.4f}, {float(env.intensity_q2):.4f}, {float(env.intensity_q3):.4f}."
                    )
                    st.write(f"5. Reward is calculated by action-severity distance: {reason}")

                st.markdown('<div class="section-label">Input Used For Prediction</div>', unsafe_allow_html=True)
                table_df = pd.DataFrame(
                    {
                        "Feature": feature_names,
                        "Value": [round(float(v), 4) for v in obs.tolist()],
                    }
                )
                st.dataframe(table_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()