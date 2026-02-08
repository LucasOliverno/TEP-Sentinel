# 🏭 TEP-Sentinel: Industrial AI System for Tennessee Eastman Process

**TEP-Sentinel** is an advanced AI system designed for real-time Fault Detection, Diagnosis, and Control of the Tennessee Eastman Process (TEP). It integrates Unsupervised Learning (Autoencoders), Reinforcement Learning (RL), and Explainable AI (RAG) into a unified dashboard.

![Architecture](Arquitetura%20png.png)

## 🚀 Key Features

*   **FDD (Fault Detection & Diagnosis):**
    *   **Autoencoder (LSTM):** Detects anomalies based on reconstruction error (MSE).
    *   **Classifier (CNN-LSTM):** Identifies specific fault types (20 classes).
    *   **Persistent Fault Injection:** Robust testing with continuous fault application logic.
*   **RL Control (Reinforcement Learning):**
    *   **PPO Agent:** Autonomous control of 11 valves to stabilize the process.
    *   **Safety Layer:** Hard-coded constraints to prevent catastrophic states.
*   **Explainable AI (RAG):**
    *   **RAG Agent:** Uses Google Gemini + ChromaDB to explain faults in natural language based on technical manuals.
    *   **Prioritization Logic:** Filters downstream effects (e.g., Kinetics) to report root causes (e.g., Feed Loss).

## 🛠️ Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/yourusername/tep-sentinel.git
    cd tep-sentinel
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Setup Environment:**
    Create a `.env` file with your Google API Key:
    ```ini
    GOOGLE_API_KEY=your_key_here
    ```

## 🏁 Usage

**Launch the Dashboard:**
Double-click `run_dashboard.bat` or run:
```bash
python launch_dashboard.py
```

### Dashboard Controls
1.  **Monitor:** Watch real-time process variables (Pressure, Temp, Flow).
2.  **Inject Fault:** Select a fault (e.g., IDV(6)) and click "Apply Condition".
3.  **Analysis:** Observe the "Anomaly Score" spike and read the AI diagnosis.
4.  **Control:** Toggle "RL Control" to let the agent stabilize the plant (requires trained model).

## 📂 Project Structure

*   `dashboard.py`: Main Streamlit interface.
*   `tep_system.py`: Core system logic (FDD + RL + RAG integration).
*   `envs/`: Gym environment for the TEP simulation.
*   `models/`: Trained Keras and RL models.
*   `rag_agent.py`: LangChain RAG implementation.
*   `Banco de Conhecimento/`: Markdown files for RAG knowledge base.

## 📄 License

MIT License.
