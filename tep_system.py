import numpy as np
import pandas as pd
import sys
import os

try:
    import tensorflow as tf
except ImportError as e:
    print("CRITICAL IMPORT ERROR: Could not import tensorflow")
    print(f"Current Python Executable: {sys.executable}")
    print(f"Current Working Directory: {os.getcwd()}")
    print("System Path:")
    for p in sys.path:
        print(p)
    raise e

import joblib
from stable_baselines3 import PPO
from envs.tep_env import TEPEnv
from rag_agent import RAGAgent

# Suppress TF logs
# Suppress TF logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

class TEPSentinelSystem:
    def __init__(self):
        print("Initializing TEP-Sentinel System...")
        
        # 1. Environment (The Plant)
        self.env = TEPEnv()
        self.state, _ = self.env.reset()
        
        # Buffer for FDD (Window Size = 100)
        self.window_size = 100
        self.feature_count = 52
        # Initialize buffer with steady state (zeros)
        # Initialize buffer with steady state noise (to match Env reset)
        # Using flat zeros causes a "shock" when real noisy data comes in.
        self.buffer = np.random.normal(0, 0.05, size=(self.window_size, self.feature_count))
        
        # 2. Load Models
        print("Loading Models...")
        
        # Resolve absolute paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.scaler = joblib.load(os.path.join(base_dir, "processed_data", "tep_scaler.pkl"))
        self.autoencoder = tf.keras.models.load_model(os.path.join(base_dir, "models", "tep_autoencoder.keras"))
        self.classifier = tf.keras.models.load_model(os.path.join(base_dir, "models", "tep_classifier.keras"))
        self.threshold = joblib.load(os.path.join(base_dir, "models", "tep_threshold.pkl"))
        
        # RL Agent (Try to load if exists)
        rl_path = os.path.join(base_dir, "models", "PPO", "best_model.zip")
        if os.path.exists(rl_path):
            self.rl_agent = PPO.load(rl_path)
            print("RL Agent Loaded.")
        else:
            self.rl_agent = None
            print("WARNING: RL Agent not found. Running in Open Loop/Random.")

        # 3. RAG Agent
        self.rag_init_error = None
        try:
            self.rag_agent = RAGAgent()
            print("RAG Agent Initialized.")
        except Exception as e:
            self.rag_init_error = str(e)
            print(f"RAG Init Failed: {self.rag_init_error}")
            self.rag_agent = None

        # State Variables
        self.fault_status = "NORMAL"
        self.fault_code = 0
        self.current_step = 0
        self.last_rag_report = None
        self.rag_cooldown = 0
        self.active_fault = None

    def get_real_values(self, scaled_state):
        """Unscale data to get physical units (kPa, degC, etc.)"""
        # reshape to (1, 52)
        values = scaled_state.reshape(1, -1)
        real = self.scaler.inverse_transform(values)
        return real[0]

    def monitor(self):
        """Fault Detection & Diagnosis (FDD)"""
        # Create Window (1, 100, 52)
        window_input = self.buffer.reshape(1, self.window_size, self.feature_count)
        
        # 1. Detection (Autoencoder)
        reconstructed = self.autoencoder.predict(window_input, verbose=0)
        mse = np.mean(np.square(window_input - reconstructed))
        
        is_fault = mse > self.threshold
        
        if is_fault:
            self.fault_status = "FAULT"
            # 2. Diagnosis (Classifier)
            probs = self.classifier.predict(window_input, verbose=0)[0]
            fault_idx = np.argmax(probs)
            # Map index to Fault ID (0=Normal, 1=Fault1...)
            # Note: Classifier training labels usually map 0->Fault1? Or 0->Normal?
            # Assuming trained on 21 classes (0=Normal, 1..20=Faults) or (0..20 faults)?
            # Usually: 0 is Normal.
            self.fault_code = fault_idx 
            
            # --- FAULT PRIORITIZATION LOGIC ---
            # Fault 13 (Reaction Kinetics) is often a downstream effect of physical faults.
            # If Model says 13, but there is significant evidence (>15%) of Fault 6 (Feed Loss),
            # prioritize Fault 6 as it is the Root Cause.
            if self.fault_code == 13:
                # Check probability of Fault 6 (Index 6)
                # Note: Indexes depend on training. Assuming Index=FaultID for simplicity.
                # If Index 0 is Normal, then Fault 6 is Index 6.
                if len(probs) > 6 and probs[6] > 0.15:
                    print(f"Start Priority Override: Fault 13 ({probs[13]:.2f}) -> Fault 6 ({probs[6]:.2f})")
                    self.fault_code = 6
            # ---------------------------------- 
        else:
            self.fault_status = "NORMAL"
            self.fault_code = 0
            
        return is_fault, mse

    def inject_fault(self, fault_type):
        """Set the active fault type for persistent injection"""
        print(f"Enabling Fault: {fault_type}")
        if fault_type == "None":
            self.active_fault = None
        else:
            self.active_fault = fault_type

    def _apply_fault_physics(self):
        """Continuously perturb state to simulate persistent fault"""
        if not self.active_fault:
            return

        # IDV(1): A/C Imbalance -> Shift Feed A (Index 0)
        if "IDV(1)" in self.active_fault:
            # Persistent Bias
            self.state[0] -= 20.0 # Force extreme Low Feed A
            self.state[3] -= 20.0 # Force extreme Low Total Feed
            self.state[5] -= 10.0 # Drop Reactor Feed Rate
            
        # IDV(6): A Feed Loss -> Zero out Feed A
        elif "IDV(6)" in self.active_fault:
            self.state[0] = -50.0 # Catastrophic Feed A Loss
            self.state[6] -= 50.0 # Reactor Pressure Drop (Vacuum?)
            self.state[7] -= 30.0 # Reactor Level Drop

    def step(self, action_override=None):
        """Run one system step"""
        self.current_step += 1
        
        # 0. Apply Persistent Faults (Before Control/Physics)
        self._apply_fault_physics()
        
        # 1. Control (RL or Manual)
        if action_override is not None:
            action = action_override
        elif self.rl_agent:
            action, _ = self.rl_agent.predict(self.state, deterministic=True)
        else:
            # action = self.env.action_space.sample() # Random actions destabilize the plant!
            action = np.zeros(11, dtype=np.float32) # Maintain Steady State (0 = Mean)
            
        # 2. Physics Simulation
        next_state, reward, done, truncated, info = self.env.step(action)
        self.state = next_state
        
        # Re-apply fault to the NEW state immediately? 
        # No, let it evolve, but we apply it again at start of next step.
        # But wait, we record 'self.state' into buffer below.
        # If we don't apply fault to 'self.state' NOW, the buffer sees the "healed" state from env.step().
        
        # Post-Step Fault Application (Ensure buffer sees the fault)
        self._apply_fault_physics() # Apply again to corrupt the measurement that goes into buffer

        # 3. Update Buffer (Rolling)
        # Shift left
        self.buffer[:-1] = self.buffer[1:]
        # Add new state at end
        self.buffer[-1] = self.state
        
        # 4. Monitoring (FDD)
        # Warmup Phase: Ignore first 50 steps to let dynamics settle
        if self.current_step < 50:
            is_fault = False
            mse = 0.0
            self.fault_status = "WARMUP"
            self.fault_code = 0
        else:
            is_fault, mse = self.monitor()
        
        # 5. Explainable AI (RAG)
        rag_report = None
        if is_fault and self.rag_agent and self.rag_cooldown == 0:
            # Trigger RAG
            real_vals = self.get_real_values(self.state)
            # Create a summary string of key Low/High vars
            # For brevity, pick a few key vars or pass summary
            sensor_summary = f"MSE Error: {mse:.4f}. Top deviations detected."
            
            try:
                fault_id = f"IDV({self.fault_code})" if self.fault_code > 0 else "Unknown Anomaly"
                rag_report = self.rag_agent.diagnose(fault_id, sensor_summary)
                self.last_rag_report = rag_report
                self.rag_cooldown = 50 # Don't spam RAG every step
                self.rag_runtime_error = None # Clear previous error
            except Exception as e:
                self.rag_runtime_error = str(e)
                print(f"RAG Error: {self.rag_runtime_error}")
        
        if self.rag_cooldown > 0:
            self.rag_cooldown -= 1
            
        return {
            "step": self.current_step,
            "state_scaled": self.state,
            "state_real": self.get_real_values(self.state),
            "status": self.fault_status,
            "fault_code": self.fault_code,
            "mse": mse,
            "action": action,
            "rag_report": self.last_rag_report
        }

if __name__ == "__main__":
    # Simple Test
    system = TEPSentinelSystem()
    print("Running Simulation Loop...")
    for i in range(200):
        data = system.step()
        if i % 20 == 0:
            print(f"Step {i}: Status={data['status']} (MSE={data['mse']:.4f})")
    print("Done.")
