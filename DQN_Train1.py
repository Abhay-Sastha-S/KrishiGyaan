import gym
import numpy as np
import random
import pandas as pd
from gym import spaces


def safe_float(value, default=0.0):
    """Safely convert a value to float, returning a default if missing or invalid."""
    try:
        v = float(value)
        if np.isnan(v):
            return default
        return v
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    """Safely convert a value to int, returning a default if missing or invalid."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


class CropRotationEnv(gym.Env):
    """
    Custom OpenAI Gym environment for optimizing crop rotation patterns using a real dataset.
    This environment loads the dataset once, caches its rows for efficiency, and supports optional
    normalization of soil properties in the observation.
    """

    def __init__(
        self,
        max_steps=50,
        dataset_path="FINALSTATICDATASET.xlsx",
        yield_scale=100,
        reward_clip=None,
        normalize_observations=False
    ):
        """
        Parameters:
          max_steps (int): Maximum steps per episode (increased to capture longer-term dynamics).
          dataset_path (str): Path to the Excel dataset.
          yield_scale (float): Scaling factor for the yield component (reduced from 1000 to 100).
          reward_clip (tuple or None): If provided as (min, max), the final reward is clipped to this range.
          normalize_observations (bool): If True, soil features are normalized based on dataset min-max.
        """
        super(CropRotationEnv, self).__init__()
        self.max_steps = max_steps
        self.current_step = 0
        self.yield_scale = yield_scale
        self.reward_clip = reward_clip
        self.normalize_observations = normalize_observations

        # Load the dataset from Excel.
        self.dataset = pd.read_excel(dataset_path)
        self.data_len = len(self.dataset)
        # Cache dataset rows as a list of dictionaries for faster access.
        self.dataset_rows = self.dataset.to_dict('records')

        # Verify that all required columns are present.
        required_columns = [
            "pH", "Moisture (%)", "N (kg/ha)", "P (kg/ha)", "K (kg/ha)",
            "Organic Matter (%)", "Action", "Reward",
            "Next_State_R1", "Next_State_R2", "Next_State_R3", "Season_Order", "Yield"
        ]
        for col in required_columns:
            if col not in self.dataset.columns:
                raise ValueError(f"Dataset missing required column: {col}")

        # Construct the list of crops from the dataset.
        crop_list = self.dataset["Action"].dropna().unique().tolist()
        for col in ["Next_State_R1", "Next_State_R2", "Next_State_R3"]:
            for crop in self.dataset[col].dropna().unique().tolist():
                if crop not in crop_list:
                    crop_list.append(crop)
        self.crops = crop_list
        self.crop_to_index = {crop: i for i, crop in enumerate(self.crops)}

        # Define the action space.
        self.action_space = spaces.Discrete(len(self.crops))

        # Define the observation space.
        self.observation_space = spaces.Dict({
            'soil': spaces.Box(low=0.0, high=1.0 if self.normalize_observations else -np.inf, shape=(6,), dtype=np.float32),
            'past_crops': spaces.MultiDiscrete([len(self.crops)] * 3),
            'season': spaces.Discrete(10)
        })

        # If normalizing observations, precompute min and max for soil features.
        if self.normalize_observations:
            self.soil_norm = {}
            for col in ["pH", "Moisture (%)", "N (kg/ha)", "P (kg/ha)", "K (kg/ha)", "Organic Matter (%)"]:
                min_val = self.dataset[col].min()
                max_val = self.dataset[col].max()
                self.soil_norm[col] = (min_val, max_val)

        self.reset()

    def reset(self):
        """Reset the environment state using a random cached row from the dataset."""
        self.current_step = 0
        self.current_data_index = random.randint(0, self.data_len - 1)
        row = self.dataset_rows[self.current_data_index]

        # Load soil properties.
        self.soil = {
            "pH": safe_float(row.get("pH"), default=7.0),
            "Moisture (%)": safe_float(row.get("Moisture (%)"), default=0.7),
            "N (kg/ha)": safe_float(row.get("N (kg/ha)"), default=1.0),
            "P (kg/ha)": safe_float(row.get("P (kg/ha)"), default=0.0),
            "K (kg/ha)": safe_float(row.get("K (kg/ha)"), default=0.0),
            "Organic Matter (%)": safe_float(row.get("Organic Matter (%)"), default=0.2)
        }

        # Initialize past crop history.
        past = []
        for col in ["Next_State_R1", "Next_State_R2", "Next_State_R3"]:
            crop_name = row.get(col, None)
            if pd.isna(crop_name) or crop_name not in self.crop_to_index:
                past.append(random.randint(0, len(self.crops) - 1))
            else:
                past.append(self.crop_to_index[crop_name])
        self.past_crops = past

        # Set season.
        self.season = safe_int(row.get("Season_Order"), default=0)

        return self._get_obs()

    def _get_obs(self):
        """Return the current observation as a dictionary."""
        soil_values = [
            self.soil["pH"],
            self.soil["Moisture (%)"],
            self.soil["N (kg/ha)"],
            self.soil["P (kg/ha)"],
            self.soil["K (kg/ha)"],
            self.soil["Organic Matter (%)"]
        ]
        soil_array = np.array(soil_values, dtype=np.float32)

        # Optionally normalize soil features.
        if self.normalize_observations:
            normalized_soil = []
            for i, col in enumerate(["pH", "Moisture (%)", "N (kg/ha)", "P (kg/ha)", "K (kg/ha)", "Organic Matter (%)"]):
                min_val, max_val = self.soil_norm[col]
                if max_val - min_val != 0:
                    normalized_val = (soil_array[i] - min_val) / (max_val - min_val)
                else:
                    normalized_val = soil_array[i]
                normalized_soil.append(normalized_val)
            soil_array = np.array(normalized_soil, dtype=np.float32)

        return {
            'soil': soil_array,
            'past_crops': np.array(self.past_crops, dtype=np.int32),
            'season': self.season
        }

    def _compute_reward(self, action, row):
        """
        Compute the reward based on:
          1. Yield: Scaled from the dataset's 'Yield' column.
          2. Soil Health: Calculated using organic matter, nutrient levels, and penalties for deviations from optimal pH and moisture.
          3. Rotation: A bonus if the chosen crop matches the expected crop in the dataset ('Action'), or a penalty if not.
        """
        # Yield Component.
        yield_value = safe_float(row.get("Yield"), default=0.0)
        yield_component = yield_value * self.yield_scale  # Reduced scaling factor.

        # Soil Health Component.
        pH = safe_float(row.get("pH"), default=7.0)
        moisture = safe_float(row.get("Moisture (%)"), default=0.7)
        organic_matter = safe_float(row.get("Organic Matter (%)"), default=0.2)
        N = safe_float(row.get("N (kg/ha)"), default=1.0)
        P = safe_float(row.get("P (kg/ha)"), default=0.0)
        K = safe_float(row.get("K (kg/ha)"), default=0.0)
        nutrient_sum = N + P + K
        soil_health_component = organic_matter + nutrient_sum - (abs(pH - 7.0) + abs(moisture - 0.7))

        # Rotation Component.
        chosen_crop = self.crops[action]
        expected_crop = row.get("Action")
        rotation_component = 1 if chosen_crop == expected_crop else -1

        # Combine components.
        reward = yield_component + soil_health_component + rotation_component

        # Optional reward clipping.
        if self.reward_clip is not None:
            reward = np.clip(reward, self.reward_clip[0], self.reward_clip[1])
        return reward

    def step(self, action):
        """
        Take an action, update the environment state using the next cached row,
        compute the reward, and indicate if the episode is done.
        """
        done = False
        info = {}

        # Advance the dataset pointer cyclically.
        self.current_data_index = (self.current_data_index + 1) % self.data_len
        next_row = self.dataset_rows[self.current_data_index]

        # Update soil properties.
        self.soil = {
            "pH": safe_float(next_row.get("pH"), default=self.soil["pH"]),
            "Moisture (%)": safe_float(next_row.get("Moisture (%)"), default=self.soil["Moisture (%)"]),
            "N (kg/ha)": safe_float(next_row.get("N (kg/ha)"), default=self.soil["N (kg/ha)"]),
            "P (kg/ha)": safe_float(next_row.get("P (kg/ha)"), default=self.soil["P (kg/ha)"]),
            "K (kg/ha)": safe_float(next_row.get("K (kg/ha)"), default=self.soil["K (kg/ha)"]),
            "Organic Matter (%)": safe_float(next_row.get("Organic Matter (%)"), default=self.soil["Organic Matter (%)"])
        }

        # Update past crop history.
        past = []
        for col in ["Next_State_R1", "Next_State_R2", "Next_State_R3"]:
            crop_name = next_row.get(col, None)
            if pd.isna(crop_name) or crop_name not in self.crop_to_index:
                past.append(random.randint(0, len(self.crops) - 1))
            else:
                past.append(self.crop_to_index[crop_name])
        self.past_crops = past

        # Update season.
        self.season = safe_int(next_row.get("Season_Order"), default=self.season)

        # Compute reward.
        reward = self._compute_reward(action, next_row)

        self.current_step += 1
        if self.current_step >= self.max_steps:
            done = True

        return self._get_obs(), reward, done, info

    def render(self, mode='human'):
        """Render the current state of the environment."""
        print(f"Step: {self.current_step}")
        print("Soil:", self.soil)
        print("Past Crops:", [self.crops[i] for i in self.past_crops])
        print("Season:", self.season)
        print("---------")


# ================================
# Integration with Stable-Baselines3 DQN
# ================================
if __name__ == "__main__":
    from stable_baselines3 import DQN
    from stable_baselines3.common.evaluation import evaluate_policy

    # Create the custom environment with improvements.
    env = CropRotationEnv(
        max_steps=50,  # Increased episode length.
        dataset_path="FINALSTATICDATASET.xlsx",
        yield_scale=100,           # Reduced yield scaling.
        reward_clip=(-1000, 1000),  # Clip extreme reward values.
        normalize_observations=True  # Enable normalization for soil features.
    )

    # Create the DQN model using MultiInputPolicy for dictionary observations.
    model = DQN(
        "MultiInputPolicy",
        env,
        device="cuda",
        verbose=1,
        learning_rate=0.0003,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        target_update_interval=1000,
        buffer_size=50000,
        learning_starts=1000,
        tau=0.005
    )

    # Train the DQN model for 200,000 timesteps.
    model.learn(total_timesteps=500000)

    # Evaluate the trained agent.
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print("Mean reward:", mean_reward, "Std reward:", std_reward)

    # Run one test episode and render the environment state.
    obs = env.reset()
    for _ in range(env.max_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()

        # Save the trained model
        model.save("dqn_crop_rotationFINALV3")
        print("✅ Model saved successfully!")

