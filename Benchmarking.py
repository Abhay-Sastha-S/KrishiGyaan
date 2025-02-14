import numpy as np
import torch
from stable_baselines3 import DQN
from ENVIRON import CropRotationEnv  # Ensure this matches your environment filename
import time

# ✅ Load the saved model
model_path = "dqn_crop_rotationFINALV3.zip"  # Change to your actual model file path
env = CropRotationEnv(max_steps=50)  # Ensure the same settings as training
model = DQN.load(model_path, env=env)

# 📊 **Benchmarking Configuration**
N_EPISODES = 100  # Number of episodes to test
success_count = 0
exploration_moves = 0
exploitation_moves = 0
total_rewards = []
episode_lengths = []
soil_changes = []
crop_diversity = set()
fps_times = []

print("\n🚀 Running Benchmarking Tests...\n")

# 📈 **Run Evaluation**
for episode in range(N_EPISODES):
    obs = env.reset()
    done = False
    episode_reward = 0
    start_time = time.time()
    step_count = 0
    initial_soil = env.soil.copy()  # Store initial soil condition

    while not done:
        # 🔍 Get AI's action (exploration vs. exploitation check)
        action, _ = model.predict(obs, deterministic=False)
        if np.random.rand() < model.exploration_rate:
            exploration_moves += 1
        else:
            exploitation_moves += 1

        # 🏆 Take action in the environment
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        step_count += 1

        # 🌾 Track Crop Rotation Diversity
        crop_diversity.add(env.crops[action])

    end_time = time.time()

    # ✅ Success Rate: If AI chooses the expected crop
    expected_crop = env.dataset.iloc[env.current_data_index]["Action"]
    if env.crops[action] == expected_crop:
        success_count += 1

    # 🚀 FPS Tracking
    fps_times.append(end_time - start_time)

    # 🌱 Soil Health Change Tracking
    final_soil = env.soil.copy()
    soil_change = {key: final_soil[key] - initial_soil[key] for key in final_soil}
    soil_changes.append(soil_change)

    total_rewards.append(episode_reward)
    episode_lengths.append(step_count)

# 📊 **Performance Metrics Calculation**
mean_reward = np.mean(total_rewards)
std_reward = np.std(total_rewards)
success_rate = (success_count / N_EPISODES) * 100
exploration_ratio = (exploration_moves / (exploration_moves + exploitation_moves)) * 100 if (exploration_moves + exploitation_moves) > 0 else 0
avg_episode_length = np.mean(episode_lengths)
fps_avg = 1 / np.mean(fps_times)
soil_health_change = {key: np.mean([sc[key] for sc in soil_changes]) for key in soil_changes[0]}
crop_rotation_diversity = len(crop_diversity) / len(env.crops)

# 📢 **Display Results**
print("\n📊 **Benchmarking Results**\n")
print(f"✅ **Mean Reward:** {mean_reward:.2f}")
print(f"📉 **Reward Standard Deviation:** {std_reward:.2f}")
print(f"🏆 **Success Rate:** {success_rate:.2f}%")
print(f"🔄 **Exploration Rate:** {exploration_ratio:.2f}%")
print(f"📏 **Avg. Episode Length:** {avg_episode_length:.2f}")
print(f"🌱 **Soil Health Change:** {soil_health_change}")
print(f"🌾 **Crop Rotation Diversity:** {crop_rotation_diversity:.2f}")
print(f"🚀 **Training Speed (FPS):** {fps_avg:.2f}\n")

print("✅ Benchmarking Complete! 🚀")
