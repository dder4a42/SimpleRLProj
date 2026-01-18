"""Capture and save consecutive frames from Pong-v5 with frameskip.

This script demonstrates how frameskip works:
- Environment runs 4 frames internally but only returns 1 observation
- Consecutive observations are NOT consecutive game frames (they're 4 frames apart)

Usage:
    python scripts/capture_pong_frames.py
"""

import cv2
import numpy as np
import gymnasium as gym
from collections import deque

try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass


def preprocess_frame(frame):
    """Preprocess frame like AtariPreprocess wrapper."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
    return resized


class FrameCaptureEnv(gym.Wrapper):
    """Wrapper that captures ALL internal frames, not just the returned one."""

    def __init__(self, env, frameskip=4):
        super().__init__(env)
        self.frameskip = frameskip
        self.all_internal_frames = []  # Capture every frame
        self.game_frame_count = 0

    def step(self, action):
        """Override step to capture all internal frames."""
        self.all_internal_frames = []
        reward_sum = 0.0

        # Execute 'frameskip' frames internally
        for i in range(self.frameskip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.game_frame_count += 1
            reward_sum += reward

            # Capture the actual RGB frame from environment
            raw_frame = self.env.render()
            self.all_internal_frames.append(raw_frame.copy())

            if terminated or truncated:
                break

        # Return only the LAST observation (standard ALE behavior)
        return obs, reward_sum, terminated, truncated, info


def capture_pong_frames():
    """Capture frames from Pong-v5 with proper frameskip demonstration."""
    print("="*60)
    print("Capturing Pong-v5 frames (demonstrating frameskip)")
    print("="*60)

    frameskip = 4
    frame_stack = 4

    # Create environment with frameskip
    base_env = gym.make('ALE/Pong-v5', frameskip=1, render_mode='rgb_array')
    env = FrameCaptureEnv(base_env, frameskip=frameskip)
    obs, info = env.reset(seed=42)

    # Setup frame stacking
    stacked_frames = deque(maxlen=frame_stack)

    # Pre-fill the stack
    initial_processed = preprocess_frame(obs)
    for _ in range(frame_stack):
        stacked_frames.append(initial_processed)

    # Collect data
    agent_observations = []      # What the agent sees (every 4th frame)
    all_internal_frames = []     # ALL game frames (for visualization)
    all_stacked_outputs = []     # Stacked observations

    print(f"\nConfiguration:")
    print(f"  Frameskip: {frameskip} (environment runs {frameskip} frames per action)")
    print(f"  Frame stack: {frame_stack} (agent sees {frame_stack} observations)")
    print(f"\nThis means:")
    print(f"  - Game runs {frameskip} frames but only returns 1 observation")
    print(f"  - Consecutive observations are {frameskip} game frames apart")
    print(f"  - Frame stack contains observations from different game times")
    print()

    step_count = 0
    max_steps = 50

    try:
        # First press FIRE multiple times to ensure ball is launched
        print("Starting game (pressing FIRE)...")
        for _ in range(5):
            obs, reward, terminated, truncated, info = env.step(1)  # FIRE

            # Store these initial frames too
            agent_observations.append(obs.copy())
            all_internal_frames.append(env.all_internal_frames.copy())

            processed = preprocess_frame(obs)
            stacked_frames.append(processed)
            stacked = np.stack(stacked_frames)
            all_stacked_outputs.append(stacked.copy())

            step_count += 1

            # Print reward to see game activity
            if reward != 0:
                print(f"  Step {step_count}: Reward = {reward}")

        # Now take varied actions to make gameplay interesting
        print("Game active! Taking movement actions (LEFT/RIGHT)...")
        # Pong uses LEFT/RIGHT (actions 3 and 2), not UP/DOWN
        actions = [0] + [3, 2, 3, 2, 3, 3, 2, 2, 3, 2] * 10  # NOOP, then varied LEFT/RIGHT movements

        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)

            # Store what the agent receives (single observation per step)
            agent_observations.append(obs.copy())

            # Store ALL internal frames (for demonstration)
            all_internal_frames.append(env.all_internal_frames.copy())

            # Process and stack (what agent actually uses)
            processed = preprocess_frame(obs)
            stacked_frames.append(processed)
            stacked = np.stack(stacked_frames)  # Shape: (4, 84, 84)
            all_stacked_outputs.append(stacked.copy())

            step_count += 1

            # Print rewards to track scoring
            if reward != 0:
                print(f"  Step {step_count}: Reward = {reward} (scoring!)")

            if step_count >= max_steps or terminated or truncated:
                break

            if step_count % 10 == 0:
                print(f"  Captured {step_count} agent steps...")

    except KeyboardInterrupt:
        print("\n  Interrupted by user")

    base_env.close()

    total_agent_steps = len(agent_observations)
    total_game_frames = sum(len(frames) for frames in all_internal_frames)
    print(f"\nCapture complete:")
    print(f"  Agent steps: {total_agent_steps}")
    print(f"  Total game frames: {total_game_frames}")
    print(f"  Ratio: {total_game_frames / total_agent_steps:.1f} game frames per agent step")

    return agent_observations, all_internal_frames, all_stacked_outputs


def save_visualization(agent_observations, all_internal_frames, all_stacked_outputs):
    """Create comprehensive visualization showing frameskip effect."""

    # Select a good starting point (after some warmup)
    start_step = 5  # Start from step 5 to have some game activity

    if start_step + 3 >= len(agent_observations):
        start_step = 0

    # Create visualization - increase height to fit all content
    fig_height = 1200
    fig_width = 1400
    canvas = np.zeros((fig_height, fig_width, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX
    y_offset = 30

    # Title
    title = f"Pong-v5 Frameskip Visualization (frameskip=4, stack=4)"
    cv2.putText(canvas, title, (20, y_offset), font, 0.8, (0, 255, 0), 2)
    y_offset += 40

    # Explanation
    explanation = "Each agent step receives 1 observation, but environment runs 4 game frames internally"
    cv2.putText(canvas, explanation, (20, y_offset), font, 0.5, (200, 200, 200), 1)
    y_offset += 30

    # For 4 consecutive agent steps
    for step_idx in range(4):
        current_step = start_step + step_idx

        # Get data for this step
        agent_obs = agent_observations[current_step]
        internal_frames = all_internal_frames[current_step]
        stacked_obs = all_stacked_outputs[current_step]

        # Check if we have space
        if y_offset + 250 > fig_height:
            break

        # Step label
        step_label = f"Agent Step {current_step}: receives 1 observation (frame {current_step*4+4}), but {len(internal_frames)} game frames ran"
        cv2.putText(canvas, step_label, (20, y_offset), font, 0.6, (255, 200, 0), 2)
        y_offset += 35

        # Show the returned observation (what agent gets)
        obs_small = cv2.resize(agent_obs, (80, 105))
        obs_bgr = cv2.cvtColor(obs_small, cv2.COLOR_RGB2BGR)
        canvas[y_offset:y_offset+105, 20:100] = obs_bgr
        cv2.putText(canvas, "Returned", (20, y_offset + 120), font, 0.4, (0, 255, 0), 1)

        # Show internal frames (what actually ran)
        for i, internal_frame in enumerate(internal_frames):
            x_pos = 120 + i * 85
            if x_pos + 80 < fig_width:
                frame_small = cv2.resize(internal_frame, (80, 105))
                frame_bgr = cv2.cvtColor(frame_small, cv2.COLOR_RGB2BGR)
                canvas[y_offset:y_offset+105, x_pos:x_pos+80] = frame_bgr

                # Label
                game_frame_num = current_step * 4 + i + 1
                label = f"G{game_frame_num}"
                cv2.putText(canvas, label, (x_pos, y_offset + 120), font, 0.4, (150, 150, 255), 1)

        cv2.putText(canvas, "Internal frames", (120, y_offset + 120), font, 0.4, (150, 150, 255), 1)

        y_offset += 140

        # Show stacked observation
        if y_offset + 200 > fig_height:
            break

        stacked_label = f"Stacked observation contains frames from agent steps [{current_step-3}, {current_step-2}, {current_step-1}, {current_step}]"
        cv2.putText(canvas, stacked_label, (20, y_offset), font, 0.5, (255, 255, 0), 1)
        y_offset += 25

        # Create 2x2 grid for stacked frames
        for stack_idx in range(4):
            row = stack_idx // 2
            col = stack_idx % 2
            x_pos = 20 + col * 90
            y_pos = y_offset + row * 90

            frame = stacked_obs[stack_idx]
            frame_scaled = cv2.resize(frame, (85, 85), interpolation=cv2.INTER_NEAREST)
            frame_bgr = cv2.cvtColor(frame_scaled, cv2.COLOR_GRAY2BGR)

            # Make sure we don't exceed canvas bounds
            if y_pos + 85 <= fig_height and x_pos + 85 <= fig_width:
                canvas[y_pos:y_pos+85, x_pos:x_pos+85] = frame_bgr

                agent_step_of_frame = current_step - 3 + stack_idx
                game_frame_of_frame = agent_step_of_frame * 4 + 4
                label = f"Step{agent_step_of_frame}"
                cv2.putText(canvas, label, (x_pos+5, y_pos+20), font, 0.3, (0, 255, 255), 1)
                label2 = f"(G{game_frame_of_frame})"
                cv2.putText(canvas, label2, (x_pos+5, y_pos+35), font, 0.3, (150, 150, 150), 1)

        y_offset += 200

    # Save
    output_path = "D:/sjtu_stuff/RL/Final/pong_frameskip_visualization.png"
    cv2.imwrite(output_path, canvas)
    print(f"\nSaved visualization to: {output_path}")

    return canvas


def save_simple_comparison(agent_observations, all_internal_frames):
    """Save simple 1x4 comparison showing the skipping effect."""

    start_step = 5

    # Get 4 consecutive agent observations
    four_agent_obs = agent_observations[start_step:start_step + 4]

    # Create display: what agent sees vs what actually happened
    cell_h = 210
    cell_w = 160
    result = np.zeros((cell_h, cell_w * 4, 3), dtype=np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, obs in enumerate(four_agent_obs):
        x_pos = i * cell_w
        obs_resized = cv2.resize(obs, (cell_w, cell_h))
        obs_bgr = cv2.cvtColor(obs_resized, cv2.COLOR_RGB2BGR)
        result[:, x_pos:x_pos+cell_w] = obs_bgr

        # Label
        step_num = start_step + i
        game_frame_num = step_num * 4
        label = f"Step {step_num}"
        label2 = f"(Game frames {game_frame_num-3} to {game_frame_num})"

        cv2.putText(result, label, (x_pos + 10, 30), font, 0.6, (0, 255, 0), 2)
        cv2.putText(result, label2, (x_pos + 10, 55), font, 0.4, (200, 200, 200), 1)

        # Mark which frame was returned
        cv2.rectangle(result, (x_pos + 70, 160), (x_pos + 90, 180), (0, 255, 255), 2)
        cv2.putText(result, "Returned", (x_pos + 25, 195), font, 0.4, (0, 255, 255), 1)

    # Add title
    title = "Agent Observations: Each contains game frame 4 of 4 internal frames"
    cv2.putText(result, title, (10, result.shape[0] - 10), font, 0.7, (255, 255, 255), 2)

    output_path = "D:/sjtu_stuff/RL/Final/pong_agent_observations.png"
    cv2.imwrite(output_path, result)
    print(f"Saved agent observations to: {output_path}")

    return result


def main():
    """Main execution."""
    print("\n" + "="*60)
    print("Demonstrating Frameskip in Pong-v5")
    print("="*60 + "\n")

    # Capture frames
    agent_obs, all_internal, all_stacked = capture_pong_frames()

    # Save visualizations
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60 + "\n")

    viz = save_visualization(agent_obs, all_internal, all_stacked)
    simple = save_simple_comparison(agent_obs, all_internal)

    print("\n" + "="*60)
    print("Complete! Generated images:")
    print("  1. pong_frameskip_visualization.png - Detailed frameskip breakdown")
    print("  2. pong_agent_observations.png - Simple agent view")
    print("="*60)

    # Display
    print("\nDisplaying visualization... (press any key to close)")

    # Resize to fit screen
    max_height = 900
    display = viz.copy()
    if display.shape[0] > max_height:
        scale = max_height / display.shape[0]
        new_width = int(display.shape[1] * scale)
        display = cv2.resize(display, (new_width, max_height))

    cv2.imshow("Pong-v5 Frameskip Visualization", display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
