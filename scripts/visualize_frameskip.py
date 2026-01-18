"""Interactive visualization of frame skipping and stacking in ALE.

This script creates an interactive window to play Atari games while visualizing:
1. Raw RGB frames from the environment
2. Frame skipping effect (frameskip parameter)
3. Frame stacking effect (frame_stack parameter)

Usage:
    python scripts/visualize_frameskip.py --env ALE/Breakout-v5
    python scripts/visualize_frameskip.py --env ALE/Breakout-v5 --frameskip 4 --frame-stack 4
"""

import argparse
import cv2
import numpy as np
import gymnasium as gym
from collections import deque

try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass


class FrameSkipVisualizer:
    """Visualize frame skipping and stacking effects."""

    def __init__(self, env_name, frameskip=4, frame_stack=4, obs_type="rgb"):
        self.env_name = env_name
        self.frameskip = frameskip
        self.frame_stack = frame_stack
        self.obs_type = obs_type.lower()

        # Create environment
        self.env = gym.make(
            env_name,
            frameskip=1,  # We'll handle frameskip manually
            repeat_action_probability=0.0,
            render_mode="rgb_array",
        )

        if self.obs_type == "ram":
            self.env = gym.make(env_name, frameskip=1, obs_type="ram", render_mode="rgb_array")

        self.env.reset(seed=42)

        self.action_space = self.env.action_space.n
        self.all_frames = []
        self.stacked_frames = deque(maxlen=frame_stack)
        self.step_count = 0
        self.game_frame_count = 0

        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def reset(self):
        """Reset the environment and buffers."""
        obs, info = self.env.reset()
        self.all_frames = []
        self.stacked_frames.clear()
        self.step_count = 0
        self.game_frame_count = 0

        if self.obs_type == "rgb":
            processed = self._preprocess_frame(obs)
            for _ in range(self.frame_stack):
                self.stacked_frames.append(processed)
        else:
            for _ in range(self.frame_stack):
                self.stacked_frames.append(obs)

        return obs, info

    def _preprocess_frame(self, frame):
        """Preprocess frame like AtariPreprocess wrapper."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def step(self, action):
        """Execute one step with manual frameskip."""
        self.all_frames = []

        for i in range(self.frameskip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.game_frame_count += 1

            if self.obs_type == "rgb":
                processed = self._preprocess_frame(obs)
                self.all_frames.append(processed)
            else:
                self.all_frames.append(obs)

            if terminated or truncated:
                break

        if len(self.all_frames) > 0:
            self.stacked_frames.append(self.all_frames[-1])
        self.step_count += 1

        return obs, reward, terminated, truncated, info

    def _create_rgb_display(self):
        """Create visualization for RGB observations."""
        # Get current raw frame from environment
        raw_frame = self.env.render()

        # Get processed frame
        if len(self.all_frames) > 0:
            proc_frame = self.all_frames[-1]
        else:
            proc_frame = np.zeros((84, 84), dtype=np.uint8)

        # Scale for display
        raw_scaled = cv2.resize(raw_frame, (320, 320))
        proc_scaled = cv2.resize(proc_frame, (320, 320), interpolation=cv2.INTER_NEAREST)
        proc_bgr = cv2.cvtColor(proc_scaled, cv2.COLOR_GRAY2BGR)

        # Create stacked display
        stack_h, stack_w = 160, 320
        stack_display = np.zeros((stack_h, stack_w, 3), dtype=np.uint8)

        if len(self.stacked_frames) == self.frame_stack:
            stacked = np.stack(self.stacked_frames)
            n_rows = (self.frame_stack + 1) // 2
            for i, frame in enumerate(stacked):
                row = i // 2
                col = i % 2
                frame_scaled = cv2.resize(frame, (160, 160), interpolation=cv2.INTER_NEAREST)
                frame_bgr = cv2.cvtColor(frame_scaled, cv2.COLOR_GRAY2BGR)
                y_start = row * 80
                y_end = y_start + 80
                x_start = col * 160
                x_end = x_start + 160
                stack_display[y_start:y_end, x_start:x_end] = cv2.resize(frame_bgr, (160, 80))

        # Build layout
        # Top row: raw | proc
        # Bottom row: stack | info
        top_row = np.hstack([raw_scaled, proc_bgr])

        # Info panel
        info_panel = np.zeros((stack_h, 320, 3), dtype=np.uint8)
        self._add_text_info(info_panel)

        bottom_row = np.hstack([stack_display, info_panel])

        # Title bar
        title_h = 50
        title_bar = np.zeros((title_h, 640, 3), dtype=np.uint8)
        title_text = f"Frame Skip: {self.frameskip} | Stack: {self.frame_stack} | Game Frame: {self.game_frame_count}"
        cv2.putText(title_bar, title_text, (10, 30), self.font, 0.6, (0, 255, 0), 2)

        display = np.vstack([title_bar, top_row, bottom_row])

        # Add labels
        cv2.putText(display, "Raw RGB", (10, title_h + 25), self.font, 0.5, (255, 255, 255), 1)
        cv2.putText(display, "Preprocessed (84x84)", (330, title_h + 25), self.font, 0.5, (255, 255, 255), 1)
        cv2.putText(display, f"Stacked ({self.frame_stack} frames)", (10, title_h + 340), self.font, 0.5, (255, 255, 255), 1)

        return display

    def _create_ram_display(self):
        """Create visualization for RAM observations."""
        if len(self.all_frames) > 0:
            ram = self.all_frames[-1]
        else:
            ram = np.zeros(128, dtype=np.uint8)

        # RAM as heatmap (8x16 grid)
        ram_grid = ram.reshape(8, 16)
        ram_display = cv2.resize(ram_grid, (320, 320), interpolation=cv2.INTER_NEAREST)
        ram_display = cv2.cvtColor(ram_display, cv2.COLOR_GRAY2BGR)

        # Apply colormap for better visualization
        ram_display = cv2.applyColorMap(ram_display, cv2.COLORMAP_JET)

        # Stacked RAM
        stack_h, stack_w = 320, 320
        stack_display = np.zeros((stack_h, stack_w, 3), dtype=np.uint8)

        if len(self.stacked_frames) == self.frame_stack:
            stacked_ram = np.stack(self.stacked_frames)
            bar_w = stack_w // self.frame_stack
            for i, frame in enumerate(stacked_ram):
                frame_grid = frame.reshape(8, 16)
                bar = cv2.resize(frame_grid, (bar_w, stack_h), interpolation=cv2.INTER_NEAREST)
                bar_bgr = cv2.cvtColor(bar, cv2.COLOR_GRAY2BGR)
                bar_colored = cv2.applyColorMap(bar_bgr, cv2.COLORMAP_JET)
                stack_display[:, i * bar_w:(i + 1) * bar_w] = bar_colored

        # Info panel
        info_panel = np.zeros((stack_h, 320, 3), dtype=np.uint8)
        self._add_text_info(info_panel)

        # Title bar
        title_h = 50
        title_bar = np.zeros((title_h, 640, 3), dtype=np.uint8)
        title_text = f"Frame Skip: {self.frameskip} | Stack: {self.frame_stack} | RAM Mode | Game Frame: {self.game_frame_count}"
        cv2.putText(title_bar, title_text, (10, 30), self.font, 0.6, (0, 255, 0), 2)

        display = np.vstack([title_bar, np.hstack([ram_display, stack_display]), info_panel])

        # Add labels
        cv2.putText(display, "Current RAM (128 bytes as 8x16 grid)", (10, title_h + 25), self.font, 0.5, (255, 255, 255), 1)
        cv2.putText(display, f"Stacked RAM ({self.frame_stack} frames)", (330, title_h + 25), self.font, 0.5, (255, 255, 255), 1)

        return display

    def _add_text_info(self, panel):
        """Add text information to panel."""
        y = 20
        line_h = 22

        lines = [
            f"Env: {self.env_name}",
            f"Agent Steps: {self.step_count}",
            f"Game Frames: {self.game_frame_count}",
            "",
            "Controls:",
            "  WASD: Move",
            "  Space: Fire",
            "  R: Reset | Q: Quit",
            "  1-9: Change frameskip",
            "  T: Toggle frame_stack",
            "  O: Toggle RGB/RAM",
        ]

        for line in lines:
            cv2.putText(panel, line, (10, y), self.font, 0.4, (200, 200, 200), 1)
            y += line_h

    def run(self):
        """Run the interactive visualization."""
        print(f"\n{'='*60}")
        print(f"Frame Skip Visualizer - {self.env_name}")
        print(f"{'='*60}")
        print(f"\nCurrent settings:")
        print(f"  Frame Skip: {self.frameskip}")
        print(f"  Frame Stack: {self.frame_stack}")
        print(f"  Observation Type: {self.obs_type}")
        print(f"\nControls:")
        print(f"  WASD or Arrow Keys: Move")
        print(f"  Space: Fire / Action")
        print(f"  R: Reset episode")
        print(f"  Q or ESC: Quit")
        print(f"  1-9: Change frameskip")
        print(f"  T: Toggle frame_stack (1/4)")
        print(f"  O: Toggle obs_type (RGB/RAM)")
        print(f"{'='*60}\n")

        obs, info = self.reset()
        terminated, truncated = False, False
        current_action = 0  # Keep track of current action

        while True:
            # Create display
            if self.obs_type == "rgb":
                display = self._create_rgb_display()
            else:
                display = self._create_ram_display()

            cv2.imshow("Frame Skip Visualizer", display)

            # Handle keyboard input (shorter delay for more responsive input)
            key = cv2.waitKey(1) & 0xFF

            # Exit
            if key == ord('q') or key == 27:  # q or ESC
                print("\nQuitting...")
                break
            # Reset
            elif key == ord('r'):
                obs, info = self.reset()
                terminated, truncated = False, False
                current_action = 0
                print("Reset!")
            # Change frameskip
            elif key in [ord(str(i)) for i in range(1, 10)]:
                new_frameskip = int(chr(key))
                print(f"Changing frameskip from {self.frameskip} to {new_frameskip}")
                self.frameskip = new_frameskip
            # Toggle frame_stack (use T to avoid conflict with WASD)
            elif key == ord('t') or key == ord('T'):
                new_stack = 1 if self.frame_stack == 4 else 4
                print(f"Changing frame_stack from {self.frame_stack} to {new_stack}")
                self.frame_stack = new_stack
                self.stacked_frames = deque(list(self.stacked_frames), maxlen=new_stack)
            # Toggle obs_type
            elif key == ord('o'):
                self.obs_type = "ram" if self.obs_type == "rgb" else "rgb"
                print(f"Changing obs_type to {self.obs_type}")
                self.env.close()
                if self.obs_type == "ram":
                    self.env = gym.make(self.env_name, frameskip=1, obs_type="ram", render_mode="rgb_array")
                else:
                    self.env = gym.make(self.env_name, frameskip=1, render_mode="rgb_array")
                self.env.reset(seed=42)
                self.stacked_frames.clear()
                self.reset()
                current_action = 0
            # Movement controls (WASD)
            elif key == ord('w') or key == ord('W'):
                current_action = self._map_action("UP")
            elif key == ord('s') or key == ord('S'):
                current_action = self._map_action("DOWN")
            elif key == ord('a') or key == ord('A'):
                current_action = self._map_action("LEFT")
            elif key == ord('d') or key == ord('D'):
                current_action = self._map_action("RIGHT")
            # Fire action
            elif key == ord(' '):
                current_action = self._map_action("FIRE")
            # Arrow keys (Windows-specific codes)
            elif key == 0:  # Up
                current_action = self._map_action("UP")
            elif key == 1:  # Down
                current_action = self._map_action("DOWN")
            elif key == 2:  # Left
                current_action = self._map_action("LEFT")
            elif key == 3:  # Right
                current_action = self._map_action("RIGHT")

            # Execute action
            if not terminated and not truncated:
                obs, reward, terminated, truncated, info = self.step(current_action)
                if reward != 0:
                    print(f"Step {self.step_count}: Reward = {reward}")
            elif terminated or truncated:
                print(f"\nEpisode finished! Press R to reset or Q to quit...")

        cv2.destroyAllWindows()
        self.env.close()

    def _map_action(self, action_name):
        """Map human-friendly action name to environment action ID."""
        action_meanings = getattr(self.env.unwrapped, "get_action_meanings", lambda: [])
        try:
            meanings = [m.upper() for m in action_meanings()]
            if action_name in meanings:
                return meanings.index(action_name)
            # Try partial match
            for i, m in enumerate(meanings):
                if action_name in m:
                    return i
        except Exception:
            pass
        return 0  # Default to NOOP


def main():
    parser = argparse.ArgumentParser(description="Visualize frame skipping effects in ALE")
    parser.add_argument("--env", type=str, default="ALE/Breakout-v5",
                        help="Gymnasium environment name")
    parser.add_argument("--frameskip", type=int, default=4,
                        help="Number of frames to skip per action (default: 4)")
    parser.add_argument("--frame-stack", type=int, default=4,
                        help="Number of frames to stack (default: 4)")
    parser.add_argument("--obs-type", type=str, default="rgb", choices=["rgb", "ram"],
                        help="Observation type (default: rgb)")

    args = parser.parse_args()

    visualizer = FrameSkipVisualizer(
        env_name=args.env,
        frameskip=args.frameskip,
        frame_stack=args.frame_stack,
        obs_type=args.obs_type,
    )
    visualizer.run()


if __name__ == "__main__":
    main()
