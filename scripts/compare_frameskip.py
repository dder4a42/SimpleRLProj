"""Side-by-side comparison of different frameskip configurations.

This script runs multiple instances of the same game with different
frameskip values to visualize the difference in real-time.

Usage:
    python scripts/compare_frameskip.py --env ALE/Pong-v5
"""

import argparse
import cv2
import numpy as np
import gymnasium as gym

try:
    import ale_py
    gym.register_envs(ale_py)
except Exception:
    pass


class FrameskipComparator:
    """Compare different frameskip settings side-by-side."""

    def __init__(self, env_name, frameskip_values=[1, 2, 4, 8]):
        self.env_name = env_name
        self.frameskip_values = frameskip_values

        # Create multiple environments
        self.envs = []
        self.observations = []

        for fs in frameskip_values:
            env = gym.make(
                env_name,
                frameskip=fs,
                repeat_action_probability=0.0,
                render_mode="rgb_array",
            )
            obs, info = env.reset(seed=42)
            self.envs.append(env)
            self.observations.append(obs)

        self.frame_count = 0
        self.action = 0
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def reset(self):
        """Reset all environments."""
        for i, env in enumerate(self.envs):
            obs, info = env.reset(seed=42)
            self.observations[i] = obs
        self.frame_count = 0

    def step(self, action):
        """Step all environments with the same action."""
        rewards = []
        terms = []
        truncs = []

        for i, env in enumerate(self.envs):
            obs, reward, term, trunc, info = env.step(action)
            self.observations[i] = obs
            rewards.append(reward)
            terms.append(term)
            truncs.append(trunc)

        return rewards, terms, truncs

    def _preprocess(self, frame):
        """Preprocess frame like the RL agent sees it."""
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def create_display(self):
        """Create side-by-side comparison display."""
        n_envs = len(self.envs)

        # Layout: 2 rows x n_envs columns
        # Top row: Raw RGB
        # Bottom row: Preprocessed

        cell_w = 240
        cell_h = 240
        total_w = n_envs * cell_w
        total_h = cell_h * 2 + 60  # +60 for title bar

        display = np.zeros((total_h, total_w, 3), dtype=np.uint8)

        # Title bar
        title = f"Frameskip Comparison - {self.env_name} | Frame: {self.frame_count} | Comparing: {self.frameskip_values}"
        cv2.rectangle(display, (0, 0), (total_w, 60), (50, 50, 50), -1)
        cv2.putText(display, title, (10, 35), self.font, 0.6, (0, 255, 0), 2)

        # Controls text
        controls = "Controls: Arrows=Move, Space=Fire, R=Reset, Q=Quit"
        cv2.putText(display, controls, (10, 55), self.font, 0.4, (200, 200, 200), 1)

        # Process each environment
        for i, (fs, obs) in enumerate(zip(self.frameskip_values, self.observations)):
            x_offset = i * cell_w

            # Raw RGB (top row)
            raw_scaled = cv2.resize(obs, (cell_w, cell_h))
            y_offset = 60
            display[y_offset:y_offset + cell_h, x_offset:x_offset + cell_w] = raw_scaled

            # Label
            label = f"FS={fs}"
            cv2.rectangle(display, (x_offset, y_offset), (x_offset + 80, y_offset + 30), (0, 0, 0), -1)
            cv2.putText(display, label, (x_offset + 5, y_offset + 22), self.font, 0.6, (0, 255, 255), 2)

            # Preprocessed (bottom row)
            proc = self._preprocess(obs)
            proc_scaled = cv2.resize(proc, (cell_w, cell_h), interpolation=cv2.INTER_NEAREST)
            proc_bgr = cv2.cvtColor(proc_scaled, cv2.COLOR_GRAY2BGR)
            y_offset = 60 + cell_h
            display[y_offset:y_offset + cell_h, x_offset:x_offset + cell_w] = proc_bgr

            # Label
            cv2.rectangle(display, (x_offset, y_offset), (x_offset + 150, y_offset + 30), (0, 0, 0), -1)
            cv2.putText(display, "Agent View", (x_offset + 5, y_offset + 22), self.font, 0.5, (255, 255, 255), 1)

        return display

    def run(self):
        """Run the comparison."""
        print(f"\n{'='*60}")
        print(f"Frameskip Comparison - {self.env_name}")
        print(f"{'='*60}")
        print(f"\nComparing frameskip values: {self.frameskip_values}")
        print(f"\nControls:")
        print(f"  WASD or Arrow Keys: Move")
        print(f"  Space: Fire / Action")
        print(f"  R: Reset")
        print(f"  Q or ESC: Quit")
        print(f"{'='*60}\n")

        while True:
            display = self.create_display()
            cv2.imshow("Frameskip Comparison", display)

            # Shorter delay for more responsive input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:
                print("\nQuitting...")
                break
            elif key == ord('r'):
                self.reset()
                print("Reset all environments!")
            # Movement controls (WASD)
            elif key == ord('w') or key == ord('W'):
                self.action = self._map_action("UP")
            elif key == ord('s') or key == ord('S'):
                self.action = self._map_action("DOWN")
            elif key == ord('a') or key == ord('A'):
                self.action = self._map_action("LEFT")
            elif key == ord('d') or key == ord('D'):
                self.action = self._map_action("RIGHT")
            # Fire action
            elif key == ord(' '):
                self.action = self._map_action("FIRE")
            # Arrow keys (Windows-specific codes)
            elif key == 0:  # Up
                self.action = self._map_action("UP")
            elif key == 1:  # Down
                self.action = self._map_action("DOWN")
            elif key == 2:  # Left
                self.action = self._map_action("LEFT")
            elif key == 3:  # Right
                self.action = self._map_action("RIGHT")
            elif key == 0xFF:
                pass  # No key
            else:
                self.action = 0  # NOOP

            # Step all environments
            rewards, terms, truncs = self.step(self.action)
            self.frame_count += 1

            # Check terminations
            if any(terms) or any(truncs):
                for i, (fs, t, tr) in enumerate(zip(self.frameskip_values, terms, truncs)):
                    if t or tr:
                        print(f"  FS={fs}: Episode finished")

        cv2.destroyAllWindows()
        for env in self.envs:
            env.close()

    def _map_action(self, action_name):
        """Map human-friendly action name to environment action ID."""
        action_meanings = getattr(self.envs[0].unwrapped, "get_action_meanings", lambda: [])
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
    parser = argparse.ArgumentParser(description="Compare frameskip settings")
    parser.add_argument("--env", type=str, default="ALE/Pong-v5",
                        help="Gymnasium environment name")
    parser.add_argument("--frameskips", type=int, nargs="+", default=[1, 2, 4, 8],
                        help="Frameskip values to compare (default: 1 2 4 8)")

    args = parser.parse_args()

    comparator = FrameskipComparator(
        env_name=args.env,
        frameskip_values=args.frameskips,
    )
    comparator.run()


if __name__ == "__main__":
    main()
