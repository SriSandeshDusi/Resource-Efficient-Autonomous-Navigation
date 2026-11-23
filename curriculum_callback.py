"""
curriculum_callback.py
Automated Curriculum Learning Callback for Stable-Baselines3.

Monitors the agent's success rate over a rolling 100-episode window
and automatically promotes the environment to harder difficulty stages
when performance thresholds are met.

Stage 1 -> 2: Success Rate >= 50%
Stage 2 -> 3: Success Rate >= 60%
Stage 3 -> 4: Success Rate >= 70%
"""

from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class CurriculumCallback(BaseCallback):
    """
    Curriculum Learning Callback.

    Tracks per-episode success/failure outcomes across all parallel
    environments and advances the curriculum stage when the rolling
    success rate meets the predefined thresholds.
    """

    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.current_stage = 1
        self.episode_outcomes = deque(maxlen=200)  # store more than 100 for safety

        # Stage advancement thresholds
        self.thresholds = {
            1: 0.50,
            2: 0.60,
            3: 0.70,
        }

    def _on_training_start(self):
        if self.verbose >= 1:
            print("[Curriculum] Training started at Stage 1.")

    def _on_step(self) -> bool:
        # Check for completed episodes in all parallel environments
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        if dones is not None:
            for i, done in enumerate(dones):
                if done and i < len(infos):
                    info = infos[i]
                    is_success = info.get("is_success", False)
                    self.episode_outcomes.append(1.0 if is_success else 0.0)

        # Evaluate curriculum advancement every 20 steps to reduce overhead
        if self.n_calls % 20 != 0:
            return True

        if len(self.episode_outcomes) < 100:
            return True

        # Calculate rolling success rate over the last 100 episodes
        recent = list(self.episode_outcomes)[-100:]
        success_rate = sum(recent) / len(recent)

        # Check if we should advance to the next stage
        if self.current_stage in self.thresholds:
            threshold = self.thresholds[self.current_stage]
            if success_rate >= threshold:
                self.current_stage += 1

                # Push new stage to all parallel environments
                self.training_env.env_method(
                    "set_curriculum_stage", self.current_stage
                )

                if self.verbose >= 1:
                    total_steps = self.num_timesteps
                    print(
                        f"[Curriculum] Advanced to Stage {self.current_stage} "
                        f"(S={success_rate:.2f}) at timestep {total_steps:,}"
                    )

        return True

    def _on_training_end(self):
        if self.verbose >= 1:
            print(f"[Curriculum] Training ended at Stage {self.current_stage}.")
            if len(self.episode_outcomes) >= 100:
                recent = list(self.episode_outcomes)[-100:]
                final_rate = sum(recent) / len(recent)
                print(f"[Curriculum] Final rolling success rate: {final_rate:.2f}")
