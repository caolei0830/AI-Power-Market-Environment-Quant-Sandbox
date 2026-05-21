# -*- coding: utf-8 -*-
"""
ppo_agent.py — 虚拟电厂 PPO 强化学习智能体 (PyTorch Actor-Critic)

与 ``vpp_environment.VPPEnvironment`` 联用，读取 1536h SQL/SQLite 自愈特征流，
训练连续功率控制策略并在测试集输出储能套利回测胜率。

用法
----
    python ppo_agent.py --production-db
    python ppo_agent.py --demo --epochs 80 --rollout-steps 1536
    python ppo_agent.py --production-db --epochs 120 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.distributions import Normal
except ImportError as exc:  # pragma: no cover
    raise ImportError("未安装 PyTorch。请执行: pip install torch") from exc

from run_experiment import (
    EXPERIMENT_LOOKBACK_HOURS_DEFAULT,
    ExperimentConfig,
    _fallback_hourly_pnl,
)
from vpp_environment import (
    MOCK_SEED_HOURS,
    VPPEnvironment,
    build_vpp_data_bundle,
    load_vpp_market_frame,
)


# ---------------------------------------------------------------------------
# 网络与 PPO 超参
# ---------------------------------------------------------------------------
@dataclass
class PPOHyperParams:
    """PPO 训练超参数。"""

    hidden_dim: int = 256
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 256
    rollout_steps: int = 512
    total_epochs: int = 60
    eval_interval: int = 10
    log_std_init: float = -0.5


class ActorCritic(nn.Module):
    """Gaussian 策略 Actor + 价值 Critic（共享特征提取）。"""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 256,
        log_std_init: float = -0.5,
    ) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((1,), float(log_std_init)))
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.shared(obs)
        mu = torch.tanh(self.mu_head(h))
        std = self.log_std.exp().expand_as(mu)
        value = self.value_head(h).squeeze(-1)
        return mu, std, value

    def act(
        self, obs: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, std, value = self.forward(obs)
        dist = Normal(mu, std)
        if deterministic:
            action = mu
        else:
            action = dist.sample()
        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(-1)
        return action, log_prob, value

    def evaluate(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, std, value = self.forward(obs)
        dist = Normal(mu, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, entropy, value


@dataclass
class RolloutBuffer:
    """单轮 rollout 缓存。"""

    obs: list[np.ndarray]
    actions: list[np.ndarray]
    rewards: list[float]
    dones: list[float]
    log_probs: list[float]
    values: list[float]

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.rewards.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()

    def __len__(self) -> int:
        return len(self.rewards)


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    *,
    gamma: float,
    gae_lambda: float,
    last_value: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Generalized Advantage Estimation。"""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    next_value = last_value

    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        last_gae = delta + gamma * gae_lambda * mask * last_gae
        advantages[t] = last_gae
        next_value = values[t]

    returns = advantages + values
    return advantages, returns


class PPOAgent:
    """PPO 裁剪策略优化器。"""

    def __init__(
        self,
        obs_dim: int,
        *,
        device: str = "cpu",
        hparams: Optional[PPOHyperParams] = None,
    ) -> None:
        self.device = torch.device(device)
        self.hparams = hparams or PPOHyperParams()
        self.net = ActorCritic(
            obs_dim,
            hidden_dim=self.hparams.hidden_dim,
            log_std_init=self.hparams.log_std_init,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.lr)

    def collect_rollout(
        self,
        env: VPPEnvironment,
        *,
        max_steps: int,
    ) -> tuple[RolloutBuffer, float, dict[str, float]]:
        """在环境中收集一条轨迹（可跨 episode reset）。"""
        buf = RolloutBuffer([], [], [], [], [], [])
        ep_returns: list[float] = []
        ep_profit_hours: list[float] = []
        cur_return = 0.0
        cur_profit = 0

        obs, _ = env.reset()
        steps = 0

        while steps < max_steps:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, value = self.net.act(obs_t)

            action_np = action.cpu().numpy().reshape(-1).astype(np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = float(terminated or truncated)

            buf.obs.append(obs.copy())
            buf.actions.append(action_np.copy())
            buf.rewards.append(float(reward))
            buf.dones.append(done)
            buf.log_probs.append(float(log_prob.item()))
            buf.values.append(float(value.item()))

            cur_return += float(reward)
            if reward > 0.0:
                cur_profit += 1
            steps += 1
            obs = next_obs

            if terminated or truncated:
                ep_returns.append(cur_return)
                ep_profit_hours.append(cur_profit / max(1, steps))
                cur_return = 0.0
                cur_profit = 0
                obs, _ = env.reset()

        stats = {
            "mean_episode_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
            "rollout_return": float(np.sum(buf.rewards)),
        }
        return buf, float(np.sum(buf.rewards)), stats

    def update(self, buf: RolloutBuffer, last_value: float = 0.0) -> dict[str, float]:
        """PPO 裁剪更新一步。"""
        hp = self.hparams
        rewards = np.asarray(buf.rewards, dtype=np.float32)
        values = np.asarray(buf.values, dtype=np.float32)
        dones = np.asarray(buf.dones, dtype=np.float32)
        old_log_probs = np.asarray(buf.log_probs, dtype=np.float32)

        advantages, returns = compute_gae(
            rewards,
            values,
            dones,
            gamma=hp.gamma,
            gae_lambda=hp.gae_lambda,
            last_value=last_value,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.as_tensor(np.stack(buf.obs), dtype=torch.float32, device=self.device)
        act_t = torch.as_tensor(np.stack(buf.actions), dtype=torch.float32, device=self.device)
        old_lp = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        n = len(buf)
        indices = np.arange(n)
        policy_loss_acc = 0.0
        value_loss_acc = 0.0
        entropy_acc = 0.0
        updates = 0

        for _ in range(hp.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, hp.minibatch_size):
                mb = indices[start : start + hp.minibatch_size]
                mb_obs = obs_t[mb]
                mb_act = act_t[mb]
                mb_old_lp = old_lp[mb]
                mb_ret = ret_t[mb]
                mb_adv = adv_t[mb]

                new_lp, entropy, value = self.net.evaluate(mb_obs, mb_act)
                ratio = torch.exp(new_lp - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - hp.clip_eps, 1.0 + hp.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(value, mb_ret)
                loss = (
                    policy_loss
                    + hp.value_coef * value_loss
                    - hp.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), hp.max_grad_norm)
                self.optimizer.step()

                policy_loss_acc += float(policy_loss.item())
                value_loss_acc += float(value_loss.item())
                entropy_acc += float(entropy.mean().item())
                updates += 1

        return {
            "policy_loss": policy_loss_acc / max(updates, 1),
            "value_loss": value_loss_acc / max(updates, 1),
            "entropy": entropy_acc / max(updates, 1),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.net.state_dict(),
                "hparams": asdict(self.hparams),
            },
            path,
        )

    def load(self, path: Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.net.load_state_dict(ckpt["state_dict"])


@dataclass
class BacktestReport:
    """测试集回测报告。"""

    total_revenue_yuan: float
    idle_revenue_yuan: float
    profitable_hour_ratio: float
    beat_idle_hour_ratio: float
    n_hours: int
    data_source: str
    alpha_vs_idle_pct: float


def run_policy_on_env(
    agent: PPOAgent,
    env: VPPEnvironment,
    *,
    deterministic: bool = True,
) -> BacktestReport:
    """在指定环境上跑满一个 episode，统计套利胜率指标。"""
    obs, _ = env.reset()
    total_rev = 0.0
    idle_rev = 0.0
    profitable = 0
    beat_idle = 0
    hours = 0

    terminated = False
    while not terminated:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = agent.net.act(obs_t, deterministic=deterministic)
        action_np = action.cpu().numpy().reshape(-1).astype(np.float32)
        obs, reward, terminated, _, info = env.step(action_np)

        price = float(info["settlement_price"])
        idle_pnl = _fallback_hourly_pnl(0.0, price, env.cycle_cost)

        total_rev += float(reward)
        idle_rev += idle_pnl
        if reward > 0.0:
            profitable += 1
        if reward > idle_pnl:
            beat_idle += 1
        hours += 1

    alpha = 0.0
    if abs(idle_rev) > 1e-6:
        alpha = (total_rev - idle_rev) / abs(idle_rev) * 100.0
    elif total_rev > 0:
        alpha = 100.0

    return BacktestReport(
        total_revenue_yuan=total_rev,
        idle_revenue_yuan=idle_rev,
        profitable_hour_ratio=profitable / max(hours, 1),
        beat_idle_hour_ratio=beat_idle / max(hours, 1),
        n_hours=hours,
        data_source=env.data_source,
        alpha_vs_idle_pct=alpha,
    )


def run_policy_on_env_with_trace(
    agent: PPOAgent,
    env: VPPEnvironment,
    *,
    deterministic: bool = True,
) -> tuple[BacktestReport, np.ndarray]:
    """回测并返回逐小时 PnL 序列（供 Web 累计收益曲线）。"""
    obs, _ = env.reset()
    hourly: list[float] = []
    total_rev = 0.0
    idle_rev = 0.0
    profitable = 0
    beat_idle = 0

    terminated = False
    while not terminated:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        with torch.no_grad():
            action, _, _ = agent.net.act(obs_t, deterministic=deterministic)
        action_np = action.cpu().numpy().reshape(-1).astype(np.float32)
        obs, reward, terminated, _, info = env.step(action_np)

        price = float(info["settlement_price"])
        idle_pnl = _fallback_hourly_pnl(0.0, price, env.cycle_cost)
        hourly.append(float(reward))
        total_rev += float(reward)
        idle_rev += idle_pnl
        if reward > 0.0:
            profitable += 1
        if reward > idle_pnl:
            beat_idle += 1

    hours = len(hourly)
    alpha = 0.0
    if abs(idle_rev) > 1e-6:
        alpha = (total_rev - idle_rev) / abs(idle_rev) * 100.0
    elif total_rev > 0:
        alpha = 100.0

    report = BacktestReport(
        total_revenue_yuan=total_rev,
        idle_revenue_yuan=idle_rev,
        profitable_hour_ratio=profitable / max(hours, 1),
        beat_idle_hour_ratio=beat_idle / max(hours, 1),
        n_hours=hours,
        data_source=env.data_source,
        alpha_vs_idle_pct=alpha,
    )
    return report, np.asarray(hourly, dtype=np.float64)


def train_ppo(
    config: ExperimentConfig,
    *,
    hparams: Optional[PPOHyperParams] = None,
    device: str = "cpu",
    save_path: Optional[Path] = None,
    verbose: bool = True,
) -> tuple[PPOAgent, BacktestReport]:
    """
    一键训练 PPO 并在测试集输出回测胜率。

    Returns
    -------
    (agent, test_report)
    """
    import os

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    hp = hparams or PPOHyperParams()
    bundle = build_vpp_data_bundle(config)

    train_env = VPPEnvironment(
        config=config,
        split="train",
        power_mw=config.storage_power_mw,
        capacity_mwh=config.storage_capacity_mwh,
        data_bundle=bundle,
    )
    test_env = VPPEnvironment(
        config=config,
        split="test",
        power_mw=config.storage_power_mw,
        capacity_mwh=config.storage_capacity_mwh,
        data_bundle=bundle,
    )

    obs_dim = train_env.observation_space.shape[0]
    agent = PPOAgent(obs_dim, device=device, hparams=hp)

    if verbose:
        print("\n" + "=" * 72)
        print("  VPP PPO 强化学习训练 — 环境微特征沙盒".center(72))
        print("=" * 72)
        print(f"  数据源     : {bundle.data_source}")
        print(f"  训练样本   : {bundle.n_train} h")
        print(f"  测试样本   : {bundle.n_test} h")
        print(f"  观测维度   : {obs_dim} (特征 {obs_dim - 1} + SOC)")
        print(f"  设备       : {device}")
        print("=" * 72)

    t0 = time.perf_counter()
    for epoch in range(1, hp.total_epochs + 1):
        buf, rollout_ret, _ = agent.collect_rollout(
            train_env, max_steps=hp.rollout_steps
        )
        last_obs = torch.as_tensor(
            train_env._obs() if train_env._t < len(train_env._prices) else np.zeros(obs_dim),
            dtype=torch.float32,
            device=agent.device,
        ).unsqueeze(0)
        with torch.no_grad():
            _, _, last_val = agent.net.forward(last_obs)
        losses = agent.update(buf, last_value=float(last_val.item()))

        if verbose and (epoch == 1 or epoch % max(1, hp.eval_interval) == 0 or epoch == hp.total_epochs):
            test_rep = run_policy_on_env(agent, test_env, deterministic=True)
            print(
                f"[Epoch {epoch:03d}/{hp.total_epochs}] "
                f"rollout_return={rollout_ret:,.0f} | "
                f"policy_loss={losses['policy_loss']:.4f} | "
                f"test_revenue=¥{test_rep.total_revenue_yuan:,.0f} | "
                f"胜率(盈利小时)={test_rep.profitable_hour_ratio:.1%} | "
                f"胜率(优于空仓)={test_rep.beat_idle_hour_ratio:.1%}"
            )

    final_report = run_policy_on_env(agent, test_env, deterministic=True)
    elapsed = time.perf_counter() - t0

    if save_path is not None:
        agent.save(save_path)

    if verbose:
        print("\n" + "-" * 72)
        print("  测试集回测 — PPO 储能套利".center(72))
        print("-" * 72)
        print(f"  累计收益 (PPO)     : ¥{final_report.total_revenue_yuan:,.2f}")
        print(f"  累计收益 (空仓基准) : ¥{final_report.idle_revenue_yuan:,.2f}")
        print(f"  Alpha vs 空仓      : {final_report.alpha_vs_idle_pct:+.2f}%")
        print(f"  盈利小时占比 (胜率) : {final_report.profitable_hour_ratio:.2%}")
        print(f"  优于空仓小时占比   : {final_report.beat_idle_hour_ratio:.2%}")
        print(f"  测试时长           : {final_report.n_hours} h")
        print(f"  训练耗时           : {elapsed:.1f} s")
        print("-" * 72 + "\n")

    return agent, final_report


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="VPP PPO 强化学习训练与储能套利回测胜率评估",
    )
    p.add_argument("--production-db", action="store_true", help="优先 PostgreSQL 生产库")
    p.add_argument("--demo", action="store_true", help="强制合成大样本管线（跳过 PG）")
    p.add_argument(
        "--lookback-hours",
        type=int,
        default=MOCK_SEED_HOURS,
        help=f"回溯小时数（默认 {MOCK_SEED_HOURS}）",
    )
    p.add_argument("--epochs", type=int, default=60, help="PPO 训练轮数")
    p.add_argument("--rollout-steps", type=int, default=512, help="每轮 rollout 步数")
    p.add_argument("--hidden-dim", type=int, default=256, help="MLP 隐层宽度")
    p.add_argument("--lr", type=float, default=3e-4, help="Adam 学习率")
    p.add_argument("--device", type=str, default="cpu", help="cpu / cuda / mps")
    p.add_argument(
        "--save",
        type=Path,
        default=Path("artifacts/vpp_ppo_policy.pt"),
        help="模型权重保存路径",
    )
    p.add_argument(
        "--export-json",
        type=Path,
        default=None,
        help="可选：导出回测 JSON 报告路径",
    )
    p.add_argument(
        "--power-mw",
        type=float,
        default=None,
        help="储能额定功率 MW（默认 run_experiment 常量）",
    )
    p.add_argument(
        "--capacity-mwh",
        type=float,
        default=None,
        help="储能容量 MWh",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    config = ExperimentConfig(
        demo=args.demo,
        production_db=not args.demo,
        lookback_hours=args.lookback_hours,
    )
    if args.power_mw is not None:
        config.storage_power_mw = args.power_mw
    if args.capacity_mwh is not None:
        config.storage_capacity_mwh = args.capacity_mwh

    # 预热数据管线（打印数据源，失败时早抛错）
    frame = load_vpp_market_frame(config)
    print(f"[数据管线] 样本={len(frame)} 行 | 来源={frame.attrs.get('vpp_data_source', '?')}")

    hp = PPOHyperParams(
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        total_epochs=args.epochs,
        rollout_steps=args.rollout_steps,
    )

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，回退 CPU。", file=sys.stderr)
        device = "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS 不可用，回退 CPU。", file=sys.stderr)
        device = "cpu"

    agent, report = train_ppo(
        config,
        hparams=hp,
        device=device,
        save_path=args.save,
        verbose=True,
    )

    if args.export_json is not None:
        args.export_json.parent.mkdir(parents=True, exist_ok=True)
        args.export_json.write_text(
            json.dumps(asdict(report), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"回测报告已写入 {args.export_json}")

    print(f"策略权重已保存 → {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
