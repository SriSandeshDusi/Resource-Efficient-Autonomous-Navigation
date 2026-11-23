# Resource-Efficient Hierarchical Autonomous Navigation via Theta* and NanoRL

[![ROS Noetic](https://img.shields.io/badge/ROS-Noetic-blue.svg)](http://wiki.ros.org/noetic)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-v0.29-orange.svg)](https://gymnasium.farama.org/)
[![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20%7C%20Quanser%20QBot-76b900.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/)
[![Paper Status](https://img.shields.io/badge/Paper-Under%20Review-yellow.svg)](#-confidentiality--code-availability-notice)
[![Patent](https://img.shields.io/badge/Patent-Pending-informational.svg)](#-confidentiality--code-availability-notice)

> **Research Internship Project** | *Centre For Development of Drone Related Technologies (CDDRT), IIT (BHU), Varanasi*  
> **Author:** Dusi Venkata Lakshmana Sri Sandesh  
> **Supervisor:** Dr. Ajay Pratap  

---

## ⚠️ Confidentiality & Code Availability Notice

> **IMPORTANT:**  
> The methodology, proprietary training pipelines, and complete hardware deployment stack associated with this project are currently **under review for academic conference publication and pending patent filing**. 
>
> To protect intellectual property and adhere to conference review guidelines, **only reference implementations of the core mathematical algorithms (3–4 key modules) and demonstration media are publicly hosted in this repository**. 
>
> Please see [CODE_AVAILABILITY.md](CODE_AVAILABILITY.md) for full disclosure, release timeline, and academic inquiry details.

---

## 💡 Project Idea

Deploying autonomous mobile robots in active, human-centric environments (such as hospitals, warehouses, and narrow indoor corridors) presents a fundamental trade-off: **global trajectory optimality vs. instantaneous reactive obstacle avoidance**, constrained by the severe computational and memory limits of embedded edge computers.

This project implements a **resource-efficient hierarchical navigation framework** tailored for embedded mobile robots (Quanser QBot on an NVIDIA Jetson):

* **Global Path Planner ($\text{Theta}^*$):** An any-angle path planner executed once at episode start on the static occupancy grid. By using continuous line-of-sight checks, $\text{Theta}^*$ produces smooth, Euclidean shortest-path trajectories, eliminating the jagged turns and unnecessary stops of standard $\text{A}^*$.
* **Local Reactive Controller (NanoRL / PPO):** A lightweight Actor-Critic neural network operating at 10 Hz over a low-dimensional **76-D state vector** (72 LiDAR rays + 2 relative waypoint tracking values + 2 robot velocities). It reactively steers around unexpected, non-cooperative dynamic obstacles (pedestrians) while adhering to the global route.
* **Curriculum Learning with Variance Freezing:** A two-phase training strategy (Phase 1: static maze corridors $\rightarrow$ Phase 2: dynamic pedestrians). Clamping/freezing the policy exploration standard deviation ($\sigma = 0.20$) upon entering Phase 2 prevents catastrophic forgetting of static navigation skills.

---

## 🎬 Visual Showcase & Demonstrations

### 1. System Pipeline & Neural Architecture
The hierarchical framework decouples long-term planning from high-frequency reactive motor control:

![Architecture Flowchart](assets/flowchart.png)

---

### 2. Dynamic Evasion in Simulation
The local NanoRL agent reactively swerves to bypass moving obstacles without losing progress toward the global $\text{Theta}^*$ waypoint:

![Dynamic Evasion Sequence](assets/dynamic_evasion_sequence.png)

---

### 3. Real-Time Hardware Resource Profiling
Profiled on onboard edge hardware executing continuous 10 Hz inference:

![Hardware Profiling Plot](assets/hardware_profiling_plot.png)

---

### 4. Working Prototype Video
> 📹 **Live Demonstration:**  
> A live prototype demonstration of the autonomous QBot navigating real-world corridors was presented on **Institute Day, IIT (BHU)**.  
>
> *(To view the full working demonstration video, see [`assets/qbot_live_demo.mp4`](assets/) or contact the authors).*

---

## 🛠️ Frameworks & Hardware Used

* **Reinforcement Learning:** PyTorch, Stable-Baselines3 (Proximal Policy Optimization)
* **Simulation:** Custom Gymnasium 2D raycasting engine (>1,000 steps/sec on CPU)
* **Global Path Planning:** Any-Angle $\text{Theta}^*$, $\text{A}^*$
* **Robotics Middleware:** ROS Noetic (Python / C++)
* **Target Hardware:** Quanser QBot with onboard NVIDIA Jetson (ARM64), 360° 2D LiDAR, Wheel Encoders
* **Development Environment:** Antigravity, VS Code, Linux / Ubuntu Focal

---

## 📊 Key Results

Benchmarked across **500 evaluation episodes** against classical and learning baselines:

* **Success Rate:** Achieved **79.0% overall success** (93.0% static / 65.0% dynamic with non-cooperative moving obstacles), outperforming classical $\text{A}^* + \text{DWA}$ and Artificial Potential Fields (APF), which suffered from the *freezing robot problem*.
* **Ultra-Low Compute Footprint:** Sustained continuous 10 Hz control on the NVIDIA Jetson using only **~6% single-core CPU utilization**.
* **Minimal Memory:** Kept memory usage stable at **~320 MB** (well under the 512 MB embedded limit), leaving ample headroom for simultaneous onboard processes like SLAM.
* **Rapid Training:** 3,000,000 environment interaction steps trained in **~2 hours** on a standard CPU.

---

## 📂 Included Reference Code Files

As per the IP notice, this repository provides 4 self-contained reference files illustrating the core architecture:

| File | Description |
| :--- | :--- |
| [`theta_star.py`](theta_star.py) | Any-angle $\text{Theta}^*$ global path planner with continuous Euclidean line-of-sight calculation. |
| [`qbot_env.py`](qbot_env.py) | Custom Gymnasium 2D environment featuring the 76-D state contract and policy-invariant reward shaping. |
| [`curriculum_callback.py`](curriculum_callback.py) | Two-phase curriculum manager that freezes policy exploration noise ($\sigma = 0.20$) in dynamic environments. |
| [`ros_bridge.py`](ros_bridge.py) | 10 Hz ROS Noetic bridge node interfacing the policy with physical Quanser QBot sensor and actuator topics. |

---

## 📜 Citation

If you reference this work in your research, please cite:

```bibtex
@inproceedings{dusi2026nanorl,
  title={A Resource-Efficient Hierarchical Framework Utilizing Theta* and NanoRL for Autonomous Navigation},
  author={Dusi, Venkata Lakshmana Sri Sandesh and Pratap, Ajay},
  booktitle={Centre for Development of Drone Related Technologies (CDDRT)},
  year={2026},
  organization={IIT (BHU), Varanasi},
  note={Under review / Patent pending}
}
```

---

## 👨‍💻 Author & Contact
* **Dusi Venkata Lakshmana Sri Sandesh**  
  *Electrical Engineering, Indian Institute of Technology (BHU), Varanasi*  
  * **Email:** [srisandeshdusi@gmail.com](mailto:srisandeshdusi@gmail.com)  
  * **LinkedIn:** [linkedin.com/in/srisandeshdusi](https://linkedin.com/in/srisandeshdusi)  
  * **GitHub:** [github.com/srisandeshdusi](https://github.com/srisandeshdusi)
