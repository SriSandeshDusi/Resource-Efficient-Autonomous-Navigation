# Code Availability & Intellectual Property Notice

This repository contains reference implementations and demonstration artifacts corresponding to the research manuscript:

> **"A Resource-Efficient Hierarchical Framework Utilizing Theta* and NanoRL for Autonomous Navigation"**  
> *Under Review / Pending Patent Application*  
> **Authors:** Dusi Venkata Lakshmana Sri Sandesh, Dr. Ajay Pratap  
> **Affiliation:** Centre For Development of Drone Related Technologies (CDDRT), Indian Institute of Technology (BHU), Varanasi  

---

## 🔒 Confidentiality & Release Schedule

1. **Academic Peer Review & Patent Filing:**
   The full algorithmic framework, complete neural network checkpoints, procedural generation seeds, and customized hardware drivers are currently under double-blind peer review and provisional patent prosecution. To respect conference submission guidelines and protect pending intellectual property rights, the complete production codebase and end-to-end dataset cannot be publicly distributed at this time.

2. **Provided Reference Modules:**
   To facilitate reproducibility and scientific transparency during review, this repository provides self-contained reference implementations of the core modules:
   * `theta_star.py`: Any-angle global path planner with Euclidean line-of-sight calculation.
   * `qbot_env.py`: 2D Gymnasium environment core with the 76-dimensional observation tensor and policy-invariant reward shaping formulation.
   * `curriculum_callback.py`: Two-phase dynamic curriculum callback featuring exploration noise variance freezing ($\sigma = 0.20$).
   * `ros_bridge.py`: Hardware abstraction layer and 10 Hz ROS inference bridge for edge deployment.

3. **Full Release Timeline:**
   The complete codebase, trained model weights, evaluation logs, and ROS launch stacks will be publicly unlocked following the formal publication of the paper and completion of the patent process.

---

## 📩 Academic Inquiries & Collaborations

For legitimate academic evaluation, review inquiries, or research collaboration requests under standard academic non-disclosure agreements, please contact:

* **Sri Sandesh Dusi:** [srisandeshdusi@gmail.com](mailto:srisandeshdusi@gmail.com)
* **CDDRT Lab, IIT (BHU), Varanasi**
