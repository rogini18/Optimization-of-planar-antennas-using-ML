# 📡 Optimization of Planar Antennas Using Machine Learning

This project focuses on optimizing the design of **planar MIMO antennas** for 5G mmWave applications using **machine learning algorithms**. Traditional antenna design methods rely heavily on simulation and trial-and-error, which are time-consuming and computationally expensive. In this work, models such as **Random Forest, KNN, and SVM** are trained on simulation data to predict and optimize antenna parameters like **gain**, **bandwidth**, **VSWR**, and **efficiency**.

The goal is to reduce development time and improve design accuracy by leveraging data-driven methods. The final optimized antenna was fabricated and tested, with performance closely matching simulation predictions.

---

## 🖼️ Image Descriptions

This repository contains the key simulation results and fabrication/testing images:

| Description                                         | Image                                  |
|----------------------------------------------------|--------------------------------------|
| **Return Loss (S11)** at 25.1 GHz – demonstrates strong resonance with return loss < -10 dB. | ![Return Loss S11](return_loss_s11_25ghz.jpg) |
| **VSWR Plot** at 25.1 GHz – a value of 1.63 shows efficient power transfer. | ![VSWR Plot](vswr_25ghz.jpg)          |
| **3D Gain Pattern** – shows the directional radiation and strength at 25.1 GHz. | ![3D Gain Pattern](gain_3d_25ghz.jpg)         |
| **Radiation Efficiency** – shows how much input power is effectively radiated (≈ 39.3%). | ![Radiation Efficiency](efficiency_radiation.jpg) |
| **Total Efficiency** – overall antenna efficiency including losses (≈ 36.8%). | ![Total Efficiency](efficiency_total.jpg) |
| **Fabricated Antenna – Front View** – shows the copper patch layout of the antenna. | ![Fabricated Antenna Front](fabricated_front.jpg) |
| **Antenna Front View (Design)** – simulation/model view showing the layout before fabrication. | ![Antenna Front View](antenna_front_view.jpg) |
| **Fabricated Antenna – Back View** – ground plane and physical structure of the prototype. | ![Fabricated Antenna Back](fabricated_back.jpg) |

---

## 👩‍💻 Technologies Used

- CST Microwave Studio (for simulation)
- Python (NumPy, Pandas, Scikit-learn, Matplotlib)
- Machine Learning Algorithms: Random Forest, K-Nearest Neighbors, Support Vector Machines

---

## 👥 Contributors

- Ranjini D  
- Rethina Pradhamesh R  
- Rithika R  
- Rogini E  

**Academic Year:** 2024–25  
**Institution:** SRM Valliammai Engineering College  
**Department:** Electronics and Communication Engineering  
**Supervisor:** Dr. S. Ramesh  
