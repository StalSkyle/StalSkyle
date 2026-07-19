# Hi there! I'm Rodion

### Machine Learning & Data Science Engineer
Welcome to my GitHub profile! I am a passionate machine learning engineer focused on deep learning architectures, computer vision, computational methods, data analysis and robust tabular forecasting/classification pipelines. I enjoy solving complex structural and scientific problems through data-driven approaches.

---

## 🛠️ Skills & Technologies
- **Languages:** Python, C++, SQL
- **Machine Learning & Deep Learning:** PyTorch, Scikit-learn, boosting libraries, time series analysis
- **Computer Vision & Multimodal:** YOLO, BLIP, LLaVA, Mask2Former, Torchvision
- **Data Engineering & Analysis:** Pandas, NumPy, Matplotlib, Seaborn
- **Scientific Computing:** Numerical Methods, Finite Difference Schemes, PINNs, SciPy
- **Misc** Git, Linux, Excel, Jupyter, Google Colab, Markdown/LaTeX

---

## 👤 About Me
* 🔭 **Current Focus:** Student at MEPhI (4th year)
* 📚 **Academic Interests:** Applied maths: probability theory, statistics, PDEs
* 💼 **Open to:** [job applications; research projects]
* ✉️ **Contact:** [s1rodion123@gmail.com] | [tg: @StalSkyle]

---

## 🌟 Featured Projects

### 🚗 Car Scene Captioning | *Computer Vision & Multimodal Pipeline*
* **Situation:** When vehicles are lost or misplaced in areas with weak or unavailable GPS signals (such as dense urban environments, underground parking lots, or under bridges), user-submitted photos are used for localization. However, manual human inspection of these images is slow, and finding specific visual landmarks can be highly inefficient.
* **Task:** Build an automated end-to-end multimodal vision pipeline (`autocaption`) to analyze photos, detect vehicle presence, normalize orientation, extract contextual environmental elements, and generate clear text descriptions of the surroundings while purposefully omitting the vehicle itself and the sky.
* **Action:** Engineered a modular Python pipeline containing 8 specialized components: `ImageLoader` for flexible input handling; `ImageRotator` (ResNet50) for auto-orientation; `CarDetector` (YOLOv8) to filter vehicle presence; `ObjectExtractor` (YOLOv8 trained on Open Images v7) and `SceneExtractor` (ResNet18 trained on BDD100K) for comprehensive feature mining. Implemented a panoptic segmentation masking layer using `Mask2Former` to perform inpainting, removing the car and sky to avoid textual distraction. Applied `BLIP` and `BLIP-VQA` for conditional text generation, coupled with `LLaVA` (`TextGenerator`) for final description synthesis.
* **Result:** Developed a highly functional, production-ready inference library capable of processing both local images and URLs. The system successfully extracts key visual identifiers (e.g., "near a playground", "under a bridge") and suppresses irrelevant details, significantly accelerating vehicle search operations in zero-GPS zones.

### 🤖 HireLander AI | *Automated AI Recruitment Platform*
* **Situation:** HR teams spend hundreds of hours manually screening resumes, managing scheduling conflicts, and conducting initial interviews. This heavy workload leads to cognitive fatigue, increasing subjectivity and the risk of missing exceptional candidates.
* **Task:** Develop a viable, high-performance automated recruitment assistant during a fast-paced hackathon capable of filtering vacancies, interviewing candidates autonomously, and detecting cheating.
* **Action:** Engineered an end-to-end prototype featuring automated CV/vacancy parsing and semantic ranking pipelines. Designed a multimodal AI interviewing agent capable of reading coding tasks, listening to candidates, and conversing dynamically in real-time. Integrated an automated proctoring subsystem to monitor integrity during coding phases. Successfully managed technical debt and code optimization within severe hackathon time constraints to ensure project viability.
* **Result:** Delivered a fully operational multi-page web application featuring custom interfaces (Main, Login, Profile, Screening, Ranking, and Coding pages) that effectively eliminates early-stage screening bias and substantially reduces manual HR overhead.

### 🌌 Stellar Variability Analysis | *Advanced Machine Learning & Class Imbalance*
* **Situation:** Processing large-scale astronomical datasets involves managing millions of stellar objects to detect and classify variable stars. The task is heavily hindered by extreme class imbalance—millions of common stars eclipse or mask rare classes, such as ERUPTIVE stars, which represent a tiny fraction of the dataset.
* **Task:** Build a high-accuracy two-stage machine learning workflow: a binary classifier to accurately isolate variable stars from static objects, and a robust multiclass model to categorize four major variability types (Eclipsing, Pulsating, Rotating, and Eruptive).
* **Action:** Implemented an intensive data preprocessing pipeline featuring multicollinearity filtering, t-SNE dimensionality reduction, and `RobustScaler` transformations. Built a binary ensemble via scikit-learn's `StackingClassifier` integrating CatBoost, XGBoost, and LightGBM with a Logistic Regression meta-model, utilizing SMOTE for class balancing. For the multiclass phase on the 9-million-row VSX Parquet dataset, engineered features like `has_period` and `amplitude`. Designed a combined model framework consisting of a primary cost-sensitive CatBoost classifier (`auto_class_weights='Balanced'`) and a dedicated ERUPTIVE class detector optimized via Optuna (50 trials) with a strict probability threshold (`>0.95`).
* **Result:** The binary stacking model achieved **0.94 Accuracy** and **0.85 Recall** ($F_1$-score: 0.74). The final comprehensive multiclass framework delivered an overall **Accuracy of 0.94** and a **Macro $F_1$-score of 0.87**, dramatically improving the rare ERUPTIVE class $F_1$-score to **0.66** and proving highly resilient to severe data imbalance.

### 🧪 Physics-Informed Neural Networks (PINN) | *Computational Mathematics*
* **Situation:** Solving complex population dynamics like the Lotka-Volterra predator-prey system using classical numerical methods yields discrete trajectories but lacks the continuous, differentiable, and parameter-adaptable nature provided by deep learning frameworks.
* **Task:** Implement a Physics-Informed Neural Network (PINN) to solve the Cauchy problem for the Lotka-Volterra system of non-linear differential equations and conduct a rigorous numerical study on how parametric shifts impact population stability.
* **Action:** Built a specialized PyTorch neural network leveraging sinusoidal activation functions to approximate continuous population paths $x(t)$ and $y(t)$ over time. Formulated a custom composite loss function combining the residual of the Lotka-Volterra differential equations and the initial condition constraints ($x_0=40.0, y_0=9.0$). Trained the network using the Adam optimizer with a dynamic learning rate schedule. Validated the network's accuracy against classical 4th-5th order Runge-Kutta integration (`solve_ivp` via SciPy) and executed sensitivity analyses across varying system coefficients ($\alpha, \beta, \gamma, \delta$).
* **Result:** The PINN model successfully approximated the continuous system dynamics, showing tight convergence and low $L_{\infty}$ error compared to classical integration. The parametric study successfully mapped phase space trajectories, identified critical thresholds leading to sudden population collapse/extinction, and verified system behaviors around stable and unstable focal states.

### 🦋 Kaggle Butterflies Classification | *Fine-Grained Image Classification*
* **Situation:** Automated fine-grained biological classification across highly visually similar species (such as 50 distinct types of butterflies) is extremely challenging due to subtle variations in wing patterns, spots, and colors, compounded by a limited training pool (4,955 images).
* **Task:** Construct a deep learning vision classifier to maximize predictive accuracy on a competitive Kaggle dataset containing 50 fine-grained butterfly classes.
* **Action:** Conducted benchmarking experiments comparing basic CNNs built from scratch against multiple Transfer Learning backbones (ResNet, VGG, EfficientNet). Selected **EfficientNet-B3** as the optimal architecture and performed full end-to-end fine-tuning of all layers to capture intricate wing textures. Developed a comprehensive preprocessing and augmentation pipeline using `Albumentations` (including resizing to 224x224, ImageNet normalization, horizontal flips, random rotations, and brightness/contrast adjustments). Evaluated deeper architectures (EfficientNet-B5) and regularizations (weight decay, learning rate schedulers, model ensembling) to control overfitting.
* **Result:** Achieved a top-tier validation accuracy of **0.97340** on the Kaggle competition leaderboard. Proved that full layer unfreezing combined with tailored spatial/color augmentations delivers superior generalization for fine-grained classification tasks compared to frozen features or model ensembles.

---

## 📈 GitHub Stats
<p align="left">
  <img src="https://github-readme-stats.vercel.app/api?username=StalSkyle&show_icons=true&theme=radical" alt="GitHub Stats" width="400"/>
  <img src="https://github-readme-stats.vercel.app/api/top-langs/?username=StalSkyle&layout=compact&theme=radical" alt="Top Langs" width="300"/>
</p>

*Feel free to explore my repositories, open an issue, or reach out if you'd like to collaborate on exciting ML projects!*
