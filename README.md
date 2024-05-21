# Teacher privileged distillation: How to deal with imperfect teachers?

[python-img]: https://img.shields.io/badge/Made%20with-Python-blue
[ama-img]: https://img.shields.io/badge/Ask%20me-anything-yellowgreen
[wip-img]:https://img.shields.io/badge/Work%20in%20progress-8A2BE2

![Made with Python][python-img]
![Ask me anything][ama-img]
![Work in progress][wip-img]

This repository contains the code for the paper _"Teacher privileged distillation: How to deal with imperfect teachers?"_. The paradigm of learning using privileged information leverages privileged features present at training time, but not at prediction, as additional training information. The privileged learning process is addressed through a knowledge distillation perspective: information from a teacher learned with privileged features is transferred to a student composed exclusively of regular features. While most approaches assume perfect knowledge for the teacher, it can commit mistakes. Assuming that, we propose a novel privileged distillation framework with a double contribution: an adaptation of the cross-entropy loss for imperfect teachers and a correction of teacher misclassifications. Its effectiveness is empirically demonstrated on datasets with imperfect teachers, significantly enhancing the performance of state-of-the-art frameworks. Furthermore, necessary but not sufficient conditions for successful privileged learning are presented, along with a dataset categorization based on the information provided by the privileged features.


<img width="500" alt="Screenshot 2024-05-21 at 17 47 07" src="https://github.com/mariomartgarcia/TPD/assets/63496191/e9fb969f-dc0e-437b-8a7f-599717f47da8">

## Content

- **code:**
  - `lrplus.py`. Main file with LRIT+ and LR+ algorithms.

## Contact

Mario Martínez García - mmartinez@bcamath.org


## Citation

The corresponding BiBTeX citation is given below:
