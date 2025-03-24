# [WACV 2025] Learning Anatomy-Disease Entangled Representation

This repository provides a PyTorch implementation of the [Learning Anatomy-Disease Entangled Representation](https://openaccess.thecvf.com/content/WACV2025/papers/Haghighi_Learning_Anatomy-Disease_Entangled_Representation_WACV_2025_paper.pdf) which is published in WACV 2025 (main conference).

Human experts demonstrate proficiency not only in disentangling anatomical structures from disease conditions but also in intertwining anatomical and disease information to accurately diagnose a variety of disorders. However, deep learning models, despite their prowess in acquiring intricate representation, often struggle to learn representation where distinct semantic aspects of the data (both anatomy and pathology) are entangled, particularly in medical images, which present a rich array of anatomical structures and potential pathological conditions. We envision that a deep model, when trained to comprehend medical images akin to human perception, would offer powerful representation with higher generalizability, robustness, and interpretability. To realize this vision, we have developed LeADER, a framework for learning anatomy-disease entangled representation from medical images. We have trained LeADER on a large corpus of around 1M chest radiographs gathered from 10 public datasets.

<br/>
<p align="center"><img width="90%" src="images/method_idea.png" /></p>
<br/>
