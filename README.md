# [SIGMA: Semantic-complete Graph Matching For Domain Adaptive Object Detection](https://arxiv.org/pdf/2203.06398.pdf)

By [Wuyang Li](https://wymancv.github.io/wuyang.github.io/)

This is the official implementation of SIGMA: Semantic-complete Graph Matching For Domain Adaptive Object Detection (CVPR22). Welcome to follow our previous work [SCAN](https://github.com/CityU-AIM-Group/SCAN) (AAAI'22 oral), which is the foundation of this work.

Codes and well-trained models will be released soon.


## Contact

Wuyang Li: wuyangli2-c@my.cityu.edu.hk


## Abstract

Domain Adaptive Object Detection (DAOD) leverages a labeled source domain to learn an object detector generalizing to a novel target domain free of annotations. Recent advances align class-conditional distributions through narrowing down cross-domain prototypes (class centers). Though great success, these works ignore the significant within-class variance and the domain-mismatched semantics within the training batch, leading to a sub-optimal adaptation. To overcome these challenges, we propose a novel SemantIc-complete Graph MAtching (SIGMA) framework for DAOD, which completes mismatched semantics and reformulates the adaptation with graph matching. Specifically, we design a Graph-embedded Semantic Completion module (GSC) that completes mismatched semantics through generating hallucination graph nodes in missing categories. Then, we establish cross-image graphs to model class-conditional distributions and learn a graph-guided memory bank for better semantic completion in turn. After representing the source and target data as graphs, we reformulate the adaptation as a graph matching problem, i.e., finding well-matched node pairs across graphs to reduce the domain gap, which is solved with a novel Bipartite Graph Matching adaptor (BGM). In a nutshell, we utilize graph nodes to establish semantic-aware node affinity and leverage graph edges as quadratic constraints in a structure-aware matching loss, achieving fine-grained adaptation with a node-to-node graph matching. Extensive experiments demonstrate that our method outperforms existing works significantly.

![image](https://github.com/CityU-AIM-Group/SIGMA/blob/main/overall.png)
