## DFANet

**T. Mao, J. Liu, J. Yang and Z. Wu, "A Self-Distillation-Based Multimodal Feature Alignment Network for Hyperspectral Image and LiDAR Classification," in IEEE Geoscience and Remote Sensing Letters, vol. 22, pp. 1-5, 2025, Art no. 5504805, doi: 10.1109/LGRS.2025.3570391.**

If you have any questions, please contact us. Email: <a href="mao19980704@163.com">mao19980704@163.com</a>

## Introduction

The joint classification of hyperspectral image (HSI) and light detection and ranging (LiDAR) data seeks to provide a more comprehensive characterization of target objects. Multi-modal data possess distinct semantic structures in both spectral and spatial dimensions, making efficient feature complementarity and redundancy elimination crucial. To this end, we propose a self-Distillation-based multimodal Feature Alignment Network (DFANet), which employs two branches to capture spectral and spatial similarities respectively, and integrates structural discriminative information from LiDAR at two stages for more effective multimodal data integration. The network comprises three main components: a Feature Alignment Fusion Module (FAFM), an Offset Attention Module (OAM), and a self-distillation mechanism. Specifically, the FAFM guides feature alignment through channel-assimilative mapping of multimodal data. The OAM addresses boundary patch classification challenges by learning offset weights of reference points. The self-distillation mechanism filters out irrelevant information during feature alignment by enhancing the coordination between high-level and low-level features.

This is a total architecture of our network:

![DFANet Architecture](https://github.com/andyleader/DFANet/blob/48ef7ecc94f559d303a362203dc9f9ae971c2492/image.png)

## Dataset
1. Trento dataset: https://github.com/A-Piece-Of-Maple/TrentoDateset
2. Muufl dataset: https://github.com/GatorSense/MUUFLGulfport
3. Houston 2013 dataset: https://ieee-dataport.org/documents/hyperspectral-data-0
