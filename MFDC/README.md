# 🐳 MFDC backbone

## 

## 🔥 Results on VOC Benchmark

Method |Paper Year | | | Split-1 | | | | | Split-2 | | | | | Split-3 | | |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|[MFDC](https://github.com/shuangw98/MFDC)|ECCV 22|63.4|66.3|67.7|69.4|68.1|42.1|46.5|53.4|55.3|53.8|56.1|58.3|59.0|62.2|63.7|
|[NIFF](https://openaccess.thecvf.com/content/CVPR2023/papers/Guirguis_NIFF_Alleviating_Forgetting_in_Generalized_Few-Shot_Object_Detection_via_Neural_CVPR_2023_paper.pdf)|CVPR 23|63.5|67.2|**68.3**|**71.1**|69.3|37.8|41.9|53.4|**56.0**|53.5|55.3|60.5|61.1|63.7|63.9|
|KD|ECCV 22|58.2|62.5|65.1|68.2|67.4|37.6|45.6|52.0|54.6|53.2|53.8|57.7|58.0|62.4|62.2|
|[Norm-VAE](https://openaccess.thecvf.com/content/CVPR2023/papers/Xu_Generating_Features_With_Increased_Crop-Related_Diversity_for_Few-Shot_Object_Detection_CVPR_2023_paper.pdf)|CVPR 23|62.1|64.9|67.8|69.2|67.5|39.9|46.8|**54.4**|54.2|53.6|58.2|60.3|61.0|64.0|65.5|
|[FPD](https://github.com/wangchen1801/FPD)|AAAI 24|46.5|62.3|65.4|68.2|69.3|32.2|43.6|50.3|52.5|56.1|43.2|53.3|56.7|62.1|64.1|
|[SMILe](https://github.com/amajee11us/SMILe-FSOD)|ECCV 24|40.9|-|-|59.7|62.0|26.5|-|-|49.5|52.3|42.6|-|-|56.4|61.4|
|[T-GSEL](https://link.springer.com/article/10.1007/s11263-024-02199-0)|IJCV 25|50.4|63.6|61.9|68.6|67.3|31.3|32.9|43.6|47.9|53.9|41.2|49.8|54.1|62.1|61.9|
|[SNIDA](https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_SNIDA_Unlocking_Few-Shot_Object_Detection_with_Non-linear_Semantic_Decoupling_Augmentation_CVPR_2024_paper.pdf)|CVPR 24|59.3|60.8|64.3|65.4|65.6|35.2|40.8|50.2|54.6|50.0|51.6|52.4|55.9|58.5|62.6|
|Ours|TPAMI 25|**64.9**|**67.3**|67.8|70.5|**70.3**|**42.9**|**48.4**|53.9|55.5|53.9|**59.4**|**62.0**|**61.2**|**64.8**|**65.8**|

You can download checkpoints from https://huggingface.co/RuoyuChen/CCL-FSOD/tree/main/MFDC-checkpoints/voc