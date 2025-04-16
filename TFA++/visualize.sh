# python -m Visualize.demo --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_10shot_v3.yaml --input datasets/VOC2007/JPEGImages/002695.jpg --opts MODEL.WEIGHTS checkpoints/voc_k/cl_k_c_7.pth MODEL.DEVICE cuda

# 10 shot

# for novel in "bird" "bus" "cow" "motorbike" "sofa"
# do
# mkdir vis_result/ours
# mkdir vis_result/ours/${novel}
# python -m demo.demo --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_10shot_v3.yaml --input datasets/VOC/${novel}/*.jpg --output vis_result/ours/${novel} --confidence-threshold 0.5 --opts MODEL.WEIGHTS checkpoints/voc/ckpt/split1-10/model_final.pth
# done

# for novel in "bird" "bus" "cow" "motorbike" "sofa"
#     do
#     for method in "TFA" "FSCE"
#         do
#         mkdir vis_result/${method}
#         mkdir vis_result/${method}/${novel}
#         python -m demo.demo --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_10shot_v3.yaml --input datasets/VOC/${novel}/*.jpg --output vis_result/${method}/${novel} --confidence-threshold 0.5 --opts MODEL.WEIGHTS checkpoints/voc/${method}/split1-10/model_final.pth
#         done
#     done

## 1-1 shot

# for novel in "bird" "bus" "cow" "motorbike" "sofa"
# do
# mkdir vis_result/ours
# mkdir vis_result/ours/${novel}
# python -m demo.demo --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_10shot_v3.yaml --input datasets/VOC/${novel}/*.jpg --output vis_result/ours/${novel} --confidence-threshold 0.7 --opts MODEL.WEIGHTS checkpoints/voc/ckpt/split1-1/model_final.pth
# done

# for novel in "bird" "bus" "cow" "motorbike" "sofa"
#     do
#     for method in "TFA" "FSCE"
#         do
#         mkdir vis_result/${method}
#         mkdir vis_result/${method}/${novel}
#         python -m demo.demo --config-file configs/PascalVOC-detection/split1/faster_rcnn_R_101_FPN_ft_all1_10shot_v3.yaml --input datasets/VOC/${novel}/*.jpg --output vis_result/${method}/${novel} --confidence-threshold 0.5 --opts MODEL.WEIGHTS checkpoints/voc/${method}/split1-1/model_final.pth
#         done
#     done


# 2-5  "aeroplane", "bottle", "cow", "horse", "sofa"

# for novel in "aeroplane" "horse"
# do
# mkdir vis_result/ours
# mkdir vis_result/ours/${novel}
# python -m demo.demo --config-file configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_5shot.yaml --input datasets/VOC/${novel}/*.jpg --output vis_result/ours/${novel} --confidence-threshold 0.6 --opts MODEL.WEIGHTS checkpoints/voc/ckpt/split2-5/model_final.pth
# done

# for novel in "aeroplane" "horse"
#     do
#     for method in "TFA" "FSCE"
#         do
#         mkdir vis_result/${method}
#         mkdir vis_result/${method}/${novel}
#         python -m demo.demo --config-file configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_5shot.yaml --input datasets/VOC/${novel}/*.jpg --output vis_result/${method}/${novel} --confidence-threshold 0.5 --opts MODEL.WEIGHTS checkpoints/voc/${method}/split2-5/model_final.pth
#         done
#     done

for novel in "horse"
    do
    for method in "TFA" "FSCE"
        do
        mkdir vis_result/${method}
        mkdir vis_result/${method}/${novel}
        python -m demo.demo --config-file configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_5shot.yaml --input datasets/VOC/${novel}/*.jpg --output vis_result/${method}/${novel} --confidence-threshold 0.2 --opts MODEL.WEIGHTS checkpoints/voc/${method}/split2-5/model_final.pth
        done
    done

# 3-10

# for novel in "boat" "cat" "sheep"
# do
# mkdir vis_result/ours
# mkdir vis_result/ours/${novel}
# python -m demo.demo --config-file configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_10shot.yaml --input datasets/VOC/${novel}/*.jpg --output vis_result/ours/${novel} --confidence-threshold 0.6 --opts MODEL.WEIGHTS checkpoints/voc/ckpt/split3-10/model_final.pth
# done

# for novel in "boat" "cat" "sheep"
#     do
#     for method in "TFA" "FSCE"
#         do
#         mkdir vis_result/${method}
#         mkdir vis_result/${method}/${novel}
#         python -m demo.demo --config-file configs/PascalVOC-detection/split3/faster_rcnn_R_101_FPN_ft_all3_10shot.yaml --input datasets/VOC/${novel}/*.jpg --output vis_result/${method}/${novel} --confidence-threshold 0.2 --opts MODEL.WEIGHTS checkpoints/voc/${method}/split3-10/model_final.pth
#         done
#     done


