# "aeroplane" "bottle" "cow" "horse" "sofa"
for novel in "cow"
do
    python -m Visualize.visualize_cam \
    --config-file configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_10shot.yaml \
    --input datasets/VOC/${novel} \
    --output /home/cry/J-20/fewshot_object_detection_results/split2-10-exp/${novel} \
    --opts \
    MODEL.WEIGHTS checkpoints/voc/ExAu/split2-10shot/model_0006399.pth \
    MODEL.DEVICE cuda \
    MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.2
done

for novel in "cow"
do
    python -m Visualize.visualize_cam --config-file configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_10shot.yaml --input datasets/VOC/${novel} --output /home/cry/J-20/fewshot_object_detection_results/split2-10/${novel} --opts MODEL.WEIGHTS checkpoints/voc/ckpt/split2-10/model_final.pth MODEL.DEVICE cuda MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.2
done

for novel in "cow"
do
    python -m Visualize.visualize_cam --config-file configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_10shot.yaml --input datasets/VOC/${novel} --output /home/cry/J-20/fewshot_object_detection_results/split2-10-00/${novel} --opts MODEL.WEIGHTS checkpoints/voc/ablation/split2-10-00.pth MODEL.DEVICE cuda MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.2
done

for novel in "cow"
do
    python -m Visualize.visualize_cam --config-file configs/PascalVOC-detection/split2/faster_rcnn_R_101_FPN_ft_all2_10shot.yaml --input datasets/VOC/${novel} --output /home/cry/J-20/fewshot_object_detection_results/split2-10-01/${novel} --opts MODEL.WEIGHTS checkpoints/voc/ablation/split2-10-01.pth MODEL.DEVICE cuda MODEL.ROI_HEADS.SCORE_THRESH_TEST 0.2
done