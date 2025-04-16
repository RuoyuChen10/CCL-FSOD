import numpy as np

original_attr = []

attr = np.loadtxt("attribute_representation.txt")
attr = (attr>0).astype(int)

CLASS_NAME = ["aeroplane","bicycle","bird","boat","bottle",
             "bus","car","cat","chair","cow",
             "diningtable","dog","horse","motorbike","person",
             "pottedplant","sheep","sofa","train","tvmonitor"]

CLASS_NAME_SPLIT1 = np.array(["aeroplane","bicycle","boat","bottle","car",
                              "cat","chair","diningtable","dog","horse",
                              "person","pottedplant","sheep","train","tvmonitor",
                              "bird","bus","cow","motorbike","sofa"])

CLASS_NAME_SPLIT2 = np.array(["bicycle","bird","boat","bus","car",
                              "cat","chair","diningtable","dog","motorbike","person","pottedplant","sheep","train","tvmonitor",
                              "aeroplane","bottle","cow","horse","sofa"])

CLASS_NAME_SPLIT3 = np.array(["aeroplane","bicycle","bird","bottle","bus",
                              "car","chair","cow","diningtable","dog",
                              "horse","person","pottedplant","train","tvmonitor",
                              "boat","cat","motorbike","sheep","sofa"])

attr_new = []

for category_name in CLASS_NAME_SPLIT1:
    attr_new.append(attr[CLASS_NAME.index(category_name)])

np.save(
    "attribute-representation-split1", np.array(attr_new)
)

