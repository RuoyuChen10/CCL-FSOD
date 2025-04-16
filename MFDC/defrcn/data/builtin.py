import os
from .meta_voc import register_meta_voc
from .meta_coco import register_meta_coco
from .builtin_meta import _get_builtin_metadata
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances

# -------- COCO -------- #
def register_all_coco(root="datasets"):

    METASPLITS = [
        ("coco14_trainval_all", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        ("coco14_trainval_base", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
        ("coco14_test_all", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco14_test_base", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("coco14_test_novel", "coco/val2014", "cocosplit/datasplit/5k.json"),
        ("removecoco14_trainval_all", "coco/trainval2014", "cocosplit/datasplit/trainvalno5k.json"),
    ]
    for prefix in ["all", "novel"]:
        for shot in [1, 2, 3, 5, 10, 30]:
            for seed in range(10):
                name = "coco14_trainval_{}_{}shot_seed{}".format(prefix, shot, seed)
                METASPLITS.append((name, "coco/trainval2014", ""))

                if prefix == "all":
                    name = "removecoco14_trainval_{}_{}shot_seed{}".format(prefix, shot, seed)
                    METASPLITS.append((name, "coco/trainval2014", ""))

    for name, imgdir, annofile in METASPLITS:
        register_meta_coco(
            name,
            _get_builtin_metadata("coco_fewshot"),
            os.path.join(root, imgdir),
            os.path.join(root, annofile),
        )


# -------- PASCAL VOC -------- #
def register_all_voc(root="datasets"):

    METASPLITS = [
        ("voc_2007_trainval_base1", "VOC2007", "trainval", "base1", 1),
        ("voc_2007_trainval_base2", "VOC2007", "trainval", "base2", 2),
        ("voc_2007_trainval_base3", "VOC2007", "trainval", "base3", 3),
        ("voc_2012_trainval_base1", "VOC2012", "trainval", "base1", 1),
        ("voc_2012_trainval_base2", "VOC2012", "trainval", "base2", 2),
        ("voc_2012_trainval_base3", "VOC2012", "trainval", "base3", 3),
        ("voc_2007_trainval_all1", "VOC2007", "trainval", "base_novel_1", 1),
        ("voc_2007_trainval_all2", "VOC2007", "trainval", "base_novel_2", 2),
        ("voc_2007_trainval_all3", "VOC2007", "trainval", "base_novel_3", 3),
        ("voc_2012_trainval_all1", "VOC2012", "trainval", "base_novel_1", 1),
        ("voc_2012_trainval_all2", "VOC2012", "trainval", "base_novel_2", 2),
        ("voc_2012_trainval_all3", "VOC2012", "trainval", "base_novel_3", 3),
        ("voc_2007_test_base1", "VOC2007", "test", "base1", 1),
        ("voc_2007_test_base2", "VOC2007", "test", "base2", 2),
        ("voc_2007_test_base3", "VOC2007", "test", "base3", 3),
        ("voc_2007_test_novel1", "VOC2007", "test", "novel1", 1),
        ("voc_2007_test_novel2", "VOC2007", "test", "novel2", 2),
        ("voc_2007_test_novel3", "VOC2007", "test", "novel3", 3),
        ("voc_2007_test_all1", "VOC2007", "test", "base_novel_1", 1),
        ("voc_2007_test_all2", "VOC2007", "test", "base_novel_2", 2),
        ("voc_2007_test_all3", "VOC2007", "test", "base_novel_3", 3),
    ]
    for prefix in ["all", "novel"]:
        for sid in range(1, 4):
            for shot in [1, 2, 3, 5, 10]:
                for year in [2007, 2012]:
                    for seed in range(30):
                        seed = "_seed{}".format(seed)
                        name = "voc_{}_trainval_{}{}_{}shot{}".format(
                            year, prefix, sid, shot, seed
                        )
                        dirname = "VOC{}".format(year)
                        img_file = "{}_{}shot_split_{}_trainval".format(
                            prefix, shot, sid
                        )
                        keepclasses = (
                            "base_novel_{}".format(sid)
                            if prefix == "all"
                            else "novel{}".format(sid)
                        )
                        METASPLITS.append(
                            (name, dirname, img_file, keepclasses, sid)
                        )

                        if prefix == "all":
                            name = "removevoc_{}_trainval_{}{}_{}shot{}".format(
                                year, prefix, sid, shot, seed
                            )
                            METASPLITS.append(
                                (name, dirname, img_file, keepclasses, sid)
                            )

    for name, dirname, split, keepclasses, sid in METASPLITS:
        year = 2007 if "2007" in name else 2012
        register_meta_voc(
            name,
            _get_builtin_metadata("voc_fewshot"),
            os.path.join(root, dirname),
            split,
            year,
            keepclasses,
            sid,
        )
        MetadataCatalog.get(name).evaluator_type = "pascal_voc"

# -------- FSOD-1K -------- #
fsod_novel_classes = ['beer',
 'musical keyboard',
 'jalapeno',
 'maple',
 'cartwheel',
 'christmas tree',
 'hiking equipment',
 'bicycle helmet',
 'laelia',
 'cattleya',
 'bran muffin',
 'goggles',
 'caribou',
 'buskin',
 'turban',
 'tortoise',
 'whiteboard',
 'chalk',
 'cider vinegar',
 'lantern',
 'bannock',
 'convenience store',
 'persimmon',
 'lifejacket',
 'squid',
 'watermelon',
 'wing tip',
 'sunflower',
 'shin guard',
 'baby shoe',
 'muffin',
 'mixer',
 'bronze sculpture',
 'euphonium',
 'skyscraper',
 'drinking straw',
 'popover',
 'segway',
 'sun hat',
 'harbor seal',
 'cat furniture',
 'fedora',
 'kitchen knife',
 'pulley',
 'walking shoe',
 'fancy dress',
 'clam',
 'hand dryer',
 'mozzarella',
 'peccary',
 'spinning rod',
 'tree house',
 'khimar',
 'earrings',
 'power plugs and sockets',
 'waste container',
 'blender',
 'briefcase',
 'soap dish',
 'hot air balloon',
 'windmill',
 'street light',
 'shotgun',
 'sports uniform',
 'manometer',
 'wood burning stove',
 'gnu',
 'earphone',
 'double hung window',
 'billboard',
 'conserve',
 'claymore',
 'vehicle registration plate',
 'ceiling fan',
 'cassette deck',
 'table tennis racket',
 'scone',
 'bouquet',
 'bidet',
 'ski boot',
 'pumpkin',
 'welsh poppy',
 'tablet computer',
 'rhinoceros',
 'cheese',
 'jacuzzi',
 'door handle',
 'puffball',
 'swimming pool',
 'rays and skates',
 'chopsticks',
 'oyster',
 'office building',
 'ratchet',
 'sambuca',
 'truffle',
 'salt and pepper shakers',
 'calla lily',
 'hard hat',
 'elephant seal',
 'peanut',
 'hind',
 'jelly fungus',
 'juice',
 'pirogi',
 'bowling equipment',
 'recycling bin',
 'skull',
 'nightstand',
 'light bulb',
 'high heels',
 'picnic basket',
 'in line skate',
 'platter',
 'bialy',
 'shelf bracket',
 'cantaloupe',
 'croissant',
 'bowling shoe',
 'ferris wheel',
 'dinosaur',
 'adhesive tape',
 'stanhopea',
 'mechanical fan',
 'winter melon',
 'cowrie',
 'adjustable wrench',
 'date bread',
 'o ring',
 'caryatid',
 'egg',
 'beehive',
 'lily',
 'leaf spring',
 'french bread',
 'cake stand',
 'sergeant major',
 'treadmill',
 'daiquiri',
 'sweet roll',
 'polypore',
 'face veil',
 'kitchen & dining room table',
 'support hose',
 'headphones',
 'chinese lantern',
 'wine rack',
 'triangle',
 'mulberry',
 'quick bread',
 'harpsichord',
 'optical disk',
 'egg yolk',
 'shallot',
 'strawflower',
 'cue',
 'corded phone',
 'blue columbine',
 'silo',
 'mascara',
 'snowman',
 'cherry tomato',
 'box wrench',
 'flipper',
 'jet ski',
 'bathrobe',
 'fireplace',
 'gill fungus',
 'blackboard',
 'thumbtack',
 'spice rack',
 'longhorn',
 'pacific walrus',
 'streptocarpus',
 'coconut',
 'addax',
 'coffeemaker',
 'fly orchid',
 'blackberry',
 'kob',
 'car tire',
 'seahorse',
 'tiara',
 'sassaby',
 'fishing rod',
 'baguet',
 'trowel',
 'light switch',
 'cornbread',
 'disa',
 'serving tray',
 'tuning fork',
 'virginia spring beauty',
 'samosa',
 'bathroom cabinet',
 'chigetai',
 'blue poppy',
 'scimitar',
 'shirt button',
 'slow cooker']


def register_all_fsod(root="datasets"):
    register_coco_instances("fsod_train_set_base", {}, "datasets/fsod/annotations/fsod_train.json", os.path.join(root, "fsod/"))
    register_coco_instances("fsod_5shot", {"thing_classes": fsod_novel_classes}, "datasets/fsod/annotations/fsod_5shot.json", os.path.join(root, "fsod/"))
    register_coco_instances("fsod_5shot_test", {"thing_classes": fsod_novel_classes}, "datasets/fsod/annotations/fsod_5shot_test.json", os.path.join(root, "fsod/"))
    register_coco_instances("fsod_test_set", {"thing_classes": fsod_novel_classes}, "datasets/fsod/annotations/fsod_test.json", os.path.join(root, "fsod/"))

# -------- FSVOD-500 -------- #
fsvod_test_classes = ['JetLev-Flyer',
 'amphibian',
 'aoudad',
 'asian_crocodile',
 'autogiro',
 'ax',
 'bactrian_camel',
 'balloon',
 'bathyscaphe',
 'belgian_hare',
 'binturong',
 'black_rabbit',
 'black_squirrel',
 'bow_(weapon)',
 'brahman',
 'canada_porcupine',
 'cheetah',
 'chiacoan_peccary',
 'chimaera',
 'chinese_paddlefish',
 'coin',
 'crab',
 'crayfish',
 'cruise_missile',
 'deer',
 'destroyer_escort',
 'dumpcart',
 'elasmobranch',
 'elk',
 'fall_cankerworm',
 'fanaloka',
 'fish',
 'flag',
 'fox',
 'garden_centipede',
 'gavial',
 'gemsbok',
 'giant_panda',
 'goat',
 'guard_ship',
 'guitar',
 'hand_truck',
 'hermit_crab',
 'hog',
 'horse_cart',
 'horseshoe_crab',
 'humvee',
 'ibex',
 'indian_rhinoceros',
 'lander',
 'langur',
 'lappet_caterpillar',
 'lemur',
 'leopard',
 'lesser_panda',
 'lion',
 'luge',
 'malayan_tapir',
 'minisub',
 'monkey',
 'mouflon',
 'mountain_goat',
 'orangutan',
 'pacific_walrus',
 'peba',
 'pedicab',
 'peludo',
 "pere_david's_deer",
 'pistol',
 'pony_cart',
 'pung',
 'rabbit',
 'raccoon',
 'reconnaissance_vehicle',
 'rubic_cube',
 'sassaby',
 'saxophone',
 'sepia',
 'serow',
 'shark',
 'shawl',
 'skibob',
 'snow_leopard',
 'snowmobil',
 'sow',
 'spider_monkey',
 'squirrel',
 'suricate',
 'tadpole_shrimp',
 'tiglon',
 'virginia_deer',
 'warthog',
 'whale',
 'wheelchair',
 'white-tailed_jackrabbit',
 'white_crocodile',
 'white_rabbit',
 'white_rhinoceros',
 'white_squirrel',
 'woolly_monkey']

def register_all_fsvod(root="datasets"):
    register_coco_instances("fsvod_train_set_base", {}, "datasets/FSVOD-500/annotations/fsvod_train_clean.json", os.path.join(root, "FSVOD-500/"))
    register_coco_instances("fsvod_5shot", {"thing_classes": fsvod_test_classes}, "datasets/FSVOD-500/annotations/offline_support_fsvod_test.json", os.path.join(root, "FSVOD-500/"))
    register_coco_instances("fsvod_test_set", {"thing_classes": fsvod_test_classes}, "datasets/FSVOD-500/annotations/fsvod_test_clean.json", os.path.join(root, "FSVOD-500/"))

register_all_coco()
register_all_voc()
register_all_fsod()
register_all_fsvod()
