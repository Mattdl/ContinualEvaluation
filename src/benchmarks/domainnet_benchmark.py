#  Copyright (c) 2022. Matthias De Lange (KU Leuven).
#  Copyrights licensed under the MIT License. All rights reserved.
#  See the accompanying LICENSE file for terms.
#
#  Codebase of paper "Continual evaluation for lifelong learning: Identifying the stability gap",
#  publicly available at https://arxiv.org/abs/2205.13452

from avalanche.benchmarks.datasets import default_dataset_location
from pathlib import Path
from typing import Union, Optional, Any, List
import pandas as pd

from avalanche.benchmarks.classic.classic_benchmarks_utils import \
    check_vision_benchmark
from avalanche.benchmarks import dataset_benchmark
from torchvision import transforms
from src.benchmarks.utils import wrap_with_task_labels

from src.benchmarks.domainnet import MiniDomainNet, DomainNet
from collections import defaultdict
from tabulate import tabulate

# See MiniDOmainnet: https://github.com/KaiyangZhou/Dassl.pytorch/blob/master/configs/datasets/da/mini_domainnet.yaml
DOMAIN_NET_W = 224
MINI_DOMAIN_NET_W = 96
TINY_DOMAIN_NET_W = MINI_DOMAIN_NET_W
_default_train_transform = transforms.Compose([
    transforms.Resize(TINY_DOMAIN_NET_W),  # rescale and crop
    transforms.CenterCrop(TINY_DOMAIN_NET_W),
    transforms.ToTensor(),  # PIL [0,255] range to [0,1]
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))  # From ImageNet
])

_default_eval_transform = _default_train_transform


def MiniDomainNetBenchmark(
        *,
        train_transform: Optional[Any] = _default_train_transform,
        eval_transform: Optional[Any] = _default_eval_transform,
        dataset_root: Union[str, Path]):
    """
    Creates a CL benchmark using a sequence of 4 MiniDomainNet tasks, where each task is one domain.

    If the dataset is not present in the computer, this method will
    automatically download and store it.

    The returned benchmark will return experiences containing all patterns of a
    subset of classes, which means that each class is only seen "once".
    This is one of the most common scenarios in the Continual Learning
    literature. Common names used in literature to describe this kind of
    scenario are "Class Incremental", "New Classes", etc. By default,
    an equal amount of classes will be assigned to each experience.

    This generator doesn't force a choice on the availability of task labels,
    a choice that is left to the user (see the `return_task_id` parameter for
    more info on task labels).

    The benchmark instance returned by this method will have two fields,
    `train_stream` and `test_stream`, which can be iterated to obtain
    training and test :class:`Experience`. Each Experience contains the
    `dataset` and the associated task label.

    The benchmark API is quite simple and is uniform across all benchmark
    generators. It is recommended to check the tutorial of the "benchmark" API,
    which contains usage examples ranging from "basic" to "advanced".

    :param train_transform: The transformation to apply to the training data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default train transformation
        will be used.
    :param eval_transform: The transformation to apply to the test data,
        e.g. a random crop, a normalization or a concatenation of different
        transformations (see torchvision.transform documentation for a
        comprehensive list of possible transformations).
        If no transformation is passed, the default eval transformation
        will be used.
    :param dataset_root: The root path of the dataset. Defaults to None, which
        means that the default location will be used.

    :returns: A properly initialized :class:`NCScenario` instance.
    """
    if dataset_root is None:
        dataset_root = default_dataset_location('MiniDomainNet')

    considered_classes = MiniDomainNet.classes_list

    # Datasets
    train_sets, test_sets = [], []
    for domain in MiniDomainNet.domains:
        train_sets.append(MiniDomainNet(classes_list=considered_classes, ds_root=dataset_root, domain=domain,
                                        train=True, transform=train_transform))
        test_sets.append(MiniDomainNet(classes_list=considered_classes, ds_root=dataset_root, domain=domain,
                                       train=False, transform=eval_transform))

    # TRAINING SUMMARY
    print(f"\n\n {'*' * 40} TRAINING SUMMARY {'*' * 40}")
    produce_class_summary(train_sets)

    # TESTING SUMMARY
    print(f"\n\n {'*' * 40} TESTING SUMMARY {'*' * 40}")
    produce_class_summary(test_sets)

    return dataset_benchmark(
        train_datasets=wrap_with_task_labels(train_sets),
        test_datasets=wrap_with_task_labels(test_sets),
        complete_test_set_only=False,  # Return test set per task (not a single test set)
        train_transform=None,  # For TRAIN Add in dataset itself! (Otherwise trouble in restoring Replay state)
        eval_transform=None,
    )


def produce_class_summary(datasets: List[MiniDomainNet]):
    """ Make a per-class summary over all classes in MiniDomainNet."""
    datacnt_per_class = defaultdict(list)  # Datapoints per class (in total)
    datacnt_per_domain = []
    domain_headers = []

    for ds_idx, ds in enumerate(datasets):  # Iterate tasks (domains)
        domain_headers.append(f"#{ds.domain}")
        per_label_dict = ds.split_dataset_by_label(ds.data)

        for c, c_samples in per_label_dict.items():  # Iterate classes with counts
            datacnt_per_class[c].append(len(c_samples))

        # Per-domain summary
        datacnt_per_domain.append(len(ds))

    # Sort classes
    sorted_per_class = [[c_cnt, c, *datacnt_per_class[c]]
                        for c_cnt, c in enumerate(sorted(list(datacnt_per_class.keys())))]

    # Add min-max summary lines
    counts_df = pd.DataFrame.from_records(sorted_per_class)  # Includes class-idxs in first two columns

    # Add summary count as column
    counts_df = counts_df.assign(Total=counts_df.iloc[:, 2:].sum(axis=1))

    col_sum, col_min, col_max = [], [], []
    for column_name in counts_df.iloc[:, 1:]:
        column = counts_df[column_name]
        col_sum.append(column.sum())
        col_min.append(column.min())
        col_max.append(column.max())

    # Append summary line
    totals_row = ['SUM', *col_sum]  # Class'label
    mins_row = ['MIN', *col_min]  # Class'label
    maxs_row = ['MAX', *col_max]  # Class'label

    # Display
    all_rows = [*counts_df.values.tolist(), totals_row, mins_row, maxs_row]
    headers = ['class_idx', 'orig Class', *domain_headers, '#Total']
    print(tabulate(all_rows, headers=headers))


if __name__ == "__main__":
    import sys

    DomainNet_path = ""
    benchmark_instance = MiniDomainNetBenchmark(dataset_root=None)
    check_vision_benchmark(benchmark_instance)
    sys.exit(0)

__all__ = [
    'MiniDomainNetBenchmark',
]

"""
MiniDomainNet summary:

 **************************************** TRAINING SUMMARY ****************************************
class_idx      orig Class    #clipart    #painting    #real    #sketch    #Total
-----------  ------------  ----------  -----------  -------  ---------  --------
SUM                 20148       12004        17145    43893      17316     90358
MIN                     1           8            4       21          8       218
MAX                   344         270          586      557        483      1431

0                       1          51          144      152        231       578
1                       3          56           51      436         80       623
2                       7          85          105      232         62       484
3                       8          61          311       37        126       535
4                      11          33          153      267        145       598
5                      12          23          185      307        153       668
6                      14          32          135      274         39       480
7                      19          42          192      165        112       511
8                      22          73          349      435         55       912
9                      24         109          259      509        335      1212
10                     27          95           11      461         87       654
11                     32         115           83      204        163       565
12                     34         119          114      512         89       834
13                     35         116           44      498        102       760
14                     38          93          197      360        225       875
15                     44         118           58      442        163       781
16                     48          32           46       21        433       532
17                     50          83           85      460         42       670
18                     51          59          107      543         46       755
19                     52          38            8      261         48       355
20                     53          46           30      123         41       240
21                     55          40          109      334         76       559
22                     56         126           50       85         48       309
23                     59          72           37      207         65       381
24                     60          47          266      490         90       893
25                     61          69           31      393        101       594
26                     63          32          157      477         39       705
27                     64          30          240      557         91       918
28                     65          24           26      151         17       218
29                     66          65          110      314         44       533
30                     67          26           92      364         16       498
31                     68          65           36      224         67       392
32                     69         156           39      275         23       493
33                     70          37           99      467         24       627
34                     76         131           54      174          8       367
35                     77         200           11      250         19       480
36                     78          67           37      472         23       599
37                     82          75          107      495        106       783
38                     83          98           12      352        199       661
39                     85         144           55      103        121       423
40                     87          74          363      258        271       966
41                     92          58          280      405         59       802
42                     93          97          261      441         88       887
43                     96          28           98      162          9       297
44                     99          99          293      281        192       865
45                    100         269          132      402        129       932
46                    101          70          130      239        139       578
47                    104         102          203      458        125       888
48                    108          37           14      485        312       848
49                    111         114           34      537         98       783
50                    112         107           39      431        198       775
51                    117         191          154      368         99       812
52                    118         154          292      320         64       830
53                    122         163          169      258         94       684
54                    124         139           58      239        123       559
55                    125         114          116      532        142       904
56                    126         130          118      279         92       619
57                    130         156           73      389        124       742
58                    131         142          443      363        483      1431
59                    133         100          229      257        120       706
60                    135         121          100      391        127       739
61                    137          63          177      361        166       767
62                    143         114           18      432        146       710
63                    144         135          112      399         80       726
64                    148          66           35      464         16       581
65                    160          40           66      149         95       350
66                    166          17          107      262        219       605
67                    167           8          350      284        263       905
68                    168          61          124      453         95       733
69                    175          70          137      312         77       596
70                    176         170          176      453        119       918
71                    177          51          174      419         74       718
72                    185          11            7      236        118       372
73                    187          88          225      396        108       817
74                    189          29           74      540        146       789
75                    192         149          291      284         64       788
76                    195          94          176      548        176       994
77                    196          27          586      457         15      1085
78                    203           9            7      340        104       460
79                    209          11          265      278         95       649
80                    215          51          311      303        127       792
81                    216          62           61      474         55       652
82                    217          35          126      319         18       498
83                    218          84          308      488        146      1026
84                    219          14          201      398         83       696
85                    223         104           95      455         80       734
86                    224          58          233      465         91       847
87                    225          53           88      419        140       700
88                    233          17           70      430         65       582
89                    234          27           34      380        157       598
90                    235          72          187      481         63       803
91                    236         129          174      472        241      1016
92                    237          21           45      277        115       458
93                    243          58          168      363         85       674
94                    244          92          387      453         69      1001
95                    251         139           43      311        244       737
96                    255         164          281      428        177      1050
97                    256         209          113      186        362       870
98                    257         142          188      128        344       802
99                    259          88          182      410        449      1129
100                   260          96          111      310        362       879
101                   264         124          130      221        408       883
102                   270         194           74      482        272      1022
103                   276         112          136      415        450      1113
104                   277         159           75      364        280       878
105                   282         270           12      246        366       894
106                   291         228          307      320        181      1036
107                   292          97           42      342         43       524
108                   293         240          280      423        142      1085
109                   294         261           99      301        216       877
110                   297         155           76      405        116       752
111                   302         155          269      435        220      1079
112                   303          76          192      367        140       775
113                   304         102           53      333        176       664
114                   306         129           12      310        141       592
115                   309          80          108      368         93       649
116                   310         105          130      155         99       489
117                   312         137           72      372        183       764
118                   314         119           21      407         78       625
119                   322          87          387      373        379      1226
120                   326          81           85      270        125       561
121                   329         175            5      200         87       467
122                   332         118          139      355        142       754
123                   335         111            4      424         72       611
124                   337          92           13      278        116       499
125                   344         221           77      332        100       730



 **************************************** TESTING SUMMARY ****************************************
class_idx      orig Class    #clipart    #painting    #real    #sketch    #Total
-----------  ------------  ----------  -----------  -------  ---------  --------
SUM                 20148        5233         7436    18888       7506     39063
MIN                     1           4            3       10          4        97
MAX                   344         117          252      238        207       619

0                       1          22           62       66        100       250
1                       3          24           23      187         35       269
2                       7          37           46      100         27       210
3                       8          27          134       17         55       233
4                      11          15           66      115         59       255
5                      12          10           80      132         66       288
6                      14          15           59      119         17       210
7                      19          19           82       72         47       220
8                      22          32          150      187         24       393
9                      24          47          112      219        145       523
10                     27          42            6      199         38       285
11                     32          50           36       91         70       247
12                     34          52           51      220         39       362
13                     35          50           20      213         44       327
14                     38          43           83      156         99       381
15                     44          52           26      190         70       338
16                     48          14           21       10        186       231
17                     50          36           37      197         19       289
18                     51          23           45      231         21       320
19                     52          17            4      113         21       155
20                     53          20           14       53         18       105
21                     55          18           47      144         33       242
22                     56          55           22       38         21       136
23                     59          31           17       88         28       164
24                     60          21          107      211         39       378
25                     61          30           14      170         44       258
26                     63          15           68      205         17       305
27                     64          13          104      238         39       394
28                     65          11           12       66          8        97
29                     66          28           48      134         20       230
30                     67          12           41      156          7       216
31                     68          29           16       96         29       170
32                     69          67           18      118         11       214
33                     70          17           43      201         11       272
34                     76          58           24       76          5       163
35                     77          87            6      105         10       208
36                     78          30           17      203         10       260
37                     82          33           46      213         46       338
38                     83          43            6      151         86       286
39                     85          63           25       51         52       191
40                     87          29          153      114        117       413
41                     92          26          121      174         26       347
42                     93          42          112      186         39       379
43                     96          13           43       71          4       131
44                     99          43          126      120         83       372
45                    100         117           57      172         55       401
46                    101          31           57      102         60       250
47                    104          45           88      197         55       385
48                    108          16            6      208        136       366
49                    111          50           15      229         42       336
50                    112          46           18      185         85       334
51                    117          83           68      157         43       351
52                    118          66          126      139         28       359
53                    122          70           73      115         39       297
54                    124          59           26      104         53       242
55                    125          49           51      229         61       390
56                    126          57           51      120         39       267
57                    130          67           36      165         57       325
58                    131          63          193      156        207       619
59                    133          45           98      112         52       307
60                    135          49           45      167         56       317
61                    137          29           79      160         70       338
62                    143          49            9      187         63       308
63                    144          59           48      175         35       317
64                    148          29           15      203          8       255
65                    160          18           30       66         42       156
66                    166           7           48      114         93       262
67                    167           4          152      123        113       392
68                    168          27           54      195         40       316
69                    175          31           59      134         33       257
70                    176          73           76      195         53       397
71                    177          23           74      183         31       311
72                    185           5            3      102         52       162
73                    187          38           97      170         47       352
74                    189          13           32      232         63       340
75                    192          66          121      120         29       336
76                    195          41           77      237         76       431
77                    196          13          252      195          7       467
78                    203           5            4      145         46       200
79                    209           5          114      116         40       275
80                    215          23          135      131         55       344
81                    216          28           28      205         25       286
82                    217          16           53      138          8       215
83                    218          37          135      208         63       443
84                    219           6           83      170         36       295
85                    223          46           42      192         35       315
86                    224          25          100      198         40       363
87                    225          24           39      180         61       304
88                    233           8           31      185         28       252
89                    234          12           15      164         69       260
90                    235          31           81      209         28       349
91                    236          57           74      203        105       439
92                    237           9           20      120         50       199
93                    243          25           72      156         37       290
94                    244          40          167      196         32       435
95                    251          62           17      134        107       320
96                    255          71          121      186         77       455
97                    256          90           49       80        154       373
98                    257          61           81       55        148       345
99                    259          38           78      177        193       486
100                   260          41           48      133        155       377
101                   264          54           56       96        180       386
102                   270          84           41      207        118       450
103                   276          49           66      177        193       485
104                   277          68           31      158        120       377
105                   282         116            6      105        155       382
106                   291          98          125      138         80       441
107                   292          42           16      146         20       224
108                   293         104          127      183         61       475
109                   294         114           45      129         93       381
110                   297          67           29      174         51       321
111                   302          66          116      189         95       466
112                   303          34           83      157         61       335
113                   304          45           24      144         76       289
114                   306          57            6      135         61       259
115                   309          35           47      156         43       281
116                   310          45           55       65         43       208
117                   312          59           32      153         78       322
118                   314          53           10      174         36       273
119                   322          37          164      157        162       520
120                   326          34           36      113         55       238
121                   329          76            4       86         37       203
122                   332          53           60      151         60       324
123                   335          48            4      180         35       267
124                   337          40            6      122         49       217
125                   344          96           33      148         44       321

"""
