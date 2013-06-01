import numpy as np
import matplotlib.pyplot as plot
#pos2green
y= [681.743, 662.12, 931.126, 1214.01, 632.229, 658.116, 603.67, 536.833, 756.438, 1044.05, 653.162, 577.872, 568.438, 562.657, 552.272, 547.224, 532.969, 505.124, 610.332, 540.592, 575.771, 498.912, 489.702, 598.537, 573.327, 584.177, 617.958, 667.46, 568.544, 516.282, 609.562, 690.694, 574.577, 614.993, 604.329, 632.576, 563.449, 558.082, 614.49, 657.338, 631.615, 626.906, 651.686, 770.814, 668.025, 616.958, 697.004, 643.536, 749.92, 722.886, 685.852, 668.536, 725.505, 772.818, 712.879, 653.323, 774.895, 734.905, 779.83, 854.317, 622.252, 756.669, 957.158, 753.328, 833.74, 813.774, 599.949, 688.538, 856.767, 824.431, 785.724, 867.578, 764.84, 761.088, 673.847, 713.189, 879.369, 678.094, 703.111, 758.669, 822.999, 873.861, 835.924, 1095.43, 952.186, 884.578, 936.288, 3362.11, 1185.11, 879.104, 677.031, 982.987, 1209.75, 957.92, 719.284, 1037.5, 796.198, 2422.94, 954.599, 1464.55, 858.934, 924.957, 1067.82, 1012.1, 2190.56, 820.816, 1255.44, 1144.36, 946.304, 954.194, 1062.74, 1467.66, 1051.94, 2419.82, 2860.71, 1053.89, 2073.97, 928.202, 886.938, 1018.56, 1175.71, 1337.36, 3102.44, 977.935, 1081.49, 948.079, 950.438, 918.01, 3888.59, 1102.57, 1697.7, 2793, 1299.08, 1223.99, 1176.7, 993.709, 1006.51, 1123.77, 1141.53, 1083.39, 1472.94, 1088.77, 1170.23, 1537.65, 1155.26, 1300.27, 1359, 2135.16, 1393, 905.614, 1145.05, 1444.29, 983.482, 1083.65, 1220.55, 1516.89, 1199.18, 1163.77, 1923.71, 1301.71, 2516.39, 1129.27, 1301.87, 1280.35, 1702.66, 1814.58, 1651.47, 2336.18, 1504.18, 1430.18, 2332.07, 1167.41, 1581.03, 1002.2, 1243.13, 1058.51, 1782.7, 1225.01, 1195.16, 1577.81, 23929, 1059.58, 5423.47, 1133.51, 1917.53, 1107.96, 1268.91, 3419.2, 1793.77, 4151.62, 1179.79, 2084.39, 1353.24, 1470.79, 1429.52, 1558.47, 1493.52, 3660.16, 1420.78, 1245.21, 1409.37, 1324.64, 1296.32, 2471.82, 2050.03, 1437.92, 1522.4, 3132.82, 10654, 5890.01, 1406.81, 1326.32, 1724.42, 2290.95, 1439.53, 1667.12, 1660.29, 2415.91, 6546.46, 1314.76, 1611.3, 2480.19, 1429.86, 1767.77, 1811.65, 2822.91, 1575.8, 2536.36, 1253.17, 1916.63, 2672.12, 2059.45, 1568, 1854.89, 3843.53, 1704.17, 3445.03, 1636.03, 1704.45, 1580.38, 1731.48, 1789.67, 2557.66, 1619.69, 1669.88, 3076.99, 7281.66, 1289.18, 1581.71, 1881.08, 1586.08, 1663.76, 6455.87, 1681.9, 6393.47, 3048.4, 1675.54, 1656.01, 14749.3, 4676.28, 7223.98, 25183.7, 1731.44, 13572.3, 3387.42, 2375.23, 3240.39, 1677.17, 4616.91, 1705.99, 1395.03, 2109.66, 2563.98, 3359.19, 5614.09, 4099.91, 1918.14, 10484.3, 2730.39, 3659.11, 1783.3, 1583.47, 7363.56, 1916.97, 2073.23, 1948.28, 1712.64, 6930.29, 1635.33, 1912.95, 1688.31, 2097.27, 5737.7, 4376.4, 4476.25, 10203.6, 2008.16, 1867.04, 2074.93, 3604.52, 7713.94, 2263.18, 3496.64, 2106.36, 2389.58, 1941.93, 2181.15, 2003.48, 1856.04, 2415.85, 1977.1, 2613.18, 1983.39, 15753.1, 1729.34, 2011.13, 3859.13, 8179.89, 2221.26, 8125.42, 1771.9, 2080.67, 1921.51, 2127, 1700.51, 2353.38, 2060.66, 18387.5, 1895.56, 2026.48, 4528.17, 7582.23, 2499.08, 2047.94, 2244.21, 2225.63, 4372.16, 5601.77, 2897.22, 5335.34, 2000.38, 13550.7, 5255.65, 10813, 1831.48, 2573.45, 13120.1, 6544.92, 3580.48, 5535.24, 4463.65, 10876.9, 21035.3, 22631.1, 8061.44, 13452.1, 24642.4, 13949.6, 9204.74, 11998.4, 5281.75, 45236.6, 12117.1, 5788.4, 11672.8, 6294.83, 8693.7, 2143.08, 6895.15, 12318.9, 5431.11, 5146.42, 28870, 6040.51, 9489.4, 5968.09, 8144.23, 7511.53, 31740.5, 22570.1, 3891, 2511.15, 5354.78, 18269.1, 16045.1, 13173, 5682.23, 25674, 21055.2, 19824.8, 14339.7, 19345.5, 6500.51, 6589.79, 26822.4, 42188.3, 6137.9, 6667.73, 13687, 17607.4, 11421.7, 5932.95, 48730.2, 7127.63, 10138.8, 3360.97, 33839.3, 13131.2, 8231.54, 18418.6, 27789.5, 8440.87, 8569.06, 16161.5, 22395.7, 7460.18, 8863.26, 12981.6, 12722.4, 3290.23, 6170.84, 23394.4, 26645.9, 26190.4, 24060.1, 20012.9, 25879.3, 13125.1, 55868.1, 12125.8, 25007.5, 32598.4, 25497.6, 27041, 12496.4, 24173.2, 11790.5, 49739.4, 4375.74, 8834.62, 17607.8, 7815.58, 26741.6, 10198.3, 10660.9, 6647.83, 14379, 24882.1, 6586.66, 35046.4, 10473.1, 37564.5, 35652.6, 27447.7, 29942.6, 25322.6, 2945.71, 34405.9, 14147.6, 10217.4, 59009.5, 9293.24, 49480, 40575.1, 54103.8, 32235.9, 32593.6, 24406.1, 65377, 32905.7, 17155.8, 18469.6, 33690.2, 22991.5, 51052.1, 42759.5, 98931.4, 36313.9, 70043.3, 15018.8, 45743.2, 98736.6, 27154.5, 33297.6, 25277.7, 41649.2, 53341.9, 28213.4, 12058.4, 43461.7, 24195.5, 13411.1, 53510.6, 5408.42, 49844, 22291.3, 37217, 36250, 45040.5, 27934.7, 23350.4, 8606.62, 46089.4, 16645, 21176.8, 5974.27, 4921.36, 39559.4, 58506.5, 55672.2, 28504.1, 26848.4, 16313.3, 16385.4, 24949.8, 25375.1, 43033.6, 38589.5, 138138, 32297.6, 29498.8, 23463.9, 19668.3, 16922.3, 4973.9, 23978.3, 9486.55, 39987.1, 20181.3, 63596.9, 17941.3, 24533.8, 93196.6, 34196.7, 32874.6, 36172.9, 35704.1, 30068.2, 23862.1, 21868.1, 16654.6, 36960.4, 13132.2, 32152.4, 37471.6, 80530.7, 20571.9, 62294.6, 18939, 30203.8, 38578.2, 21315.8, 6796.62, 44202.6, 38294.4, 40568.3, 48245.7, 36564.1, 63809.6, 22523.2, 76868.6, 5760.12, 44153.3, 3575.75, 23618.5, 17583.3, 4078.61, 3555.44, 35145.4, 39034.5, 15989.3, 34611, 33922.7, 20767.8, 76214.6, 132816, 57810.6, 59038.9, 64896.3, 37417.1, 30619.3, 77341.8, 17271.5, 26133, 34230.4, 30024.9, 45931.1, 47610.3, 6123.11, 28327.3, 60511.7, 48278.2, 36342, 20393.3, 41307, 34378, 65206.7, 37869.1, 48936.3, 36284.1, 20644.5, 27136.3, 43090.6, 3856.1, 47390.5, 105371, 95623.8, 31408.5, 86566.1, 40960.1, 22986, 70859.3, 146400, 52548.8, 36374.1, 64135.9, 65109.9, 22198.7, 5575.37, 69587.1, 27239.2, 27162.9, 21177.7, 37809.3, 72692.8, 26366.9, 28650.5, 40022.9, 45985.8, 23640, 45092.6, 83745.1, 5141.75, 49769.8, 56259.1, 46780.2, 59016.2, 3971.98, 6739.78, 21849.9, 55271.8, 69229.7, 56925, 29833, 73973.7, 72531, 38204, 66877.7, 22564.8, 33877.5, 41992.2, 34361.4, 51078.1, 61805.5, 51426.4, 53489.5, 42156.5, 50946.3, 66129, 5645.18, 4823.37, 44501.6, 36930.9, 50654.6, 39622.5, 44066.4, 4062.11, 27244.9, 28766.5, 81498.4, 32875.4, 61151.7, 33294.4, 37596.5, 92853.2, 40586.6, 39932.5, 61394.2, 7799.78, 6237.06, 27655.9, 57931.4, 132553, 3724.75, 57728.8, 75947.5, 7319.06, 3949.84, 53195.7, 7842.14, 77541.6, 42739.8, 88916.4, 42425.2, 63093.3, 57315.8, 52557.8, 56712.7, 50769.3, 100589, 6675.86, 64297.2, 82295.6, 44531.5, 58754.1, 4766.3, 119179, 74638.9, 95997.9, 65025.2, 4721.07, 13539.6, 13102.5, 7485.12, 54090.6, 44410.2, 84027.9, 44325.1, 47587.5, 76094.3, 44969, 35599.5, 86226, 14324.6, 54845, 4426.81, 46756.2, 71578.2, 59334.6, 63169, 49948.4, 91508, 47096.3, 103631, 42901.6, 49046.7, 9835.06, 64919, 49592.1, 50410, 37430.8, 69280.2, 79929.7, 13098.8, 59906.2, 43524.5, 90587.6, 85733.9, 63013.8, 7194.28, 11005.4, 86249.2, 78298, 65777.6, 51196.4, 60964, 90430.5, 86296.1, 13995.2, 63891.8, 62598.2, 44361.1, 64331.6, 56223.4, 5108.35, 163709, 49726.9, 61080.7, 58848, 5871.35, 59436.4, 60737.6, 53703.2, 52377.1, 71293.6, 56901, 53153.6, 48983.7, 63438.6, 59098.7, 81402.2, 49110.1, 4375.64, 43909.8, 61835.9, 15527.8, 13710.3, 82468.9, 60570.7, 17052, 8148.16, 70369.6, 58206.4, 13271.4, 65552.1, 8874.01, 57901.9, 58696.5, 5332.27, 62792.8, 86875.2, 58215.2, 61148.4, 71872.1, 4876.7, 13308.6, 92341.4, 74564.2, 67693.9, 10326.5, 11598.5, 59704.7, 65194.8, 68453.4, 93051.1, 72639.9, 111978, 62109.5, 5317.88, 57122.6, 72096.1, 48818.8, 59345.4, 12872.2, 10708.1, 63614.7, 71964.5, 120129, 62175.4, 58723.6, 91531.7, 80278.5, 71782.1, 24598.3, 61741.2, 96108.6, 81256.7, 12570.5, 89902.4, 6462.38, 95028.6, 10596.4, 11038.5, 99600.6, 66316.5, 92988, 78046.9, 78903.4, 67743.1, 92541, 81190.7, 75018.5, 7166.41, 23959, 11243.9, 71535.5, 16302.4, 78499.2, 93503.1, 10011.6, 100367, 98180.2, 70130.1, 79870.8, 73093.8, 7829.21, 82850.9, 18861.9, 73479.4, 14041.6, 91267.9, 124262, 32196.9, 21360.6, 10780.4, 119193, 15488.8, 9343.45, 12823.4, 8634.23, 21193.7, 77982.7, 19240.9, 28027.2, 82691.4, 15112.8, 126556, 141808, 13535.2, 12793.3, 29263, 59273.3, 30243.5, 20689.7, 80290.4, 21165.4, 10308.4, 24851.2, 11457.8, 11402.3, 15188.5, 61145.8, 32465.6, 14362.1, 14079.9, 14435.9, 28660.7, 28717.9, 87708.5, 36121.2, 18565.8, 14411.4, 15290.9, 40018.9, 30999.1]

x=[476.955, 481.235, 490.79, 490.805, 491.725, 496.71, 498.36, 504.335, 504.89, 506.56, 507.72, 508.22, 508.25, 508.45, 509.09, 509.34, 509.395, 509.715, 509.72, 510.055, 510.155, 510.28, 510.365, 510.385, 510.445, 510.445, 510.455, 510.505, 510.69, 510.72, 510.74, 510.855, 510.865, 511.13, 511.275, 511.315, 511.325, 511.41, 511.48, 512.04, 512.985, 514.06, 515.12, 516.175, 517.245, 518.32, 519.375, 520.435, 521.51, 522.56, 523.635, 524.685, 525.755, 526.82, 527.875, 528.955, 530.015, 531.07, 532.14, 533.195, 534.26, 535.325, 536.39, 537.45, 538.515, 539.575, 540.645, 541.705, 542.77, 543.83, 544.895, 545.96, 547.02, 548.085, 549.15, 550.215, 551.275, 552.34, 553.405, 554.465, 555.53, 556.595, 557.66, 558.72, 559.79, 560.85, 561.915, 562.975, 564.045, 565.105, 566.17, 567.23, 568.295, 569.36, 570.42, 571.49, 572.55, 573.61, 574.675, 575.74, 576.81, 577.865, 578.93, 579.995, 581.06, 582.12, 583.185, 584.255, 585.31, 586.375, 587.44, 588.5, 589.565, 590.63, 591.695, 592.755, 593.82, 594.885, 595.95, 597.01, 598.075, 599.145, 600.2, 601.265, 602.33, 603.395, 604.455, 605.52, 606.585, 607.645, 608.71, 609.775, 610.84, 611.9, 612.965, 614.03, 615.09, 616.155, 617.22, 618.285, 619.345, 620.41, 621.475, 622.54, 623.6, 624.665, 625.73, 626.79, 627.855, 628.92, 629.985, 631.05, 632.115, 633.175, 634.235, 635.3, 636.365, 637.43, 638.49, 639.555, 640.62, 641.685, 642.745, 643.81, 644.875, 645.935, 647, 648.065, 649.13, 650.19, 651.255, 652.325, 653.38, 654.445, 655.51, 656.575, 657.635, 658.7, 659.765, 660.825, 661.89, 662.955, 664.02, 665.085, 666.145, 667.21, 668.275, 669.34, 670.4, 671.465, 672.525, 673.59, 674.655, 675.72, 676.78, 677.845, 678.91, 679.97, 681.035, 682.1, 683.165, 684.23, 685.295, 686.36, 687.415, 688.48, 689.545, 690.61, 691.67, 692.735, 693.8, 694.87, 695.925, 696.99, 698.055, 699.12, 700.18, 701.245, 702.31, 703.37, 704.435, 705.51, 706.56, 707.625, 708.69, 709.755, 710.825, 711.885, 712.945, 714.015, 715.07, 716.145, 717.2, 718.265, 719.33, 720.39, 721.465, 722.515, 723.585, 724.645, 725.71, 726.77, 727.835, 728.915, 729.98, 731.025, 732.095, 733.15, 734.22, 735.28, 736.36, 737.405, 738.485, 739.55, 740.605, 741.67, 742.735, 743.81, 744.85, 745.92, 746.995, 748.045, 749.105, 750.175, 751.27, 752.315, 753.36, 754.425, 755.49, 756.55, 757.625, 758.68, 759.76, 760.815, 761.875, 762.945, 764.015, 765.07, 766.145, 767.2, 768.265, 769.315, 770.385, 771.45, 772.515, 773.57, 774.67, 775.71, 776.765, 777.855, 778.905, 779.955, 781.025, 782.09, 783.14, 784.215, 785.275, 786.34, 787.4, 788.485, 789.525, 790.6, 791.65, 792.74, 793.79, 794.85, 795.915, 796.97, 798.035, 799.115, 800.18, 801.23, 802.295, 803.415, 804.425, 805.485, 806.54, 807.605, 808.68, 809.745, 810.8, 811.885, 812.97, 813.985, 815.065, 816.135, 817.18, 818.245, 819.31, 820.38, 821.43, 822.51, 823.575, 824.64, 825.685, 826.765, 827.855, 828.915, 829.94, 831.01, 832.075, 833.175, 834.2, 835.28, 836.33, 837.415, 838.595, 839.53, 840.625, 841.65, 842.73, 843.77, 844.845, 845.94, 846.99, 848.115, 849.115, 850.2, 851.3, 852.36, 853.53, 854.51, 855.525, 856.635, 857.595, 858.705, 859.72, 860.87, 861.85, 862.995, 864.02, 865.16, 866.24, 867.175, 868.265, 869.34, 870.395, 871.56, 872.5, 873.765, 874.76, 875.69, 876.915, 877.925, 878.915, 880.06, 881.03, 882.085, 883.12, 884.235, 885.325, 886.365, 887.445, 888.475, 889.505, 890.595, 891.635, 892.705, 893.87, 894.875, 895.955, 897.11, 898.085, 899.18, 900.14, 901.225, 902.295, 903.43, 904.615, 905.555, 906.535, 907.625, 908.725, 909.76, 910.775, 911.885, 912.985, 913.99, 915.165, 916.155, 917.18, 918.48, 919.3, 920.36, 921.44, 922.53, 923.63, 924.825, 925.695, 926.735, 927.885, 928.98, 930.065, 931.05, 932.415, 933.23, 934.255, 935.32, 936.625, 937.395, 938.45, 939.54, 940.68, 941.96, 942.7, 943.975, 944.845, 946.03, 946.965, 948.235, 949.31, 950.21, 951.21, 952.295, 953.43, 954.555, 955.46, 956.675, 957.57, 958.765, 959.945, 960.9, 961.915, 962.92, 964.05, 965.105, 966.13, 967.325, 968.265, 969.38, 970.43, 972.015, 972.63, 974.34, 974.915, 975.89, 976.965, 978.035, 979.35, 980.135, 981.37, 982.105, 983.14, 984.43, 985.365, 986.51, 987.425, 988.66, 989.5, 990.57, 991.685, 992.715, 993.925, 994.915, 996.4, 997.1, 998.12, 999.225, 1000.14, 1001.32, 1002.47, 1003.45, 1004.47, 1005.71, 1007.45, 1008.11, 1008.83, 1009.74, 1010.78, 1012.34, 1013.05, 1014.32, 1015.46, 1016.2, 1017.22, 1018.81, 1019.85, 1020.55, 1021.63, 1022.48, 1023.77, 1025.05, 1026.69, 1026.95, 1028.12, 1028.88, 1030.07, 1031.2, 1032.21, 1033.17, 1034.28, 1035.66, 1036.98, 1037.51, 1038.59, 1039.56, 1040.8, 1042.44, 1043.67, 1044.14, 1044.94, 1046.4, 1047.1, 1048.13, 1049.09, 1050.18, 1051.49, 1053.02, 1054.52, 1054.62, 1055.67, 1056.55, 1058.31, 1058.73, 1059.99, 1060.78, 1061.9, 1063.03, 1063.98, 1065.34, 1066.26, 1067.15, 1068.38, 1069.47, 1070.37, 1071.75, 1072.89, 1074.26, 1075.43, 1075.81, 1076.85, 1077.82, 1078.94, 1080.19, 1081, 1082.3, 1083.1, 1084.27, 1086.06, 1086.29, 1087.69, 1088.71, 1090.29, 1090.72, 1091.68, 1092.69, 1093.92, 1095.44, 1095.94, 1097.02, 1099.92, 1100.72, 1101.9, 1102.31, 1103.04, 1103.39, 1105.58, 1105.68, 1106.79, 1107.79, 1109.02, 1109.69, 1111.98, 1112.17, 1113.56, 1114.01, 1115.12, 1116.18, 1117.39, 1118.29, 1120.55, 1120.82, 1121.58, 1122.54, 1123.67, 1125.4, 1125.94, 1126.9, 1127.83, 1129.01, 1130.05, 1131.84, 1132.09, 1133.18, 1135.2, 1135.24, 1138.06, 1138.71, 1138.73, 1142.22, 1143.07, 1143.1, 1143.53, 1144, 1144.79, 1146.11, 1147.39, 1148.36, 1149.46, 1150.19, 1151.7, 1153.2, 1153.76, 1154.86, 1155.44, 1156.54, 1157.61, 1159.31, 1159.76, 1160.94, 1161.93, 1163.57, 1164.01, 1165.19, 1166.81, 1167.27, 1170.73, 1170.83, 1172.51, 1172.89, 1173.05, 1173.86, 1174.98, 1175.97, 1176.68, 1179.72, 1183.27, 1183.74, 1183.91, 1184.81, 1184.86, 1186.02, 1186.66, 1187.2, 1188.06, 1188.95, 1190.23, 1190.86, 1192.14, 1192.84, 1195.08, 1196.12, 1198.06, 1198.28, 1198.31, 1199.07, 1201.99, 1202.29, 1202.32, 1204.34, 1204.64, 1205.48, 1207.69, 1208.33, 1209.17, 1209.65, 1212.44, 1212.8, 1213.4, 1213.91, 1215.58, 1219.1, 1219.39, 1219.94, 1220.32, 1221.84, 1222.43, 1223.6, 1223.66, 1226.91, 1228.43, 1230.99, 1231.86, 1232.34, 1232.65, 1232.97, 1233.52, 1234.88, 1235.66, 1235.83, 1237.11, 1237.39, 1239.77, 1242.17, 1243.24, 1243.69, 1243.8, 1244.8, 1245.08, 1245.81, 1247.55, 1247.94, 1249.57, 1250.64, 1251.27, 1252.78, 1255.01, 1255.2, 1259.02, 1259.08, 1260.49, 1261.25, 1261.59, 1264.93, 1267.47, 1267.92, 1269.08, 1269.33, 1272.7, 1272.71, 1272.78, 1273.08, 1273.49, 1274.69, 1275.44, 1278.68, 1281.65, 1282.52, 1282.67, 1284.01, 1284.35, 1284.42, 1286.11, 1288.92, 1289.05, 1289.81, 1289.95, 1290.8, 1292.72, 1292.93, 1294.24, 1295.94, 1296.49, 1297.17, 1297.94, 1298.46, 1303.51, 1303.9, 1307.9, 1308.02, 1308.91, 1311.44, 1312.31, 1312.67, 1313.17, 1314.77, 1315.71, 1319.74, 1320.44, 1320.62, 1321.85, 1322.11, 1322.56, 1323.23, 1328.73, 1328.95, 1335.44, 1337.9, 1341.02, 1342.33, 1343.22, 1347.7, 1348.26, 1348.91, 1350.56, 1351.68, 1356.09, 1356.9, 1358.8, 1361.73, 1361.95, 1363.16, 1365.05, 1365.15, 1367.26, 1369.09, 1369.82, 1372.17, 1374.85, 1375.17, 1380.13, 1382.59, 1384.06, 1390.94, 1398.3, 1399.94, 1401.34, 1401.42, 1404.11, 1407.56, 1409.44, 1409.79, 1410.38, 1412.47, 1414.53, 1415.04, 1415.22, 1415.94, 1424.47, 1424.92, 1425.81, 1426.16, 1426.89, 1427.05, 1428.13, 1428.28, 1431.08, 1448.83, 1462.28, 1467.28, 1472.24, 1475.01, 1481.66, 1491.71, 1492.1, 1493.39, 1501.56, 1508.19, 1510.34, 1517.3, 1522.89, 1525.12, 1527.1, 1529.14, 1530.76, 1536.15, 1540.71, 1548.15, 1550.05, 1553.65, 1555.67, 1557.11, 1564.11, 1575.24, 1581.66, 1583.35, 1604.53, 1605.8, 1649.47, 1656.36, 1657.05, 1657.53, 1664.23, 1664.86, 1672.6, 1675.35, 1703.63, 1705.12, 1705.58, 1708.38, 1714.99, 1751.62, 1796, 1796.85, 1919.04, 1981.48, 1984.73, 1994.78, 2035.6, 2060.94, 2083.54, 2103.58, 2186.93, 2255.43, 2259.26, 2290.61, 2330.15, 2349.48, 2375.01, 2376.64, 2392.32, 2477.42, 2514.92, 2586.97, 2597.65]

plot.scatter(x,y)
x1 = np.linspace(np.min(x), np.max(x))
y1 = 12.6357 * x1 + -6020.38#(Covariance minimization)
y2 = 7.04674 * x1 + -3184.88#(Weighted fit of lowes 10 points, half of the set used for each run)
y3 = 7.1201 * x1 + -3442.35#(lowest points from each interval, linear fit over random number of points between 5 and 10)
y4 = 3.13496 * x1 + -967.006#(real RANSAC)
y5 = 3.44498 * x1 + -1134.83#(real RANSAC with weighted fit instead of linear fit)
y6 = 5.76174 * x1+ -2345.82#(linear fit of 8 randomly selected points)


plot.plot(x1,y1,label='Fit I')
plot.plot(x1,y2, label = 'Fit II')
plot.plot(x1,y3, label='Fit III')
plot.plot(x1, y4, label = 'Fit IV')
plot.plot(x1, y5, label = 'Fit V')
plot.plot(x1, y6, label = 'Fit VI')

plot.legend()
plot.xlabel("mean intensities", fontsize=15)
plot.ylabel("variance", fontsize=15)
plot.tick_params(axis='both', labelsize=12)
plot.legend(loc = 2, prop={'size':15})
dx = np.max(x)-np.min(x)
plot.axis([np.min(x)-0.1*dx, np.max(x)+0.1*dx, np.min(y)-1000, 30000])
plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/geradenplots/pos2green.png', dpi=300)