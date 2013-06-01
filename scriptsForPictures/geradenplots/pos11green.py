import numpy as np
import matplotlib.pyplot as plot
#pos11green
y=[165.902, 140.24, 159.099, 171.192, 184.314, 182.394, 156.51, 178.124, 185.27, 187.442, 186.964, 173.604, 151.094, 187.04, 206.606, 180.401, 204.699, 204.526, 192.797, 221.184, 174.794, 227.26, 206.506, 167.624, 205.161, 206.215, 185.266, 200.967, 199.07, 196.946, 187.202, 205.686, 212.092, 235.764, 198.81, 204.028, 195.247, 214.299, 199.833, 176.276, 203.677, 168.704, 229.01, 178.382, 232.602, 282.439, 195.482, 230.829, 230.496, 227.189, 219.449, 227.312, 203.489, 240.107, 243.511, 246.741, 242.69, 267.934, 271.798, 231.265, 267.464, 248.885, 227.03, 336.894, 262.714, 243.772, 291.777, 283.228, 253.368, 268.514, 244.649, 261.91, 199.528, 249.634, 277.856, 331.967, 227.172, 288.889, 309.328, 225.61, 270.302, 313.009, 256.592, 315.318, 381.055, 289.214, 256.648, 301.17, 270.939, 319.654, 332.248, 301.217, 285.477, 280.349, 293.544, 357.054, 369.16, 339.055, 388.744, 358.245, 319.178, 382.924, 373.34, 335.991, 309.191, 364.157, 301.41, 371.432, 399.289, 326.016, 397.388, 288.858, 342.516, 335.319, 335.322, 394.989, 327.637, 427.408, 365.361, 381.775, 352.094, 368.851, 462.425, 322.904, 348.075, 461.21, 431.084, 504.514, 375.642, 389.887, 436.888, 433.888, 414.913, 534.769, 388.527, 421.878, 417.144, 373.666, 730.218, 419.132, 544.259, 432.168, 457.13, 369.602, 552.159, 437.885, 518.338, 390.776, 429.244, 499.338, 397.78, 412.749, 475.994, 480.378, 407.168, 427.387, 506.159, 487.854, 447.81, 515.54, 659.059, 400.914, 534.857, 390.718, 496.214, 375.08, 451.141, 493.751, 422.317, 542.92, 563.962, 529.959, 418.286, 464.256, 694.508, 411.452, 548.139, 575.872, 522.319, 519.497, 466.078, 569.271, 483.795, 494.954, 603.081, 489.815, 470.166, 475.635, 491.49, 528.274, 522.054, 618.212, 552.027, 665.568, 440.118, 464.903, 607.399, 544.317, 541.528, 445.71, 419.206, 476.831, 642.232, 532.759, 488.798, 502.22, 544.362, 535.479, 616.395, 631.991, 556.086, 647.834, 630.497, 552.957, 908.596, 544.734, 754.564, 1485.41, 668.587, 649.459, 731.894, 575.28, 759.04, 568.499, 1319.43, 614.427, 3202.07, 564.564, 518.81, 751.68, 593.121, 1783.26, 1401.85, 972.972, 544.499, 606.156, 1195.6, 4500.25, 664.562, 3651.46, 785.992, 939.369, 644.372, 796.298, 633.171, 803.795, 793.294, 778.141, 539.885, 718.626, 687.335, 653.98, 751.754, 801.192, 696.259, 969.323, 778.148, 7982.28, 1118.87, 681.179, 662.13, 550.188, 721.26, 981.586, 755.151, 630.405, 795.631, 819.952, 817.23, 695.152, 803.128, 704.722, 696.031, 741.046, 805.454, 919.877, 711.25, 827.666, 822.344, 623.384, 742.878, 742.69, 678.859, 686.306, 549.334, 1118.79, 822.269, 699.564, 620.688, 721.878, 1181.84, 953.315, 864.521, 1349.81, 1712.45, 1007.77, 832.578, 879.179, 2808.38, 988.018, 1162.4, 914.802, 796.529, 708.606, 1010.89, 1030.66, 799.518, 742.42, 755.2, 1903.55, 870.078, 3638.92, 703.896, 800.459, 675.43, 985.874, 893.864, 904.869, 882.373, 574.698, 2779.41, 1080.82, 1661.33, 1074.14, 911.788, 849.52, 764.187, 862.997, 1000.74, 827.101, 1225.71, 823.43, 781.102, 850.158, 837.465, 751.694, 1481.05, 816.679, 762.208, 1880.89, 909.306, 783.314, 968.904, 1106.25, 1028.02, 721.619, 1192.91, 936.17, 1240.31, 1174.5, 981.433, 3311.92, 1524.55, 822.314, 812.81, 903.961, 779.338, 1103.26, 998.869, 910.607, 5632.74, 763.019, 976.626, 992.1, 655.682, 826.459, 863.396, 924.47, 941.12, 1022.68, 924.631, 5780.49, 2481.78, 1758.1, 1162.22, 2974.27, 1595.88, 1195.61, 1251.26, 1042.07, 1178.8, 2607.35, 2735.3, 944.633, 1072.22, 887.302, 862.986, 886.724, 1017.15, 947.866, 995.208, 952.342, 1045.05, 920.053, 856.22, 2621.6, 893.879, 1862.62, 796.807, 1349.45, 2185.91, 1348.78, 1795.19, 905.327, 2030.92, 2105.52, 942.938, 1384.96, 1690.55, 1358.35, 1006.9, 2960.64, 926.689, 945.889, 1503.38, 1168.73, 1087.08, 1011.92, 1145.74, 1282.87, 1544.69, 3835.41, 1018.48, 1173.48, 1160.03, 2044.8, 3277.63, 1180.24, 2086.1, 1251.64, 1448.11, 1097.17, 1255.76, 1223.16, 1131.42, 1176.78, 1118.82, 1080.65, 2580.6, 1489.85, 1593.38, 5430.86, 1276.54, 1091.84, 1299.65, 1018.38, 1564.47, 1579.51, 1076.37, 2085.27, 1034.91, 999.119, 2335.36, 1213.25, 1538.77, 1406.31, 1175.4, 1211.55, 1423.57, 1743.66, 5538.9, 1585.17, 1201.19, 1953.54, 2555.81, 1415.58, 1848.51, 2965.84, 1564.67, 1103.95, 1270.63, 1879.78, 1213.05, 1234.76, 1314.46, 3848.28, 1271.63, 2091.85, 1286.01, 1079.93, 2226.59, 3414.77, 1529.67, 1669.1, 1646.42, 1557.82, 1099.57, 1854.07, 1599.95, 3100.92, 2164.93, 2562.61, 1728.05, 1949.89, 1193.68, 1622.54, 2326.23, 1341.91, 3928.82, 2091.62, 1356.52, 1965.56, 2807.35, 1246.31, 1264.15, 1539.56, 1823.71, 3119.86, 5828.19, 5176.07, 3019.6, 1838.14, 1366.31, 5335.13, 1192.66, 1690.92, 1617.05, 1166.57, 1241.35, 2778.06, 2907.56, 1609.74, 1830.34, 1165.31, 1415.4, 1615.45, 1304.62, 1705.46, 2452.16, 1390.93, 1452.87, 2142.46, 2776.82, 2189.04, 3918.43, 3637.24, 1349.79, 3387.99, 3144.74, 1712.83, 1999.97, 2042.04, 7917.52, 2881.8, 1907.58, 1638.32, 1792.89, 2863.31, 4097.35, 4612.56, 3116.12, 2030.11, 1919.37, 1649.65, 1602.22, 2309.44, 3677.2, 2495.26, 2926.19, 6291.05, 2135.25, 3775.53, 1601.69, 2217.54, 3342.02, 2023.12, 1478.35, 6296.56, 3892, 1691.46, 2681.33, 3456.17, 2568.47, 2120.75, 3798.22, 2657.88, 4106.52, 2612.56, 2844.14, 4850.65, 1430.39, 2964.31, 1810.39, 3609.16, 1629.88, 7575.57, 2320.74, 2635.93, 3973.07, 5315.29, 1671.15, 1534.37, 5788.9, 3325.49, 2493.2, 8448.85, 3311.26, 2509.87, 4869.25, 3226.36, 3999.4, 4304.28, 4523.98, 7844.81, 1473.16, 1507.47, 4583.93, 2198.16, 2247.5, 3446.98, 2615.76, 12054.3, 1830.98, 3631.21, 3670.64, 4327.7, 2937.81, 5269.54, 1699.8, 6314.94, 6553.87, 1914.84, 1928.89, 5748.95, 4270.81, 4758.11, 4113.63, 2748.27, 2658.14, 3000.25, 1966.03, 3611.89, 1596.34, 3075.45, 1866.28, 5328.12, 6304.22, 3134.68, 9403.06, 1614.41, 2087.56, 1815.88, 4341.92, 3445.52, 2396.51, 4179.15, 6894.4, 3393.81, 3275.02, 7414.71, 8598.95, 7129.49, 2309.23, 1931.73, 10613.5, 4229.6, 4108.86, 2194.25, 4752.14, 3737.5, 2989.61, 2809.62, 3654.11, 6839.03, 7223.2, 3666.71, 2537.77, 6530.08, 5155.73, 3825.89, 3669.56, 4786.54, 2224.09, 4292.35, 4140.13, 3643.37, 3470.39, 4035.29, 2968.43, 5409.74, 6754.99, 5864.92, 10933.6, 2828.22, 5644.21, 4620.01, 2255.69, 2704.18, 2181.01, 4864.44, 5072.57, 8026.67, 3656.28, 2461.51, 2405.89, 5543.88, 4209.3, 4717.81, 5739.54, 3451.57, 2997.77, 13765, 3320.01, 7224.06, 7084.87, 3554.78, 6693.37, 8615.22, 7470.14, 4391.01, 4767.37, 8703.69, 2252.06, 3541.45, 5997.24, 2607.49, 5311.51, 3026.66, 3918.51, 5693.03, 5132.71, 5201.56, 10476, 4821.1, 3234.2, 3294.77, 4630.84, 4041.98, 7115.59, 11785.6, 3565.96, 6621.33, 20407.9, 5433.45, 2703.45, 10007.8, 5326.85, 6272.38, 3494.43, 4259.18, 4603.36, 5388.81, 3658.01, 5085.5, 4615.56, 16086.3, 16260.4, 6842.95, 3446.4, 6334.88, 5240.87, 7311.74, 11392.7, 1876.52, 7478.41, 3381.09, 4108.21, 7269.09, 10782.6, 6068.49, 4487.13, 7114.7, 8997.39, 4570.71, 16641.6, 6814.84, 3886.56, 6127.02, 2585.5, 3917.4, 3413.97, 4442.81, 8207.66, 4063.47, 3811.29, 3734.68, 6296.6, 7503.17, 10680.8, 7539.06, 3986.81, 10781.6, 6962.5, 6425.3, 10562.3, 13366.6, 12552.4, 6001.68, 19900.7, 11956.9, 3176.03, 2380.96, 3115.39, 15961.4, 3938.06, 4340.5, 3028.7, 5089.46, 11230.3, 5852.19, 10024.4, 8409.63, 4379.22, 4913.3, 6157.2, 7273.01, 4537.48, 14111.1, 13303.7, 10577.9, 6913.72, 2846.91, 12525.9, 3931.88, 4462.03, 9984.29, 9752.36, 2725.35, 5059.33, 4304.26, 3327.59, 3390.26, 16631.9, 13651.6, 2371.04, 4727.9, 8292.02, 6402.11, 4419.96, 12262.1, 9112.48, 6518.5, 4646.04, 13547.2, 8139.23, 7624.6, 4195.6, 9191.78, 3750.88, 10454.3, 6507.84, 4442.82, 6122.25, 6011, 19244.4, 8574.74, 6002.78, 9943.97, 4847.22, 7891.93, 3803.11, 17447.8, 6773.91, 10253.1, 5771.36, 5503.64, 7510.83, 6147.93, 9959.12, 11970.6, 7965.36, 5033.62, 27189.2, 9229.5, 4872.8, 3241.48, 7316.26, 8280, 8005.54, 4728.3, 8360.65, 11680.8, 8218.18, 5830.89, 5594.35, 13569, 5028.8, 7746.55, 7670.71, 19262.7, 9603.29, 11557.7, 14870.5, 12767.1, 13255.7, 17500.5, 12590, 4992.84, 12715.9, 7210.16, 4168.29, 16639.4, 13599.9, 4139.37, 3785.42, 5704.85, 5439.65, 13971.3, 3762.33, 4374.08, 9569.17, 6199.33, 10179.3, 12126.5, 9649.45, 3715.13, 5203.61, 23816.7, 9930.04, 12322.6, 9190.49, 3823.63, 5516.49, 6055.1, 8989.35, 13634.2, 13791.9, 6025.28, 8336.71, 5136, 9868.08, 11792.3, 4832.28, 6645.88, 20551.1, 4684.11, 5039.78, 11720.8, 6074.26, 3335.05, 10521.1, 26510, 4801.2, 8737.12, 4609.15, 6041.05, 25055.9, 25844.6, 25861.2, 18009.7, 7221.17, 11918.5, 10768.4, 29350.3, 7173.33, 6317.56, 4692.21, 4669.18, 7043.3, 13694.9, 6360.17, 22364.7, 3627.39, 7531.27, 17960.7, 12347.1, 19580.1, 7089.16, 5825.73, 15633.3, 16603.8, 6586.78, 5785.7, 9763.18, 13253.1, 5253.04, 11735.8, 9720.13, 6401.54, 18333.7, 14621.5, 5992.96, 7741.54, 11195.9, 6650.55, 5356.1, 4842.89, 16276.8, 15019.3, 5933.35, 6296.97, 6981.5, 8512.69, 7146.71, 6315.37, 4146.48, 6461.28, 30556.3, 4294.56, 17691.1, 24705.5, 10257.1, 19273.6, 15566.4, 11210.2, 30297.2, 5642.38, 10706.6, 18813.7, 12438.3, 13795.5, 6421.28, 10322.7, 23208.3, 18108.9, 13418.3, 10807.1, 13640.4, 26281.8, 23732.1, 7648.48, 5926.17, 6616.85, 29670.6, 4806.85, 9305.09, 5721.02, 16392.4, 10546.4, 9401.4, 16186.4, 10887.5, 14458.4, 6816.34, 6977.8, 7498.08, 5811.33, 6006.59, 10972.9, 10231.7, 8732.59, 15055.1, 13363, 12366.3, 4216.73, 12656.3, 6929.03, 11791.6, 10903.2, 6252.38, 6138.71, 7597.14, 26895, 10828.4, 12823.9, 24627.2, 9034.32, 22974.9, 8085.04, 7704.27, 18614.2, 6570.93, 6472.43, 10846.4, 25123.2, 57256.3, 9421.02, 6578.18, 12370.4, 14195.6, 8046.25, 8005.45, 25391, 21918.5, 8465.74, 20172.4, 30983.6, 5232.47, 12445.6, 11695.2, 19476.5, 19112.2, 12397.9, 5680.45, 7871.87, 47926.3, 9765.94, 29445.4, 45440.3, 12634.3, 14328.3, 23839.7, 31662.3, 10836.6, 16930.1, 7589.44, 10763.9, 11733, 7534.51, 12060.8, 9875.57, 8301.82, 19196.6, 12293.1, 15561.5, 11871.2, 5106.24, 25279.4, 20730.6, 8600.17, 11246.5, 13181.5, 16419.8, 8231.48, 7849.46, 17960.7, 23003, 23663.4, 8358.14, 5913.73, 8984.02, 8352.1, 16133.7, 8124.55, 5327.53, 20488.8, 14175.5, 5528.85, 6053.66, 10625.9, 4980.22, 12491.5, 9742.6, 8354.43, 5675.34, 10168.6, 17026.1, 9435.96, 13865.6, 12797.3, 9053.26, 8954.85, 17201.9, 7106.56, 9362.85, 6998.84, 11967.1, 31244.1, 24050.9, 17028.2, 32316.7, 10009.7, 7836.13, 8480.91, 25463.9, 21304.9, 5805.97, 10297.2, 5706.29, 8223.17, 32814.9, 28480.3, 12072.4, 12093.8, 9168.16, 17529.4, 17300.6, 25186.2, 21154.1, 18906.5, 7121.67, 31523.1, 8366.45, 11655.5, 21458.3, 8908.21, 12644.5, 11624.4, 9569.05, 7001.17, 22272.5, 40911.4, 10304.4, 14465.6, 8565.1, 24739.8, 7892.11, 11747.3, 33447.2, 21182.8, 13575.3, 15361.7, 8854.09, 10261.5, 8755.41, 7816.55, 10993.4, 37046.5, 14951.6, 6005.82, 13501.3, 24416.4, 12443.2, 16649.2, 7854, 6240.1, 9222.02, 23789.8, 9628.39, 11780.8, 23777.2, 8788.43, 7072.3, 24300.4, 11859, 19529.4, 45210.9, 8829.19, 7870.68, 23305, 9361.76, 10662.8, 8070.79, 8537.35, 33413.5, 13826.1, 9345.24, 18866.9, 13711.2, 14217.7, 10168.3, 15041.9, 20169.5, 13257.5, 17871.6, 5905.26, 9612.06, 25622.1, 31365.3, 13241.5, 11369.7, 24425.3, 5689.89, 19635.7, 13732.7, 22342.8, 10017.9, 15599.5, 32567.4, 16142.7, 18512.3, 8834.15, 23455.6, 19698.4, 27524.9, 8901.04, 22773, 11066.4, 10873.7, 13438.7, 38184.2, 8344.97, 11239.4, 6283.08, 6982.74, 10731.4, 12197.2, 40870.6, 13392.3, 21136.8, 7796.26, 15061.1, 9691.8, 13753.7, 15396.7, 22461.8, 19063.3, 22393.9, 8278.5, 17912.4, 17758.9, 8848.28, 33919.8, 9518.7, 19676, 7550.28, 27934.7, 10987.2, 28049.3, 9067.35, 11134.9, 12307.6, 23674.6, 18118.8, 8819.25, 8160.73, 13587.2, 18553.3, 22848, 22399, 9634.46, 13065.9, 22462.4, 19535, 12537.2, 10795.6, 21650.3, 10546.1, 27125, 34004, 14697.3, 28401.5, 5831.2, 12728.8, 18453.3, 9971.29, 9758.39, 11704.6, 20566.2, 9462.97, 11180.2, 27646.3, 11989.9, 13027.9, 17242.3, 20564.3, 8067.26, 7318.57, 18755, 11458.6, 11866, 17652.8, 10430.9, 20301.1, 7407.44, 10808.4, 35280.4, 10833.4, 7784.61, 8221.33, 11424.7, 11242.1, 16163.7, 11225.8, 12294.6, 7200.96, 21223.6, 31192.9, 19736.2, 8517.91, 24202.4, 10361.8, 25029.2, 17076.1, 25614.9, 14527.2, 61544.8, 15054.2, 7667.52, 8848.25, 23324.7, 14465.9, 17091.8, 15393.1, 10318, 31147.6, 34972.5, 25174.8, 14511.3, 8865.98, 21429, 56683.1, 11009.8, 19162.7, 6401.7, 11927, 14821.3, 6778.92, 15316, 19592.2, 14345.8, 16342.8, 21257.5, 33659.5, 13821.6, 29340.4, 9955.45, 10685.7, 16732.4, 9265.82, 8059.16, 20197.4, 8281.89, 20855.4, 13533.7, 22775.4, 10791.8, 7720.08, 42787.7, 8568.18, 6969.09, 15694.2, 10639.2, 12247.6, 16993.1, 24838.9, 29335.4, 6185.51, 36545.5, 9339.22, 10829.3, 15334.1, 13503.3, 21428.5, 9629.9, 15916.9, 31413.3, 9064.86, 30587.9, 8402.92, 28743.5, 18383.6, 13569.6, 22824.9, 16517.8, 8190.76, 19400.2, 4730.03, 13392.5, 13393.5, 31966.4, 23726.9, 19271.7, 14806.9, 23484.9, 30866.6, 9110.56, 23806.9, 18550.2, 15910.1, 15621.2, 14081.2, 34886.3, 23870.2, 32029.5, 30719.7, 16056.6, 23192.2, 26757.7, 21768.6, 12511.5, 8784.75, 14515.7, 28733.2, 8632.3, 22732, 25896.7, 15362, 20332.7, 17358.4, 27038, 24353.9, 15068.3, 27484.7, 39466.8, 24142.3, 6307.02, 11335.5, 28975.6, 25303.2, 13731.9, 27663.9, 25872.4, 20825.2, 19302.2, 15705.6, 30808.1, 16278.4, 28513.3, 14291.2, 20744.2, 11673, 22721.7, 22422.3, 22098.2, 8956.28, 10004.9, 32071.3, 28017.1, 28265.7, 28725.1, 28953.3, 31764, 32169.8, 21129.8, 16821.2, 42663.2, 23187, 31124.7, 31220.8, 40816.1, 34489.2, 19993.5, 44941.1, 26758.2, 17778.6, 12266.6, 11261, 12974.9, 27199.9, 13541.8, 24856.7, 32484.7, 34090.6, 21547.9, 23589.1, 12429.6, 23916.2, 31477.1, 29346, 25907.2, 14148.2, 24976.1, 31342.1, 37217.6, 23500.9, 34573.3, 19904.1, 30226.9, 36336.6, 14333.8, 29573.9, 14323.2, 36763.4, 36818.4, 35910, 46990.9, 35820.2, 20800.3, 40492.4, 24054.8, 25968.5, 15253.1, 22145.1, 40786.8, 19415.2, 27323.4, 18141.1, 15140.1, 35600.7, 6791.52, 35339.5, 26702.2, 19508.5, 6376.43, 17818.6, 22945, 27103.3, 16952.1, 26170.7, 9109.42,]

x=[410.24, 411.01, 412.395, 413.695, 416.145, 416.31, 416.765, 417.105, 418.02, 418.28, 418.825, 418.855, 418.895, 420.14, 420.685, 421.17, 421.475, 421.565, 421.865, 422.175, 422.19, 422.505, 422.56, 422.625, 422.935, 422.985, 423.685, 423.865, 424.36, 424.44, 425.09, 425.095, 425.24, 425.34, 425.51, 425.705, 425.85, 425.875, 426.045, 426.06, 426.195, 426.225, 426.235, 426.26, 426.28, 426.475, 426.885, 427.325, 427.71, 428.125, 428.53, 428.945, 429.355, 429.77, 430.185, 430.595, 431.01, 431.42, 431.835, 432.245, 432.66, 433.07, 433.485, 433.895, 434.31, 434.72, 435.135, 435.545, 435.96, 436.375, 436.785, 437.2, 437.61, 438.025, 438.435, 438.85, 439.26, 439.675, 440.085, 440.5, 440.91, 441.325, 441.74, 442.15, 442.57, 442.975, 443.39, 443.8, 444.215, 444.625, 445.04, 445.45, 445.865, 446.275, 446.69, 447.105, 447.515, 447.93, 448.34, 448.755, 449.165, 449.58, 449.99, 450.405, 450.815, 451.23, 451.64, 452.055, 452.465, 452.88, 453.295, 453.705, 454.12, 454.53, 454.945, 455.355, 455.77, 456.18, 456.595, 457.005, 457.42, 457.83, 458.245, 458.66, 459.07, 459.485, 459.895, 460.31, 460.72, 461.135, 461.545, 461.96, 462.37, 462.785, 463.195, 463.61, 464.025, 464.435, 464.85, 465.26, 465.675, 466.085, 466.5, 466.91, 467.325, 467.735, 468.15, 468.56, 468.975, 469.39, 469.8, 470.215, 470.625, 471.04, 471.45, 471.865, 472.275, 472.69, 473.1, 473.515, 473.925, 474.34, 474.75, 475.165, 475.58, 475.99, 476.405, 476.815, 477.23, 477.64, 478.055, 478.465, 478.88, 479.29, 479.705, 480.115, 480.53, 480.945, 481.355, 481.77, 482.18, 482.595, 483.005, 483.42, 483.83, 484.245, 484.655, 485.07, 485.48, 485.895, 486.31, 486.72, 487.135, 487.545, 487.96, 488.37, 488.785, 489.195, 489.61, 490.02, 490.435, 490.845, 491.26, 491.675, 492.085, 492.5, 492.91, 493.325, 493.735, 494.155, 494.56, 494.975, 495.385, 495.805, 496.21, 496.625, 497.035, 497.45, 497.865, 498.275, 498.69, 499.1, 499.515, 499.925, 500.34, 500.75, 501.165, 501.575, 501.99, 502.4, 502.815, 503.235, 503.64, 504.055, 504.465, 504.88, 505.295, 505.705, 506.115, 506.53, 506.945, 507.355, 507.78, 508.18, 508.595, 509.005, 509.42, 509.83, 510.245, 510.655, 511.07, 511.48, 511.895, 512.305, 512.725, 513.13, 513.545, 513.96, 514.37, 514.785, 515.2, 515.61, 516.02, 516.435, 516.845, 517.265, 517.67, 518.09, 518.495, 518.91, 519.32, 519.74, 520.155, 520.56, 520.975, 521.385, 521.8, 522.21, 522.625, 523.035, 523.45, 523.86, 524.275, 524.685, 525.105, 525.515, 525.925, 526.34, 526.75, 527.165, 527.58, 527.995, 528.405, 528.815, 529.225, 529.64, 530.05, 530.465, 530.885, 531.295, 531.705, 532.115, 532.535, 532.94, 533.355, 533.765, 534.18, 534.6, 535.01, 535.425, 535.835, 536.25, 536.655, 537.075, 537.48, 537.895, 538.31, 538.725, 539.13, 539.545, 539.965, 540.375, 540.79, 541.195, 541.61, 542.02, 542.445, 542.85, 543.26, 543.67, 544.085, 544.5, 544.91, 545.32, 545.735, 546.19, 546.56, 546.97, 547.39, 547.8, 548.21, 548.625, 549.035, 549.465, 549.865, 550.275, 550.69, 551.1, 551.51, 551.94, 552.335, 552.75, 553.17, 553.575, 553.99, 554.405, 554.835, 555.235, 555.645, 556.05, 556.465, 556.875, 557.29, 557.7, 558.115, 558.535, 558.94, 559.36, 559.765, 560.195, 560.595, 561.025, 561.43, 561.83, 562.25, 562.66, 563.08, 563.485, 563.895, 564.305, 564.72, 565.15, 565.555, 565.955, 566.385, 566.78, 567.21, 567.625, 568.03, 568.44, 568.85, 569.28, 569.67, 570.13, 570.515, 570.92, 571.325, 571.77, 572.15, 572.56, 572.975, 573.385, 573.81, 574.23, 574.62, 575.045, 575.455, 575.87, 576.275, 576.69, 577.1, 577.51, 577.925, 578.355, 578.775, 579.165, 579.585, 579.995, 580.4, 580.815, 581.23, 581.64, 582.05, 582.485, 582.875, 583.29, 583.72, 584.125, 584.535, 585, 585.36, 585.765, 586.18, 586.595, 587.02, 587.42, 587.83, 588.24, 588.655, 589.07, 589.49, 589.89, 590.305, 590.735, 591.13, 591.54, 591.965, 592.38, 592.78, 593.2, 593.62, 594.03, 594.435, 594.855, 595.255, 595.685, 596.095, 596.505, 596.91, 597.32, 597.735, 598.145, 598.57, 598.98, 599.395, 599.795, 600.21, 600.63, 601.055, 601.445, 601.9, 602.285, 602.695, 603.14, 603.51, 603.985, 604.38, 604.775, 605.16, 605.625, 605.99, 606.4, 606.81, 607.24, 607.64, 608.07, 608.465, 608.895, 609.295, 609.72, 610.145, 610.535, 610.94, 611.365, 611.765, 612.18, 612.63, 613.02, 613.43, 613.94, 614.27, 614.68, 615.075, 615.5, 615.915, 616.315, 616.725, 617.13, 617.55, 617.99, 618.39, 618.78, 619.21, 619.61, 620.025, 620.43, 620.855, 621.265, 621.67, 622.095, 622.51, 622.905, 623.325, 623.73, 624.155, 624.56, 624.975, 625.4, 625.83, 626.235, 626.635, 627.04, 627.465, 627.86, 628.27, 628.71, 629.145, 629.51, 629.925, 630.345, 630.775, 631.18, 631.59, 632, 632.4, 632.81, 633.235, 633.67, 634.05, 634.49, 634.93, 635.29, 635.705, 636.15, 636.59, 636.965, 637.37, 637.78, 638.215, 638.6, 639.01, 639.44, 639.825, 640.255, 640.68, 641.09, 641.485, 641.915, 642.32, 642.725, 643.13, 643.59, 643.97, 644.455, 644.82, 645.21, 645.605, 646.025, 646.5, 646.845, 647.26, 647.68, 648.085, 648.515, 648.915, 649.365, 649.79, 650.19, 650.59, 651.1, 651.415, 651.8, 652.205, 652.635, 653.095, 653.505, 653.885, 654.295, 654.685, 655.105, 655.51, 655.99, 656.35, 656.75, 657.185, 657.575, 658.05, 658.41, 658.81, 659.23, 659.635, 660.08, 660.5, 660.875, 661.285, 661.75, 662.125, 662.53, 662.995, 663.415, 663.825, 664.19, 664.66, 665.01, 665.43, 665.825, 666.245, 666.665, 667.13, 667.52, 667.895, 668.305, 668.72, 669.2, 669.575, 669.99, 670.375, 670.795, 671.23, 671.605, 672.08, 672.435, 672.925, 673.31, 673.685, 674.085, 674.515, 675.035, 675.365, 675.74, 676.165, 676.61, 676.995, 677.4, 677.815, 678.215, 678.65, 679.085, 679.57, 679.935, 680.275, 680.685, 681.12, 681.52, 681.92, 682.445, 682.765, 683.19, 683.57, 683.99, 684.4, 684.835, 685.24, 685.76, 686.045, 686.48, 686.885, 687.325, 687.705, 688.135, 688.53, 688.935, 689.405, 689.81, 690.195, 690.62, 691.005, 691.455, 691.835, 692.38, 692.655, 693.115, 693.48, 693.89, 694.31, 694.715, 695.145, 695.545, 695.985, 696.48, 696.78, 697.195, 697.635, 698.015, 698.49, 698.855, 699.33, 699.695, 700.145, 700.53, 700.91, 701.335, 701.825, 702.16, 702.595, 702.975, 703.42, 703.815, 704.23, 704.63, 705.07, 705.445, 705.86, 706.3, 706.68, 707.15, 707.555, 707.955, 708.355, 708.79, 709.255, 709.595, 710.025, 710.43, 710.83, 711.25, 711.725, 712.265, 712.47, 712.88, 713.305, 713.74, 714.11, 714.545, 714.955, 715.46, 715.84, 716.22, 716.685, 717.01, 717.46, 717.84, 718.295, 718.65, 719.065, 719.575, 719.96, 720.325, 720.72, 721.16, 721.54, 721.975, 722.41, 722.8, 723.205, 723.6, 724.015, 724.455, 724.85, 725.25, 725.695, 726.13, 726.53, 727.04, 727.42, 727.755, 728.18, 728.555, 728.99, 729.38, 729.85, 730.235, 730.645, 731.12, 731.45, 732.125, 732.295, 732.725, 733.15, 733.55, 733.94, 734.385, 734.825, 735.27, 735.57, 736, 736.405, 736.805, 737.245, 737.685, 738.045, 738.465, 738.875, 739.325, 739.715, 740.26, 740.53, 740.99, 741.39, 741.81, 742.205, 742.615, 743.01, 743.42, 743.865, 744.245, 744.68, 745.065, 745.495, 745.89, 746.3, 746.785, 747.17, 747.545, 748.015, 748.37, 748.83, 749.21, 749.6, 750.03, 750.425, 750.875, 751.285, 751.76, 752.085, 752.5, 752.905, 753.36, 753.755, 754.245, 754.575, 755.025, 755.46, 755.865, 756.225, 756.64, 757.085, 757.445, 757.895, 758.295, 758.685, 759.09, 759.525, 759.95, 760.365, 760.765, 761.155, 761.575, 762.015, 762.405, 762.865, 763.24, 763.66, 764.055, 764.68, 764.985, 765.295, 765.72, 766.18, 766.605, 766.965, 767.455, 768.04, 768.195, 768.705, 769.055, 769.745, 769.83, 770.265, 770.69, 771.075, 771.565, 772.02, 772.43, 772.76, 773.28, 773.61, 774.04, 774.425, 774.845, 775.235, 775.78, 776.04, 776.47, 776.855, 777.26, 777.675, 778.145, 778.525, 778.92, 779.33, 779.74, 780.25, 780.665, 781.035, 781.53, 781.86, 782.245, 782.755, 783.05, 783.53, 784.32, 784.53, 784.77, 785.13, 785.525, 785.96, 786.4, 787.05, 787.185, 787.85, 788.02, 788.605, 788.925, 789.315, 789.63, 790.15, 790.705, 791.14, 791.335, 791.795, 792.12, 792.54, 793.045, 793.37, 793.91, 794.195, 794.63, 795.275, 795.495, 795.84, 796.475, 796.805, 797.105, 797.57, 797.9, 798.3, 798.965, 799.16, 799.565, 799.985, 800.48, 800.79, 801.305, 801.6, 802.035, 802.435, 802.85, 803.25, 803.755, 804.285, 804.495, 805.02, 805.415, 806.105, 806.145, 806.59, 807.15, 807.42, 807.9, 808.255, 808.72, 809.105, 809.7, 810.08, 810.345, 810.71, 811.13, 811.715, 812.02, 812.405, 812.905, 813.355, 813.59, 814.1, 814.545, 814.83, 815.61, 815.68, 816.05, 816.675, 817.22, 817.395, 817.915, 818.355, 818.555, 818.94, 819.46, 819.885, 820.19, 820.615, 821.065, 821.46, 821.825, 822.275, 822.96, 823.115, 823.54, 823.935, 824.365, 824.755, 825.12, 825.575, 825.95, 826.455, 826.89, 827.45, 827.75, 828.155, 828.43, 828.865, 829.315, 829.77, 830.09, 830.555, 831.07, 831.32, 831.745, 832.14, 832.555, 833.345, 833.455, 834.12, 834.305, 834.915, 835.06, 835.445, 835.94, 836.315, 836.835, 837.175, 837.55, 838.415, 838.425, 838.785, 839.265, 839.69, 840.045, 840.63, 840.865, 841.515, 842.465, 842.6, 842.77, 842.875, 843.44, 843.825, 844.11, 844.57, 844.955, 845.45, 845.85, 846.46, 846.675, 847.27, 847.425, 848.11, 848.64, 848.79, 849.1, 849.715, 849.92, 850.485, 850.99, 851.125, 851.6, 852.315, 852.56, 852.84, 853.22, 853.7, 854.245, 854.88, 854.905, 855.325, 855.795, 856.11, 856.635, 856.96, 857.33, 857.82, 858.38, 858.55, 858.96, 859.57, 859.82, 860.335, 860.735, 861.165, 861.665, 861.9, 862.39, 862.685, 863.375, 863.505, 863.96, 864.425, 864.78, 865.155, 865.95, 865.98, 866.94, 866.955, 867.27, 867.74, 868.505, 868.965, 869, 869.68, 869.85, 870.115, 870.585, 871.105, 871.89, 872.035, 872.2, 872.63, 872.995, 873.47, 873.925, 874.25, 875.02, 875.055, 875.53, 875.98, 876.395, 876.78, 877.325, 877.595, 878.005, 878.715, 879.12, 879.215, 879.68, 880.055, 880.535, 880.875, 881.295, 881.725, 882.735, 882.76, 882.91, 883.625, 883.76, 884.18, 884.575, 884.965, 885.385, 885.785, 886.295, 886.715, 887.155, 887.59, 888.09, 888.275, 889.125, 889.345, 889.57, 890.355, 890.665, 890.89, 891.2, 891.715, 891.995, 892.45, 892.835, 893.345, 893.9, 894.08, 894.525, 895.21, 895.435, 895.73, 896.345, 896.895, 897, 897.34, 897.825, 898.19, 898.705, 899.085, 899.5, 900.68, 900.735, 901.07, 901.185, 901.97, 902.405, 902.895, 902.935, 903.22, 903.56, 904.62, 904.65, 905.15, 905.455, 906.02, 906.215, 906.43, 906.935, 907.34, 907.685, 908.27, 908.605, 908.93, 909.72, 909.725, 910.245, 911.465, 911.635, 911.785, 911.975, 912.28, 912.67, 913.585, 914.305, 914.585, 914.605, 914.895, 915.24, 915.95, 916.5, 916.505, 916.82, 917.205, 917.695, 917.98, 918.495, 918.855, 919.285, 919.76, 920.075, 920.48, 921.1, 921.28, 921.71, 922.105, 922.625, 923.375, 923.425, 923.995, 924.565, 924.63, 925.155, 926.155, 926.18, 927.305, 927.695, 928.05, 928.115, 928.115, 928.655, 928.9, 929.145, 929.695, 930.05, 930.365, 930.96, 931.775, 931.865, 932.04, 932.695, 933.96, 934.23, 934.34, 934.405, 934.915, 935.08, 935.52, 935.815, 936.41, 937.285, 937.58, 937.7, 937.965, 938.335, 938.69, 939.09, 939.46, 940.785, 940.96, 941.18, 941.2, 941.675, 942.185, 942.735, 943.085, 943.26, 944.065, 944.62, 944.765, 945.12, 945.22, 945.82, 946.335, 946.59, 947.025, 947.61, 947.765, 948.115, 948.685, 949.46, 949.52, 950.675, 950.975, 951.305, 951.875, 951.98, 952.315, 952.335, 952.805, 953.34, 953.585, 954.385, 954.44, 955.15, 955.43, 955.65, 956.055, 956.49, 957.215, 957.69, 958.485, 958.615, 959.445, 959.48, 959.7, 960.395, 960.825, 961.705, 961.93, 962.035, 962.17, 962.34, 963.415, 964.5, 964.63, 964.75, 965.44, 967.23, 967.39, 968.09, 968.55, 968.94, 969.83, 969.95, 970.11, 971.485, 974.025, 975.9, 976.43, 976.5, 976.71, 977.055, 978.24, 978.885, 979.625, 979.64, 979.715, 979.755, 979.775, 979.84, 979.97, 980.075, 980.075, 980.3, 980.51, 981.005, 981.005, 981.275, 981.42, 981.62, 981.735, 982.085, 982.65, 982.67, 982.68, 983.345, 983.62, 983.8, 984.155, 984.345, 985.815, 986.055, 987.83, 988.37, 989.885, 990.015, 991.34, 991.88, 992.02, 992.245, 993.56, 993.615, 995.66, 996.01, 996.075, 996.645, 997.12, 997.65, 1002.22, 1002.28, 1003.29, 1003.71, 1004.59, 1004.81, 1004.84, 1005.11, 1005.72, 1005.9, 1006.33, 1006.77, 1007.72, 1008.14, 1008.38, 1008.84, 1008.92, 1008.99, 1009.23, 1009.47, 1010.53, 1011.36, 1011.51, 1011.59, 1011.83, 1012.12, 1012.41, 1012.82, 1012.94, 1013.14, 1014.59, 1014.84, 1015.08, 1016.06, 1016.23, 1016.38, 1017.28, 1017.47, 1017.91, 1018.73, 1019.59, 1019.68, 1020.42, 1024.01, 1025.77, 1025.92, 1026.68, 1026.83, 1027.18, 1029.25, 1030.34, 1030.7, 1030.86, 1032.05, 1032.86, 1033.72, 1034.8, 1035.36, 1035.68, 1036.74, 1037, 1037.45, 1037.5, 1038.09, 1038.59, 1038.86, 1038.91, 1041.13, 1041.92, 1043.14, 1043.21, 1045.17, 1045.81, 1050.91, 1053.62, 1053.72, 1054.43, 1056.7, 1059.88, 1060.68, 1060.89, 1061.4, 1063.99, 1064.14, 1064.7, 1065.07, 1066.46, 1067.19, 1067.26, 1070.3, 1071.97, 1072.61, 1074.89, 1078.49, 1079.09, 1079.21, 1079.81, 1080.83, 1081.08, 1081.67, 1083, 1083.8, 1090.1, 1090.3, 1090.71, 1091.04, 1091.21, 1091.99, 1092.56, 1092.9, 1093.09, 1094.56, 1095.44, 1099.66, 1102.86, 1105.59, 1106.13, 1107.1, 1107.15, 1108.66, 1109.69, 1111.29, 1116.37, 1116.69, 1118.33, 1120.63, 1123.3, 1133.65, 1134.49, 1137.62, 1140.27, 1140.63, 1143.89, 1147.59, 1149.28, 1153.2, 1155.81, 1162.72, 1165.56, 1183.04, 1233.27,]
plot.scatter(x,y)
x1 = np.linspace(np.min(x), np.max(x))
y1 = 12.9364 * x1 + -5712.19#(Covariance minimization)
y2 = 8.16669 * x1 + -3227.31#(Weighted fit of lowes 10 points, half of the set used for each run)
y3 = 9.1512 * x1 + -4339.77#(lowest points from each interval, linear fit over random number of points between 5 and 10)
y4 = 4.50882 * x1 + -1695.14#(real RANSAC)
y5 = 4.59387 * x1 + -1737.56#(real RANSAC with weighted fit instead of linear fit)
y6 = 5.8256 * x1+ -2271.89#(linear fit of 8 randomly selected points)



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
plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/geradenplots/pos11green.png', dpi=300)