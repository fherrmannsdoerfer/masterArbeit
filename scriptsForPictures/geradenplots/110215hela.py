import numpy as np
import matplotlib.pyplot as plot
#tetrahigh568

y=[381.938, 340.201, 393.614, 356.578, 934.978, 355.998, 343.156, 373.328, 395.021, 387.555, 363.129, 396.477, 377.581, 373.457, 405.55, 415.528, 453.61, 354.007, 339.609, 438.27, 401.802, 377.778, 373.551, 379.101, 443.858, 444.643, 354.525, 386.172, 426.348, 396.01, 392.208, 434.377, 348.27, 449.329, 371.174, 452.524, 383.612, 476.869, 355.285, 1331.04, 407.123, 379.169, 337.242, 473.312, 452.62, 425.03, 428.934, 437.256, 387, 304.933, 482.815, 477.126, 452.897, 342.009, 401.899, 408.74, 407.797, 434.371, 385.508, 399, 372.453, 380.316, 396.552, 344.91, 461.567, 439.615, 406.912, 405.95, 386.447, 466.199, 401.289, 448.16, 457.694, 345.291, 348.654, 430.84, 455.832, 417.368, 479.868, 388.494, 458.456, 396.318, 425.476, 469.802, 380.615, 400.352, 478.788, 391.134, 427.176, 361.72, 435.909, 449.188, 334.29, 392.705, 464.486, 436.945, 485.249, 482.797, 437.588, 420.332, 381.949, 420.3, 392.959, 415.334, 595.482, 535.003, 400.547, 522.055, 411.866, 462.05, 469.875, 427.448, 548.341, 496.519, 567.014, 419.088, 419.412, 366.992, 553.08, 567.834, 365.971, 447.704, 500.814, 488.061, 430.154, 590.273, 439.532, 449.57, 679.02, 539.284, 467.12, 575.88, 417.934, 608.02, 467.96, 426.795, 519.479, 529.036, 501.936, 485.279, 463.556, 528.926, 461.069, 531.765, 577.274, 466.997, 543.893, 533.998, 461.408, 535.311, 564.907, 614.067, 512.19, 532.096, 665.985, 515.218, 526.033, 615.812, 563.736, 541.489, 517.665, 717.654, 588.537, 510.063, 477.802, 531.904, 852.089, 492.688, 441.43, 562.462, 625.718, 515.848, 509.33, 584.566, 630.591, 700.646, 538.802, 538.35, 551.782, 498.606, 504.868, 747.287, 640.619, 584.084, 645.823, 564.335, 527.91, 521.938, 538.489, 524.774, 523.662, 500.642, 600.954, 498.089, 499.258, 1128.1, 578.144, 530.432, 932.813, 551.167, 597.855, 624.866, 735.41, 609.41, 625.078, 687.979, 596.454, 741.932, 652.573, 669.257, 645.594, 792.045, 728.462, 628.066, 750.514, 781.426, 595.47, 558.018, 653.608, 1637.84, 571.09, 707.51, 1083.73, 696.38, 734.46, 660.038, 806.222, 521.899, 667.69, 779.524, 1430.51, 712.411, 836.284, 747.271, 746.141, 646.794, 712.64, 671.607, 630.758, 598.684, 712.048, 721.118, 918.958, 1204.92, 796.447, 559.27, 734.426, 1094.19, 827.851, 965.464, 1866.14, 593.501, 761.254, 762.4, 613.079, 589.39, 691.348, 978.357, 1139.67, 668.663, 869.81, 1654.17, 2019.33, 1791.3, 1704.4, 697.08, 1430.05, 609.296, 1656.78, 597.982, 3383.75, 957.607, 860.937, 1233.39, 3623.95, 1726.77, 791.119, 863.248, 792.15, 2146.21, 813.933, 1649.33, 1703.67, 1381.25, 1783.97, 743.754, 789.999, 767.847, 1790.94, 1215.11, 3968.21, 812.502, 1007.4, 5002.73, 3705.53, 868.26, 1872.57, 795.613, 742.138, 5583.84, 2205.63, 701.004, 844.602, 1043.32, 1835.98, 802.949, 1000.67, 2156.41, 881.686, 1082.92, 1301.13, 1062.72, 2304.33, 861.76, 803.338, 1971.63, 1218.13, 931.102, 932.703, 901.317, 1003.46, 6162.65, 3539.78, 1195.87, 1136.53, 2326.72, 1665.52, 743.294, 744.793, 2290.5, 1803.64, 7462.09, 891.761, 3137.22, 1236.28, 1492.56, 947.028, 6537.36, 1049.81, 1346.47, 1164.15, 1364.22, 4015.16, 1192.7, 967.366, 3104.07, 1932.32, 3111.59, 6127.18, 1447.92, 1529.88, 2512.41, 2522.04, 4730.48, 1113.51, 3454.62, 1299.98, 1268.06, 1371.99, 2947.55, 1218.7, 2308.69, 5427, 1210.46, 3437.07, 2141.79, 1001.96, 20829.4, 1376.71, 1105.16, 3931.87, 3103.24, 2984.52, 3072.71, 6589.38, 1315.9, 2104.22, 1733.18, 3424.87, 1910.12, 5631.91, 2156.83, 1095.32, 2313.27, 1150.2, 2355.73, 7672.3, 1179.76, 3768.22, 2665.29, 5543.81, 2786.98, 2140.97, 1038.82, 7232.2, 1242.7, 2632.51, 1613.97, 1688.28, 8057.05, 2691.84, 3638.93, 12164.2, 3940.93, 1889.97, 1931.85, 2341.96, 3470.51, 2593.96, 1908.13, 1964.9, 2979.9, 2243.23, 2714.26, 1357.84, 2863.39, 4182.15, 9030.8, 7114.3, 1708.07, 6323.02, 1599.98, 1468.31, 4965.27, 1271.94, 1865.91, 5854.46, 6787.45, 4342.55, 3688.7, 2148.16, 2276.18, 2073.11, 5451.14, 7471.55, 17416, 2322.82, 3832.85, 1566.41, 1637.11, 3364.19, 3971.28, 3091.85, 2735.98, 1259.07, 9839.41, 5670.03, 3269.19, 2837.37, 20310.6, 7554.17, 2398.77, 2095.58, 8624.5, 1127.26, 10473.6, 4073.59, 3323.49, 7670.94, 3843.99, 4451, 2155.43, 2685.3, 5805.44, 4292.33, 4751.53, 3293.62, 10262.9, 6402.97, 3947.34, 1680.23, 4521.07, 8930.73, 5946.88, 8363.17, 2079.73, 4933.26, 2961.07, 2325.51, 11294.8, 2749.18, 6934.4, 9006.74, 2583.3, 7603.57, 3432.1, 4201.08, 4182.22, 4009.02, 4786.86, 1947.05, 3240.28, 7505.04, 3862.61, 3529.7, 3690.61, 6825.52, 2108.31, 3517.27, 2313.15, 2569.31, 1924.4, 3111.21, 5224.38, 8991.45, 3103.15, 3432.74, 5377.89, 16859.6, 8145.12, 3619.58, 3825.02, 4861.38, 5514.39, 3392.33, 9750.28, 3457.3, 4263.45, 7718.7, 5515.91, 4463.64, 3298.92, 5687.26, 5444.93, 5394.83, 10547.1, 5778.59, 5802.8, 9751.06, 11923.8, 2640.62, 6660.02, 8289.51, 8187.79, 5614.79, 9829.34, 6488.14, 12030.3, 10741.7, 5136.56, 7400.33, 4302.43, 9402.38, 11538.8, 8374.11, 3962.95, 18842.8, 9424.3, 10718.4, 6940.19, 7988.51, 4719.45, 6573.5, 9729.8, 7384.88, 25667.5, 5940.79, 5385.96, 10145.2, 10198.1, 10867.7, 3653.76, 3304.07, 12508, 10037.3, 4410.72, 5988.29, 10945.5, 11163.9, 16131.9, 8270.04, 11555.4, 7119.25, 8950.38, 9643.96, 1542.86, 32552.2, 6458.98, 8700.26, 20456.3, 2677.01, 12377.8, 21833.8, 14200.4, 6495.73, 8112.46, 12674.1, 4316.19, 13572, 11486.8, 9960.3, 23178.1, 17204.4, 7470.8, 6909.8, 14615.1, 12705.7, 5450.73, 12640.2, 15328.2, 9163.13, 5791.96, 7349.41, 9770.64, 5076.26, 10783.3, 7842.24, 6402.65, 7604.98, 7911.09, 13955.7, 11911.9, 15238.8, 10569.8, 32331.8, 9110.99, 18903.2, 7665.94, 16893.2, 12417.5, 8503.94, 11243, 18234.5, 8635.98, 18285.4, 5649.27, 18689.4, 6179.87, 3415.77, 17897.5, 24195, 8371.44, 12174.2, 12577, 6408.86, 11441.2, 27804, 14414.4, 10136.6, 8755.56, 21781.6, 10680.7, 8834.62, 12098.5, 10701.1, 19490, 5888.91, 13051.4, 13368.9, 26646.3, 26631, 3587.85, 27927.1, 4636.35, 15617.9, 12224.8, 5987.2, 10573.7, 12793, 21844.7, 17249, 5487.99, 5301.69, 6963.39, 56056, 7718.79, 14079.8, 22024.9, 9273.82, 20730.2, 27030.4, 4183.94, 24738.5, 11481, 62696.4, 15328.1, 25617.6, 10555, 10744.4, 11398.1, 34971.6, 2186.24, 31263.2, 13300.5, 10516.8, 25829.3, 16255.7, 10706.9, 9497.51, 32357.8, 22150.5, 19547.8, 5782.37, 10906.4, 3880.7, 17394.7, 15645.2, 3907.74, 32784.2, 15764.3, 10140, 20244.6, 10215.9, 9430.22, 9574.9, 4571.81, 17709.6, 8550.27, 14386.2, 27342.6, 3530.29, 8825.85, 12955.1, 13456.3, 29165.5, 19322.7, 12795.7, 11729.9, 17873.1, 15431, 10499.8, 12194.4, 10604.2, 39569.7, 10455.7, 20055, 21233.7, 7147.53, 14674.6, 6410.49, 5538.4, 13801.9, 11421.1, 12981.9, 9423.88, 2865.65, 21025.2, 30123.8, 7493.57, 4681.58, 11837, 13206.8, 11386.6, 7268.11, 23586.2, 13861.8, 18638, 3769.49, 15741.9, 17086, 27148.1, 15788.2, 29934.1, 10267.2, 15888.4, 4676.75, 3417.45, 10934.9, 11697.7, 15287.1, 34319.9, 4831.2, 15210.2, 39319.7, 12427.3, 14946.4, 20696, 13247.2, 19611.9, 16913.4, 22763.4, 18473.2, 16089.9, 13647.1, 19615.3, 4566.67, 4094.7, 28094.1, 29002.9, 17840.9, 7783.02, 41731.9, 18503.3, 3627.99, 18094.4, 22953.7, 39897.9, 3131.17, 5617.67, 16990.2, 2977.04, 35536.2, 4168.86, 35816.1, 32318.9, 4531.24, 5535.81, 62911, 6068.04, 19723.2, 47573.2, 20821.7, 51967.9, 40101.8, 39331.5, 4547.77, 4515.99, 8462.26, 52298.7, 23514.1, 59627.8, 5053.39, 5104.3, 63231.1, 9089.7, 11053.3, 4994.56, 5410.22, 26416.2, 3712.07, 8164.26, 5729.4, 8953.82, 7137.28, 4715.37, 7577.63, 7475.05, 6129.15, 6692.21, 5343.28, 7225.3, 8257.07, 7330.08]
x=[434.165, 434.405, 434.975, 435.35, 436.835, 437.545, 437.56, 438.05, 439.17, 440.755, 441.645, 442.23, 442.405, 442.73, 445.86, 446.455, 446.495, 446.615, 446.645, 446.99, 447.055, 447.11, 447.17, 447.17, 447.18, 447.205, 447.265, 447.305, 447.39, 447.64, 447.68, 447.805, 447.86, 447.875, 447.895, 447.895, 447.91, 447.925, 448.015, 448.08, 448.13, 448.215, 448.22, 448.22, 448.255, 448.3, 448.375, 448.38, 448.4, 448.415, 448.43, 448.435, 448.45, 448.47, 448.475, 448.51, 448.555, 448.595, 448.61, 448.64, 448.665, 448.685, 448.695, 448.7, 448.73, 448.735, 448.74, 448.745, 448.77, 448.785, 448.785, 448.8, 448.81, 448.815, 448.855, 448.86, 448.91, 448.915, 448.96, 448.965, 449.12, 449.35, 449.56, 449.78, 450.005, 450.22, 450.45, 450.66, 450.88, 451.1, 451.325, 451.545, 451.765, 451.985, 452.21, 452.43, 452.645, 452.865, 453.085, 453.305, 453.525, 453.745, 453.97, 454.19, 454.41, 454.63, 454.85, 455.07, 455.29, 455.51, 455.735, 455.95, 456.17, 456.395, 456.625, 456.835, 457.055, 457.28, 457.495, 457.715, 457.935, 458.16, 458.375, 458.595, 458.825, 459.045, 459.26, 459.48, 459.7, 459.92, 460.14, 460.36, 460.58, 460.8, 461.02, 461.245, 461.465, 461.685, 461.905, 462.125, 462.345, 462.565, 462.785, 463.005, 463.225, 463.445, 463.665, 463.89, 464.11, 464.33, 464.55, 464.77, 464.99, 465.21, 465.43, 465.65, 465.87, 466.09, 466.315, 466.535, 466.755, 466.975, 467.195, 467.415, 467.635, 467.855, 468.075, 468.295, 468.515, 468.74, 468.96, 469.18, 469.4, 469.62, 469.845, 470.06, 470.28, 470.5, 470.72, 470.94, 471.165, 471.385, 471.605, 471.825, 472.045, 472.265, 472.485, 472.705, 472.925, 473.145, 473.365, 473.59, 473.81, 474.03, 474.25, 474.47, 474.69, 474.91, 475.13, 475.35, 475.57, 475.79, 476.01, 476.235, 476.455, 476.675, 476.895, 477.115, 477.335, 477.555, 477.775, 477.995, 478.22, 478.435, 478.66, 478.88, 479.1, 479.32, 479.54, 479.76, 479.98, 480.2, 480.42, 480.64, 480.86, 481.085, 481.305, 481.525, 481.745, 481.965, 482.185, 482.405, 482.625, 482.845, 483.065, 483.285, 483.51, 483.73, 483.95, 484.175, 484.39, 484.61, 484.835, 485.05, 485.27, 485.49, 485.71, 485.93, 486.155, 486.375, 486.595, 486.815, 487.035, 487.255, 487.475, 487.7, 487.915, 488.135, 488.355, 488.585, 488.8, 489.02, 489.245, 489.465, 489.68, 489.9, 490.12, 490.345, 490.56, 490.78, 491.005, 491.23, 491.445, 491.665, 491.885, 492.105, 492.325, 492.545, 492.765, 492.985, 493.205, 493.43, 493.65, 493.87, 494.09, 494.31, 494.53, 494.75, 494.975, 495.195, 495.415, 495.635, 495.85, 496.075, 496.295, 496.515, 496.74, 496.955, 497.18, 497.395, 497.615, 497.84, 498.055, 498.28, 498.5, 498.725, 498.945, 499.16, 499.38, 499.6, 499.825, 500.04, 500.265, 500.48, 500.705, 500.925, 501.145, 501.365, 501.585, 501.805, 502.035, 502.245, 502.465, 502.685, 502.91, 503.13, 503.35, 503.575, 503.795, 504.01, 504.24, 504.45, 504.67, 504.89, 505.115, 505.345, 505.55, 505.77, 505.995, 506.23, 506.435, 506.655, 506.875, 507.1, 507.315, 507.54, 507.755, 507.995, 508.195, 508.42, 508.645, 508.86, 509.08, 509.305, 509.52, 509.74, 509.97, 510.18, 510.41, 510.62, 510.845, 511.07, 511.29, 511.52, 511.725, 511.945, 512.17, 512.4, 512.605, 512.825, 513.045, 513.28, 513.49, 513.715, 513.93, 514.15, 514.38, 514.6, 514.81, 515.03, 515.26, 515.48, 515.7, 515.94, 516.14, 516.355, 516.575, 516.795, 517.015, 517.235, 517.46, 517.68, 517.895, 518.125, 518.345, 518.56, 518.79, 519.02, 519.265, 519.445, 519.68, 519.885, 520.1, 520.325, 520.555, 520.845, 520.985, 521.21, 521.44, 521.665, 521.87, 522.085, 522.31, 522.525, 522.8, 522.97, 523.195, 523.41, 523.65, 523.86, 524.075, 524.295, 524.515, 524.73, 524.95, 525.215, 525.41, 525.625, 525.845, 526.055, 526.285, 526.495, 526.725, 526.935, 527.185, 527.38, 527.625, 527.865, 528.05, 528.265, 528.5, 528.71, 528.98, 529.14, 529.39, 529.595, 529.805, 530.035, 530.24, 530.475, 530.685, 530.93, 531.14, 531.355, 531.57, 531.785, 532.005, 532.225, 532.455, 532.67, 532.885, 533.12, 533.33, 533.55, 533.77, 534.06, 534.23, 534.43, 534.655, 534.92, 535.12, 535.31, 535.55, 535.76, 535.995, 536.205, 536.42, 536.705, 536.89, 537.135, 537.32, 537.53, 537.74, 537.97, 538.18, 538.46, 538.63, 538.88, 539.075, 539.28, 539.505, 539.725, 539.96, 540.21, 540.405, 540.605, 540.845, 541.045, 541.285, 541.495, 541.73, 541.93, 542.145, 542.365, 542.595, 542.805, 543.035, 543.25, 543.505, 543.745, 543.96, 544.13, 544.36, 544.58, 544.795, 545.035, 545.245, 545.49, 545.725, 545.895, 546.145, 546.335, 546.605, 546.835, 547.005, 547.235, 547.45, 547.655, 547.89, 548.1, 548.355, 548.55, 548.775, 549.02, 549.235, 549.43, 549.66, 549.865, 550.085, 550.325, 550.525, 550.78, 551.06, 551.195, 551.415, 551.64, 551.855, 552.095, 552.3, 552.505, 552.75, 552.955, 553.195, 553.41, 553.625, 553.845, 554.09, 554.37, 554.49, 554.715, 554.955, 555.25, 555.535, 555.665, 555.825, 556.035, 556.255, 556.505, 556.695, 556.935, 557.205, 557.355, 557.635, 557.85, 558.04, 558.25, 558.47, 558.79, 559.005, 559.12, 559.36, 559.57, 559.815, 560.035, 560.22, 560.455, 560.715, 560.945, 561.135, 561.34, 561.55, 561.81, 562.015, 562.295, 562.465, 562.74, 562.87, 563.29, 563.385, 563.71, 563.81, 564.015, 564.27, 564.45, 564.65, 564.965, 565.19, 565.34, 565.7, 565.77, 565.955, 566.33, 566.405, 566.81, 566.845, 567.06, 567.29, 567.575, 567.825, 568.03, 568.335, 568.445, 568.9, 569.13, 569.225, 569.365, 569.52, 569.76, 569.935, 570.53, 570.585, 570.8, 570.92, 571.085, 571.375, 571.47, 571.73, 571.92, 572.555, 572.855, 572.98, 573.275, 573.495, 573.58, 573.62, 573.77, 574.035, 574.25, 574.605, 574.76, 575.145, 575.485, 575.78, 575.81, 575.93, 576.225, 576.395, 576.585, 576.625, 577.08, 577.19, 577.26, 577.755, 577.945, 578.035, 578.525, 578.97, 579.005, 579.185, 579.245, 579.52, 579.565, 579.65, 579.855, 580.235, 580.3, 580.825, 580.975, 580.975, 581.58, 581.755, 581.86, 582.015, 582.085, 582.5, 582.6, 583.095, 583.16, 583.19, 583.415, 583.77, 583.86, 584.075, 584.305, 584.83, 584.885, 585.02, 585.275, 585.875, 586.425, 586.49, 586.615, 586.935, 586.995, 587.27, 587.325, 587.545, 587.58, 587.85, 588.02, 588.4, 588.775, 588.795, 589.19, 590.375, 590.42, 590.525, 590.78, 590.925, 592.23, 592.635, 592.67, 592.815, 593.09, 593.17, 594.025, 594.42, 594.665, 594.915, 594.98, 595.02, 595.04, 595.565, 595.995, 596.005, 596.5, 596.605, 596.755, 597.385, 598.32, 599.075, 599.155, 599.53, 599.835, 601.065, 601.37, 601.75, 602.8, 603.36, 603.45, 603.815, 603.995, 604.01, 604.315, 604.505, 605.04, 605.215, 605.235, 605.59, 606.015, 606.35, 606.605, 607.125, 607.79, 607.85, 608.83, 610.475, 611.025, 612.52, 612.545, 613.755, 614.3, 615.66, 616.105, 616.485, 616.755, 616.775, 617.665, 618.115, 618.33, 618.41, 618.41, 618.46, 619.075, 620.795, 621.93, 622.09, 622.68, 622.775, 623.465, 624.435, 624.905, 627.235, 628.805, 629.715, 630.64, 631.16, 631.255, 631.55, 631.94, 633.14, 635.15, 636.085, 637.405, 637.425, 638.405, 642.35, 642.52, 644.2, 644.245, 644.27, 647.02, 647.82, 647.835, 652.06, 653.53, 656.555, 658.235, 659.12, 660.1, 660.1, 676.095, 683.78, 705.285, 729.5, 737.325, 760.39, 781.25, 782.595, 800.685, 804.145, 829.43, 872.3]

plot.scatter(x,y)
x1 = np.linspace(np.min(x), np.max(x))
y1 = 21.4335 * x1 + -9331.27#(Covariance minimization)
y2 = 16.6857 * x1 + -7178.47#(Weighted fit of lowes 10 points, half of the set used for each run)
y3 = 19.9654 * x1 + -9468.54#(lowest points from each interval, linear fit over random number of points between 5 and 10)
y4 = 2.36989 * x1 + -667.451#(real RANSAC)
y5 = 4.1651 * x1 + -1455.98#(real RANSAC with weighted fit instead of linear fit)
y6 = 8.37265 * x1+ -3366.44#(linear fit of 8 randomly selected points)



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
plot.legend(loc = 0, prop={'size':15})
dx = np.max(x)-np.min(x)
plot.axis([np.min(x)-0.1*dx, np.max(x)+0.1*dx, np.min(y)-1000, 30000])
plot.savefig('/home/herrmannsdoerfer/MasterArbeit/pictures/geradenplots/110215hela.png', dpi=300)