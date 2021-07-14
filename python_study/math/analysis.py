import numpy as np
import math as m

# https://www.youtube.com/watch?v=aucrInPKUdg&list=PLCu17akcB-9098RQw5v8dO_7gsBhyEdMU&index=5
# 平均・偏差・標準偏差・分散・共分散の解説動画

# 身長|170|160|165|158|172|cm
# 体重| 68| 49| 56| 50| 77|kg

##########【平均】をpythonで実装 ##########

# np.average(配列) 平均を求める関数
height = np.array([170, 160, 165, 158, 172])
height_ave = np.average(height)
print('平均（身長）：')
print(np.average(height))
# または
weight = np.array([68, 49, 56, 50, 77])
weight_ave = np.sum(weight) / len(weight)
print('平均（体重）：')
print(weight_ave)


##########【偏差】をpythonで実装 ##########
height_deviation = height - height_ave # 元データから平均をマイナスすることで偏差を求める
print('偏差（身長）：')
print(height_deviation)

weight_deviation = weight - weight_ave
print('偏差（体重）：')
print(weight_deviation)


##########【分散】をpythonで実装 ##########
height_variance = np.sum(height_deviation **2) / len(height_deviation) # 偏差を2乗し、マイナスを消した後にデータ数で割る
print('分散（身長）：')
print(height_variance)

weight_variance = np.sum(weight_deviation **2) / len(weight_deviation)
print('分散（体重）：')
print(weight_variance)


##########【標準偏差】をpythonで実装 ##########
height_standard_deviation = m.sqrt(height_variance)
print('標準偏差（身長）：')
print(height_standard_deviation) # 分散した値をルート（平方根）にすることで標準偏差が求められる　※平均165cmから上下に5.4cm程度バラついていると言える

weight_standard_deviation = m.sqrt(weight_variance)
print('標準偏差（体重）：')
print(weight_standard_deviation)


##########【共分散】をpythonで実装 ##########
covariance = np.sum((height_deviation * weight_deviation) / len(height_deviation))
print('共分散：')
print(covariance)


##########【相関係数】をpythonで実装 ##########
correlation_coefficient = covariance / (height_standard_deviation * weight_standard_deviation) # 共分散 / (X（身長）の標準偏差 * Y（体重）の標準偏差)
print(correlation_coefficient)
# 相関係数は -1 <= correlation_coefficient <= 1 となり、-1に近いほど負の相関関係にあり、1に近いほど正の相関関係にある。0に近いと相関関係はないと判断できる
# 正の相関とはXが増えるとYも増える、またはXが減るとYも減る。　負の相関とはXが増えるとYが減り、Xが減るとYが増える。