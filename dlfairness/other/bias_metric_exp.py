import numpy as np
import sys

arr = np.empty((2, 2, 2)) # GT, Pred, Group
arr[0][0][0] = 80
arr[0][1][0] = 20
arr[1][0][0] = 10
arr[1][1][0] = 90
arr[0][0][1] = 60
arr[0][1][1] = 40
arr[1][0][1] = 70
arr[1][1][1] = 30

# SP, class 0 
sp_f = arr[:, 0, :].sum() / arr.sum()
sp_f_0 = arr[:, 0, 0].sum() / arr[:, :, 0].sum()
sp_f_1 = arr[:, 0, 1].sum() / arr[:, :, 1].sum()
bias_0 = abs(sp_f - sp_f_0) * (arr[:, :, 0].sum() / arr.sum()) + abs(sp_f - sp_f_1) * (arr[:, :, 1].sum() / arr.sum())

# class 1
sp_f = arr[:, 1, :].sum() / arr.sum()
sp_f_0 = arr[:, 1, 0].sum() / arr[:, :, 0].sum()
sp_f_1 = arr[:, 1, 1].sum() / arr[:, :, 1].sum()
bias_1 = abs(sp_f - sp_f_0) * (arr[:, :, 0].sum() / arr.sum()) + abs(sp_f - sp_f_1) * (arr[:, :, 1].sum() / arr.sum())

bias = (bias_0 + bias_1) / 2
print("SP:", bias_0, bias_1, bias)

# FP, class 0 
fp_f = arr[0, 1, :].sum() / arr[0, :, :].sum()
fp_f_0 = arr[0, 1, 0].sum() / arr[0, :, 0].sum()
fp_f_1 = arr[0, 1, 1].sum() / arr[0, :, 1].sum()
bias_0 = abs(fp_f - fp_f_0) * (arr[:, :, 0].sum() / arr.sum()) + abs(fp_f - fp_f_1) * (arr[:, :, 1].sum() / arr.sum())

# class 1
fp_f = arr[1, 0, :].sum() / arr[1, :, :].sum()
fp_f_0 = arr[1, 0, 0].sum() / arr[1, :, 0].sum()
fp_f_1 = arr[1, 0, 1].sum() / arr[1, :, 1].sum()
bias_1 = abs(fp_f - fp_f_0) * (arr[:, :, 0].sum() / arr.sum()) + abs(fp_f - fp_f_1) * (arr[:, :, 1].sum() / arr.sum())

bias = (bias_0 + bias_1) / 2
print("FP:", bias_0, bias_1, bias)

# DP, class 0 
dp_0 = arr[:, 0, 0].sum() / arr[:, :, 0].sum()
dp_1 = arr[:, 0, 1].sum() / arr[:, :, 1].sum()
#print(dp_0, dp_1)
bias_0 = abs(dp_0 - dp_1)

dp_0 = arr[:, 1, 0].sum() / arr[:, :, 0].sum()
dp_1 = arr[:, 1, 1].sum() / arr[:, :, 1].sum()
bias_1 = abs(dp_0 - dp_1)

bias = (bias_0 + bias_1) / 2
print("DP:", bias_0, bias_1, bias)

# DI, class 0 
di_0 = arr[:, 0, 0].sum() / arr[:, :, 0].sum()
di_1 = arr[:, 0, 1].sum() / arr[:, :, 1].sum()
bias_0 = 1.0 - min(di_0 / di_1, di_1 / di_0)

di_0 = arr[:, 1, 0].sum() / arr[:, :, 0].sum()
di_1 = arr[:, 1, 1].sum() / arr[:, :, 1].sum()
bias_1 = 1.0 - min(di_0 / di_1, di_1 / di_0)

bias = (bias_0 + bias_1) / 2
print("DI:", bias_0, bias_1, bias)

# EO-TP, class 0
eo_0 = arr[0, 0, 0].sum() / arr[0, :, 0].sum()
eo_1 = arr[0, 0, 1].sum() / arr[0, :, 1].sum()
bias_0 = abs(eo_0 - eo_1)
#print(eo_0, eo_1)

eo_0 = arr[1, 1, 0].sum() / arr[1, :, 0].sum()
eo_1 = arr[1, 1, 1].sum() / arr[1, :, 1].sum()
bias_1 = abs(eo_0 - eo_1)

bias = (bias_0 + bias_1) / 2
print("EO-TP:", bias_0, bias_1, bias)

# EO-FP, class 0
eo_0 = arr[0, 1, 0].sum() / arr[0, :, 0].sum()
eo_1 = arr[0, 1, 1].sum() / arr[0, :, 1].sum()
bias_0 = abs(eo_0 - eo_1)
#print(eo_0, eo_1)

eo_0 = arr[1, 0, 0].sum() / arr[1, :, 0].sum()
eo_1 = arr[1, 0, 1].sum() / arr[1, :, 1].sum()
bias_1 = abs(eo_0 - eo_1)

bias = (bias_0 + bias_1) / 2
print("EO-FP:", bias_0, bias_1, bias)

# BA
bs_0 = arr[0, :, 0].sum() / arr[0, :, :].sum()
bs_1 = arr[0, :, 1].sum() / arr[0, :, :].sum()
b_s_0 = bs_0 / (bs_0 + bs_1)
b_s_1 = bs_1 / (bs_0 + bs_1)
bt_0 = arr[:, 0, 0].sum() / arr[:, 0, :].sum()
bt_1 = arr[:, 0, 1].sum() / arr[:, 0, :].sum()
b_t_0 = bt_0 / (bt_0 + bt_1)
b_t_1 = bt_1 / (bt_0 + bt_1)
if b_s_0 > 0.5:
    bias_0 = abs(b_s_0 - b_t_0)
elif b_s_1 > 0.5:
    bias_0 = abs(b_s_1 - b_t_1)
else:
    bias_0 = abs(b_s_0 - b_t_0)
# print(b_s_0, b_s_1)
# print(b_t_0, b_t_1)

bs_0 = arr[1, :, 0].sum() / arr[1, :, :].sum()
bs_1 = arr[1, :, 1].sum() / arr[1, :, :].sum()
b_s_0 = bs_0 / (bs_0 + bs_1)
b_s_1 = bs_1 / (bs_0 + bs_1)
bt_0 = arr[:, 1, 0].sum() / arr[:, 1, :].sum()
bt_1 = arr[:, 1, 1].sum() / arr[:, 1, :].sum()
b_t_0 = bt_0 / (bt_0 + bt_1)
b_t_1 = bt_1 / (bt_0 + bt_1)
if b_s_0 > 0.5:
    bias_1 = abs(b_s_0 - b_t_0)
elif b_s_1 > 0.5:
    bias_1 = abs(b_s_1 - b_t_1)
else:
    bias_1 = abs(b_s_0 - b_t_0)

bias = (bias_0 + bias_1) / 2
print("BA:", bias_0, bias_1, bias)

