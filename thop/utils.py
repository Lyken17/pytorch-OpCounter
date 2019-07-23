from collections import Iterable


def clever_format(nums, format="%.2f"):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        if num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        if num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        if num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    if len(clever_nums) == 0:
        clever_nums = clever_nums[0]

    return clever_nums
