from collections.abc import Iterable

COLOR_RED = "91m"
COLOR_GREEN = "92m"
COLOR_YELLOW = "93m"

def colorful_print(fn_print, color=COLOR_RED):
    def actual_call(*args, **kwargs):
        print(f"\033[{color}", end="")
        fn_print(*args, **kwargs)
        print("\033[00m", end="")
    return actual_call

prRed = colorful_print(print, color=COLOR_RED)
prGreen = colorful_print(print, color=COLOR_GREEN)
prYellow = colorful_print(print, color=COLOR_YELLOW)

# def prRed(skk):
#     print("\033[91m{}\033[00m".format(skk))

# def prGreen(skk):
#     print("\033[92m{}\033[00m".format(skk))

# def prYellow(skk):
#     print("\033[93m{}\033[00m".format(skk))


def clever_format(nums, format="%.2f"):
    if not isinstance(nums, Iterable):
        nums = [nums]
    clever_nums = []

    for num in nums:
        if num > 1e12:
            clever_nums.append(format % (num / 1e12) + "T")
        elif num > 1e9:
            clever_nums.append(format % (num / 1e9) + "G")
        elif num > 1e6:
            clever_nums.append(format % (num / 1e6) + "M")
        elif num > 1e3:
            clever_nums.append(format % (num / 1e3) + "K")
        else:
            clever_nums.append(format % num + "B")

    clever_nums = clever_nums[0] if len(clever_nums) == 1 else (*clever_nums,)

    return clever_nums


if __name__ == "__main__":
    prRed("hello", "world")
    prGreen("hello", "world")
    prYellow("hello", "world")