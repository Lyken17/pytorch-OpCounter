import re
import datetime


def func(base_str="(0.0.30-1908282117)"):
    # pattern = re.compile(r'''\((.*)\)''')
    #
    # for l in pattern.findall(base_str):
    #     prev_update = l.split("-")[-1]
    prev_update = base_str.strip()[1:-1].split("-")[-1]
    curr_update = (datetime.datetime.now() - datetime.timedelta(weeks=1)).strftime(
        "%Y%m%d%H%M"
    )[2:]
    if curr_update > prev_update:
        exit(0)
    else:
        exit(-1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--msg")
    args = parser.parse_args()
    func(base_str=args.msg)
