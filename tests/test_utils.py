from thop import utils
import pytest


class TestUtils:
    def test_clever_format_returns_formatted_number(self):
        nums = 1
        format = "%.2f"
        clever_nums = utils.clever_format(nums, format)
        assert clever_nums == "1.00B"

    def test_clever_format_returns_formatted_numbers(self):
        nums = [1, 2]
        format = "%.2f"
        clever_nums = utils.clever_format(nums, format)
        assert clever_nums == ("1.00B", "2.00B")
