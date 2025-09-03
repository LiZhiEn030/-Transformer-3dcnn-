import re
import sys


def main():
    currency_info = {
        'CNY': ('fen', 1.0),
        'HKD': ('cents', 100 / 123),
        'JPY': ('sen', 100 / 1825),
        'EUR': ('eurocents', 100 / 14),
        'GBP': ('pence', 100 / 12)
    }

    try:
        lines = [line.strip() for line in sys.stdin if line.strip()]
        if not lines:
            raise ValueError("无输入数据")

        n = int(lines[0])
        if not (0 < n < 100):
            raise ValueError(f"记录数需在1-99之间，输入为: {n}")
        if len(lines) - 1 < n:
            raise ValueError(f"需{n}条记录，实际收到{len(lines) - 1}条")

        total_fen = 0
        for i in range(n):
            record = lines[i + 1]
            for currency, (sub_unit, rate) in currency_info.items():
                match = re.fullmatch(rf'^(\d+{currency})?(\d+{sub_unit})?$', record)
                if match:
                    yuan_part, fen_part = match.groups()
                    amount = 0.0
                    if yuan_part:
                        amount += float(re.findall(r'\d+', yuan_part)[0])
                    if fen_part:
                        amount += float(re.findall(r'\d+', fen_part)[0]) / 100
                    total_fen += int(amount * rate * 100)
                    break
            else:
                print(f"第{i + 1}条记录格式错误: {record}")

        print(total_fen)

    except ValueError as e:
        print(f"输入错误: {e}")


if __name__ == "__main__":
    main()
