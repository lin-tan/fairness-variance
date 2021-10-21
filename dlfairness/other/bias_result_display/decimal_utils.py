import decimal
from decimal import Decimal, localcontext

GLOBAL_ROUND_DIGIT = 4

def round_significant_digit(num, digit=GLOBAL_ROUND_DIGIT):
    with localcontext() as ctx:
        ctx.prec = digit # decimal.ROUND_HALF_EVEN
        return ctx.create_decimal(num)

def round_significant_format(num, digit=GLOBAL_ROUND_DIGIT, scientific=False):
    with localcontext() as ctx:
        ctx.prec = digit
        t = ctx.create_decimal(num)
        if scientific:
            format_str = '#.' + str(digit - 1) + 'e'
        else:
            format_str = '#.' + str(digit) + 'g'
        
        return format(float(t), format_str)


def round_list(tlist, digit=GLOBAL_ROUND_DIGIT):
    if not isinstance(tlist, list):
        return tlist
    return [round_significant_digit(e, digit=digit) for e in tlist]
