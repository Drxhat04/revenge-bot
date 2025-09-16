# mt5_sanity_buy.py
import time
import MetaTrader5 as mt5

LOGIN    = 81496583
PASSWORD = "Password2$"
SERVER   = "Exness-MT5Trial10"
SYMBOL   = "XAUUSDz"

assert mt5.initialize(), mt5.last_error()
assert mt5.login(login=LOGIN, password=PASSWORD, server=SERVER), mt5.last_error()
assert mt5.symbol_select(SYMBOL, True)

for _ in range(5):
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick: break
    time.sleep(0.2)

vol = 0.02
price = mt5.symbol_info_tick(SYMBOL).ask
req = dict(
    action=mt5.TRADE_ACTION_DEAL,
    symbol=SYMBOL,
    volume=vol,
    type=mt5.ORDER_TYPE_BUY,
    price=price,
    deviation=300,
    magic=234000,
    comment="sanity",
    type_time=mt5.ORDER_TIME_GTC,
    type_filling=mt5.ORDER_FILLING_IOC,
)

chk = mt5.order_check(req)
print("check:", chk.retcode, getattr(chk, "comment", ""))

res = mt5.order_send(req)
print("send:", None if res is None else res.retcode, getattr(res, "comment", ""))
print("last_error:", mt5.last_error())

mt5.shutdown()
