POSITION_LIMITS = {
    "KELP": 50,
    "RAINFOREST_RESIN": 50,
}

FAIR_MKT_VALUE = {
    "KELP": lambda x: ((
        sum(p * x.buy_orders[p] for p in x.buy_orders)
        - sum(p * x.sell_orders[p] for p in x.sell_orders)
    )
    / (sum(x.buy_orders.values()) - sum(x.sell_orders.values()))) if x.buy_orders and x.sell_orders else 0,
    "RAINFOREST_RESIN": lambda _: 10000,
}
