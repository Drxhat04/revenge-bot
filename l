### Why *my* algo fires ≈ 40 × a year while the scammer posts “a setup every day”

| layer                   | what the scammer (very likely) does                                                                              | what **my scanner** currently enforces                                                       |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **1.  Bias**            | “price swept Asia liquidity” is **optional** – they’ll still mark an OB if structure looks good.                 | Hard‑coded: trade **only** if price pierces the 18:00→09:00 box *and* passes the FVG‑gap test. |
| **2.  OB pick**         | Any fresh opposing candle on M15/H1 that precedes a sharp H1 impulse qualifies. There is *always* one somewhere. | Only the **first OB *after*** the pierce is allowed, width capped at 4 USD.                    |
| **3.  Filters**         | None when they want to keep the signal cadence high.                                                             | Asia‑pierce buffer, FVG, zero‑width guard, zone‑width cap.                                     |
| **4.  Cadence promise** | “Daily signal” is marketing –  they’ll post even a 50 cents zone if nothing better prints.                       | My code refuses < 0.6 USD zones and cancels the day.                                         |

So the difference is **not** data snooping – it’s simply that you encoded *quality gates* the scammer doesn’t honour.
If you want a mechanical system that fires (almost) every London session **without over‑fitting**, you can loosen the *logical* constraints once, then freeze them.

---

## Principled ways to raise the daily hit‑rate (no walk‑forward tuning)

| change                                                                            | once‑off constant                       | expected effect                            | risk                                                 |
| --------------------------------------------------------------------------------- | --------------------------------------- | ------------------------------------------ | ---------------------------------------------------- |
| ① **Drop the Asia‑pierce requirement**. Just pick the freshest valid OB at 07:00. | `require_pierce = False`                | 200‑250 trades/yr                          | More small losses; needs TP/SL ratio to be ≥ 1 : 1.3 |
| ② **Keep the pierce rule but dump the FVG filter.**                               | `require_fvg = False` (already allowed) | \~ 3 × signals                             | Win‑rate falls from \~95 % → 60‑70 %                 |
| ③ **Adaptive Asia window** – Tokyo only: 00:00→06:00 UTC                          | fixed forever                           | more true sweeps, less “box too wide” days | still a fixed rule; no over‑fit                      |
| ④ **Always publish the *closest* OB even if no sweep.**                           | deterministic pick                      | exactly one setup Mon‑Fri                  | PF probably ≤ 1, but matches scammer cadence         |

Pick **one** of those rules, codify it, *never* touch it again except for obvious bugs – that avoids data snooping.

---

## Minimal code switch‑board

Add **two booleans** to `config.yaml` and one line in `scanner_debug.py` / `zone_scanner.py`:

```yaml
require_pierce:   true     # set false to accept any OB
require_fvg:      true     # already there
tokyo_only:       false    # true → window = 00:00‑06:00 UTC
```

```python
# inside scan() loop, before the hard pierce/FVG test
if not CFG.get("require_pierce", True):
    # take latest qualifying OB of the day even if no pierce
    ob = last_OB(day15, len(day15)-1, side="BUY")
    # same zone‑width logic …
```

Everything else (risk‑mult, width cap, TP ratios) stays frozen, so the system’s future performance can be measured cleanly.

---


