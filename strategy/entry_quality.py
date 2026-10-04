"""Causal completed-minute trade-flow confirmation; no L2 or inferred trade side.

Uses the venue's aggregate taker-buy quote volume, not candle-sign volume.
Missing, stale and zero-volume context fail closed for entries only.
"""
from collections import deque
import math


class EntryQualityHistory:
    def __init__(self, config):
        self.minimum_share = float(config.get('minimum_recent_buy_share', .5))
        self.minimum_improvement = float(config.get('minimum_share_improvement', .05))
        if not (0 <= self.minimum_share <= 1 and 0 <= self.minimum_improvement <= 1):
            raise ValueError('Invalid entry-flow thresholds')
        self.rows, self.snapshots, self.resets = {}, {}, {}

    def observe(self, symbol, closed_at, close, quote, buy_quote, now):
        if (not isinstance(closed_at, int) or closed_at % 60 or closed_at > now
                or not all(math.isfinite(v) for v in (close, quote, buy_quote))
                or close <= 0 or not 0 <= buy_quote <= quote):
            raise ValueError('Invalid completed-minute flow observation')
        row = (closed_at, close, quote, buy_quote)
        rows = self.rows.setdefault(symbol, deque(maxlen=20))
        if rows and closed_at <= rows[-1][0]:
            if row == rows[-1]:
                return
            raise ValueError('Conflicting or out-of-order flow observation')
        if rows and closed_at - rows[-1][0] != 60:
            rows.clear()
            self.resets[symbol] = self.resets.get(symbol, 0) + 1
        rows.append(row)
        self.snapshots.pop(symbol, None)
        if len(rows) < 20:
            return
        previous, recent = list(rows)[:15], list(rows)[15:]
        old_quote = sum(r[2] for r in previous)
        new_quote = sum(r[2] for r in recent)
        if min(old_quote, new_quote) <= 0:
            return
        old_share = sum(r[3] for r in previous) / old_quote
        new_share = sum(r[3] for r in recent) / new_quote
        self.snapshots[symbol] = dict(closed_at=closed_at, previous_buy_share=old_share,
            recent_buy_share=new_share, share_improvement=new_share - old_share,
            recent_return_bps=(rows[-1][1] / rows[-6][1] - 1) * 10000)

    def check(self, symbol, now):
        snapshot = self.snapshots.get(symbol)
        if snapshot is None or not 0 <= now - snapshot['closed_at'] < 60:
            return {'reason': 'flow_unready_or_stale'}
        reason = 'qualified'
        if snapshot['recent_buy_share'] < self.minimum_share:
            reason = 'sell_pressure_persists'
        elif snapshot['share_improvement'] < self.minimum_improvement:
            reason = 'flow_not_improving'
        elif snapshot['recent_return_bps'] <= 0:
            reason = 'price_not_recovering'
        return dict(snapshot, reason=reason)
