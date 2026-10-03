"""Roostoo spot execution with bounded passive quotes and explicit reconciliation.

No placement retries: Roostoo does not document an idempotent client order ID.
A timeout after POST is an unknown order, not permission to submit another one.
"""
from __future__ import annotations

import asyncio
import logging
import math
import time
from datetime import datetime
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Any, Dict, Optional

import aiohttp

from core.models import Order, OrderStatus, OrderType, Side
from execution.executor import BaseExecutor
from execution.request_budget import RequestBudget, RequestBudgetExceeded
from plugins.roostoo.auth import RoostooAuth

logger = logging.getLogger(__name__)


class RoostooExecutor(BaseExecutor):
    def __init__(self, config: dict):
        self._base_url = config.get("base_url", "https://mock-api.roostoo.com")
        self._auth = RoostooAuth(config.get("api_key", ""), config.get("api_secret", ""))
        self._session: Optional[aiohttp.ClientSession] = None
        self._pair_info: Dict[str, Dict[str, Any]] = {}
        self._trade_logger = None
        self._orders: dict[str, Order] = {}
        self._cancel_requested: set[str] = set()
        self._budget = RequestBudget(config.get("max_requests_per_minute", 5),
                                     config.get("emergency_request_reserve", 1))
        self._offset_bps = config.get("maker_offset_bps", 0)
        self._starting = False

    def set_trade_logger(self, trade_logger):
        self._trade_logger = trade_logger

    async def start(self):
        self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15))
        self._starting = True
        try:
            clock = await self._unsigned_request("GET", "/v3/serverTime")
            if not clock or abs(int(clock.get("ServerTime", 0)) - self._auth.get_timestamp()) > 60000:
                raise RuntimeError("Cannot verify Roostoo server time")
            await self._load_exchange_info()
            pending = await self._signed_request("GET", "/v3/pending_count", {})
            if not pending or not (pending.get("TotalPending") == 0 or
                    pending.get("ErrMsg") == "no pending order under this account"):
                raise RuntimeError("Existing or unknown exchange orders: reconcile before starting")
        except Exception:
            await self.stop()
            raise
        finally:
            self._starting = False

    async def stop(self):
        if self._session and not self._session.closed:
            await self._session.close()

    @staticmethod
    def to_roostoo_symbol(symbol):
        return symbol.replace("/USDT", "/USD")

    @staticmethod
    def to_internal_symbol(symbol):
        return symbol.replace("/USD", "/USDT")

    async def execute(self, order: Order) -> Order:
        try:
            info = self._pair_info.get(order.symbol)
            if not info or not info.get("can_trade", True):
                raise ValueError("Missing or disabled venue instrument")
            order.quantity = self._round_quantity(order.symbol, order.quantity)
            if order.quantity <= 0 or not math.isfinite(order.quantity):
                raise ValueError("Invalid rounded quantity")
            if order.maker_preferred and not order.urgent:
                quote = await self.get_quote(order.symbol)
                bid, ask = quote["bid"], quote["ask"]
                offset = self._offset_bps / 10000
                target = bid * (1 - offset) if order.side == Side.BUY else ask * (1 + offset)
                # Preserve explicit strategy target limits when more passive.
                if order.price is not None:
                    target = min(target, order.price) if order.side == Side.BUY else max(target, order.price)
                rounding = ROUND_DOWN if order.side == Side.BUY else ROUND_UP
                step = Decimal(1).scaleb(-info["price_precision"])
                order.price = float(Decimal(str(target)).quantize(step, rounding=rounding))
                if not (0 < order.price < ask if order.side == Side.BUY else order.price > bid):
                    raise ValueError("Cannot form a passive quote at venue tick size")
                order.order_type = OrderType.LIMIT
            if order.order_type == OrderType.LIMIT:
                if not order.price or not math.isfinite(order.price) or order.price <= 0:
                    raise ValueError("Invalid limit price")
                if order.quantity * order.price <= info["min_notional"]:
                    raise ValueError("Order below venue minimum notional")
        except (ValueError, RequestBudgetExceeded, RuntimeError) as exc:
            order.status = OrderStatus.REJECTED
            order.reason = str(exc)
            return order

        params = {"pair": self.to_roostoo_symbol(order.symbol), "side": order.side.value,
                  "type": order.order_type.value, "quantity": str(order.quantity)}
        if order.order_type == OrderType.LIMIT:
            params["price"] = str(order.price)
        start = time.monotonic()
        try:
            data = await self._signed_request("POST", "/v3/place_order", params, urgent=order.urgent)
        except RequestBudgetExceeded:
            order.status = OrderStatus.REJECTED  # Definitely not sent.
            order.reason = "request_budget_deferred"
            return order
        if data and data.get("Success") is True:
            detail = data.get("OrderDetail", {})
            if detail.get("OrderID") is not None:
                order.order_id = str(detail["OrderID"])
                order.exchange_acknowledged = True
            try:
                order = self._parse(detail, order)
            except (ValueError, TypeError):
                order.status = OrderStatus.UNKNOWN
                logger.exception("Invalid placement receipt; reconciliation required")
        elif data and data.get("Success") is False:
            order.status = OrderStatus.REJECTED
            order.reason = str(data.get("ErrMsg", "rejected"))
        else:
            order.status = OrderStatus.UNKNOWN
        self._orders[order.order_id] = order.model_copy(deep=True)
        if self._trade_logger:
            await self._trade_logger.log_order(
                symbol=order.symbol, side=order.side.value, order_type=order.order_type.value,
                quantity=order.quantity, price=order.price, order_id=order.order_id,
                status=order.status.value, roostoo_response=data,
                latency_ms=(time.monotonic() - start) * 1000)
        return order

    def _parse(self, detail: dict, original: Order) -> Order:
        if str(detail.get("OrderID")) != original.order_id or not original.exchange_acknowledged:
            raise ValueError("Missing or mismatched exchange order identity")
        status = str(detail.get("Status", "UNKNOWN")).upper()
        status_map = {"NEW": OrderStatus.SUBMITTED, "PENDING": OrderStatus.SUBMITTED,
                      "FILLED": OrderStatus.FILLED, "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
                      "CANCELED": OrderStatus.CANCELLED, "CANCELLED": OrderStatus.CANCELLED,
                      "REJECTED": OrderStatus.REJECTED}
        qty = float(detail.get("FilledQuantity", 0))
        price = float(detail.get("FilledAverPrice", 0) or 0)
        if not math.isfinite(qty) or qty < 0 or qty > original.quantity + 1e-8:
            raise ValueError("Invalid executed quantity")
        if qty and (not math.isfinite(price) or price <= 0):
            raise ValueError("Positive filled quantity without execution price")
        mapped = status_map.get(status, OrderStatus.UNKNOWN)
        if mapped == OrderStatus.FILLED and abs(qty - original.quantity) > 1e-8:
            raise ValueError("FILLED receipt missing full quantity")
        if 0 < qty < original.quantity and mapped == OrderStatus.SUBMITTED:
            mapped = OrderStatus.PARTIALLY_FILLED
        commission = detail.get("CommissionChargeValue")
        if qty and (commission is None or not detail.get("CommissionCoin")):
            raise ValueError("Executed order missing actual commission details")
        commission = float(commission) if commission is not None else None
        if commission is not None and (not math.isfinite(commission) or commission < 0):
            raise ValueError("Invalid commission")
        role = detail.get("Role")
        finish = float(detail.get("FinishTimestamp", 0) or 0)
        if not math.isfinite(finish) or finish < 0 or finish > self._auth.get_timestamp() + 5000:
            raise ValueError("Invalid exchange completion timestamp")
        # FinishTimestamp is completion, not proof of the first partial fill.
        # A partial/cancel receipt without execution time uses observation time.
        filled_at = datetime.utcfromtimestamp(finish / 1000) if qty and finish and mapped == OrderStatus.FILLED else None
        return original.model_copy(update={
            "status": mapped, "filled_quantity": qty, "filled_price": price or None,
            "filled_at": filled_at, "fill_time_source": "venue_finish" if filled_at else "observation",
            "liquidity": role,
            "commission": commission, "commission_asset": detail.get("CommissionCoin"),
            "fee_bps": 5.0 if role == "MAKER" else 10.0,
        })

    async def get_status(self, order_id: str, symbol: str) -> Order:
        original = self._orders.get(order_id)
        if original is None or not original.exchange_acknowledged:
            raise RuntimeError("Unknown placement ID requires explicit reconciliation")
        data = await self._signed_request("POST", "/v3/query_order", {"order_id": order_id},
                                          urgent=original.urgent or order_id in self._cancel_requested)
        if not data or not data.get("Success"):
            raise RuntimeError("Order query did not confirm status")
        detail = next((d for d in data.get("OrderMatched", []) if str(d.get("OrderID")) == order_id), None)
        if detail is None:
            raise RuntimeError("Order missing from query response")
        updated = self._parse(detail, original)
        self._orders[order_id] = updated
        return updated.model_copy(deep=True)

    async def cancel(self, order_id: str, symbol: str) -> Order:
        original = self._orders.get(order_id)
        if original is None or not original.exchange_acknowledged:
            raise RuntimeError("Cannot cancel an unacknowledged placement")
        if order_id not in self._cancel_requested:
            data = await self._signed_request("POST", "/v3/cancel_order", {"order_id": order_id}, urgent=True)
            # Even a failed/ambiguous cancel must query for a racing fill first.
            self._cancel_requested.add(order_id)
            logger.info("Cancel response %s: %s", order_id, data)
        updated = await self.get_status(order_id, symbol)
        if updated.status not in {OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED}:
            self._cancel_requested.discard(order_id)
        return updated

    async def get_balance(self) -> Dict[str, float]:
        data = await self._signed_request("GET", "/v3/balance", {})
        if not data or not data.get("Success"):
            raise RuntimeError("Cannot reconcile account balances")
        wallet = data.get("SpotWallet") or data.get("Wallet") or {}
        if any(float(amounts.get("Lock", 0)) > 0 for amounts in wallet.values()):
            raise RuntimeError("Locked funds require order reconciliation")
        return {asset: float(amounts.get("Free", 0)) for asset, amounts in wallet.items()}

    async def get_exchange_info(self):
        return await self._unsigned_request("GET", "/v3/exchangeInfo")

    async def get_quote(self, symbol):
        pair = self.to_roostoo_symbol(symbol)
        data = await self._signed_request("GET", "/v3/ticker", {"pair": pair})
        if not data or not data.get("Success"):
            raise RuntimeError("Missing venue quote")
        quote = data.get("Data", {}).get(pair, {})
        bid, ask = float(quote.get("MaxBid", 0)), float(quote.get("MinAsk", 0))
        if not all(math.isfinite(v) and v > 0 for v in (bid, ask)) or bid > ask:
            raise ValueError("Invalid venue quote")
        server_time = float(data.get("ServerTime", 0))
        if abs(self._auth.get_timestamp() - server_time) > 5000:
            raise ValueError("Stale venue quote")
        return {"bid": bid, "ask": ask, "last": float(quote.get("LastPrice", (bid + ask) / 2))}

    async def get_ticker(self, symbol):
        return (await self.get_quote(symbol))["last"]

    async def _load_exchange_info(self):
        data = await self.get_exchange_info()
        if not data or not data.get("TradePairs"):
            raise RuntimeError("Cannot load venue instrument rules")
        for symbol, info in data["TradePairs"].items():
            self._pair_info[self.to_internal_symbol(symbol)] = {
                "qty_precision": int(info["AmountPrecision"]),
                "price_precision": int(info["PricePrecision"]),
                "min_notional": float(info["MiniOrder"]), "can_trade": info.get("CanTrade", True)}

    def _round_quantity(self, symbol, quantity):
        step = Decimal(1).scaleb(-self._pair_info[symbol]["qty_precision"])
        return float(Decimal(str(quantity)).quantize(step, rounding=ROUND_DOWN))

    async def _signed_request(self, method, endpoint, params, *, urgent=False):
        return await self._request(method, endpoint, params, signed=True, urgent=urgent)

    async def _unsigned_request(self, method, endpoint, params=None):
        return await self._request(method, endpoint, params or {}, signed=False)

    async def _request(self, method, endpoint, params, *, signed, urgent=False):
        # No runtime sleeps or hidden retries. Every HTTP attempt consumes the same budget.
        while True:
            try:
                self._budget.take(urgent=urgent)
                break
            except RequestBudgetExceeded:
                if not self._starting:
                    raise
                await asyncio.sleep(1)
        url = self._base_url + endpoint
        payload = dict(params)
        headers = {}
        if signed:
            payload["timestamp"] = str(self._auth.get_timestamp())
            headers, encoded = self._auth.sign(payload)
        try:
            if method == "GET":
                response = await self._session.get(
                    url + ("?" + encoded if signed else ""),
                    headers=headers, **({} if signed else {"params": payload}))
            else:
                response = await self._session.post(url, data=encoded if signed else payload,
                    headers={**headers, "Content-Type": "application/x-www-form-urlencoded"})
            try:
                data = await response.json(content_type=None)
                if self._trade_logger:
                    await self._trade_logger.log_api(endpoint=endpoint, params=payload,
                        response_code=response.status, success=response.status == 200)
                return data if response.status == 200 else None
            finally:
                response.release()
        except Exception:
            logger.exception("Roostoo request outcome unavailable: %s", endpoint)
            return None
