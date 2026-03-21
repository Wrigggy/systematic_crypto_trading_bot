"""Roostoo mock exchange executor — REST API order execution.

Uses Binance WebSocket for market data; Roostoo REST API only for order
execution and balance tracking.

Symbol mapping: internal BTC/USDT -> Roostoo BTC/USD at the boundary.
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal, InvalidOperation, ROUND_DOWN
import time
from datetime import datetime
from typing import Any, Dict, Optional

import aiohttp

from core.models import Order, OrderStatus, OrderType, Side
from data.roostoo_auth import RoostooAuth
from execution.executor import BaseExecutor

logger = logging.getLogger(__name__)

# Max retries for transient HTTP errors (429, 5xx)
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 1.0  # seconds


class RoostooExecutor(BaseExecutor):
    """Executes orders against the Roostoo mock exchange via REST API."""

    def __init__(self, config: dict):
        self._base_url: str = config.get("base_url", "https://mock-api.roostoo.com")
        api_key = config.get("api_key", "")
        api_secret = config.get("api_secret", "")
        self._auth = RoostooAuth(api_key, api_secret)
        self._session: Optional[aiohttp.ClientSession] = None
        # Pair info cache: symbol -> {min_qty, qty_precision, min_notional, price_precision, price_step}
        self._pair_info: Dict[str, Dict[str, Any]] = {}
        self._trade_logger = None  # set externally via set_trade_logger

    def set_trade_logger(self, trade_logger) -> None:
        """Inject the TradeLogger for structured event logging."""
        self._trade_logger = trade_logger

    async def start(self) -> None:
        """Initialize HTTP session and cache exchange info."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
        )
        # Validate server time
        ok = await self._auth.validate_server_time(self._base_url)
        if not ok:
            logger.warning("Server time drift is large; proceeding anyway")

        # Cache exchange info
        await self._load_exchange_info()
        logger.info("RoostooExecutor started, %d pairs loaded", len(self._pair_info))

    async def stop(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("RoostooExecutor stopped")

    # ── Symbol Mapping ──

    @staticmethod
    def to_roostoo_symbol(internal_symbol: str) -> str:
        """Convert internal symbol (BTC/USDT) to Roostoo (BTC/USD)."""
        return internal_symbol.replace("/USDT", "/USD")

    @staticmethod
    def to_internal_symbol(roostoo_symbol: str) -> str:
        """Convert Roostoo symbol (BTC/USD) to internal (BTC/USDT)."""
        return roostoo_symbol.replace("/USD", "/USDT")

    # ── Core Methods ──

    async def execute(self, order: Order) -> Order:
        """Place an order on Roostoo. POST /v3/place_order."""
        roostoo_pair = self.to_roostoo_symbol(order.symbol)
        timestamp = self._auth.get_timestamp()

        params = {
            "pair": roostoo_pair,
            "side": order.side.value,
            "type": order.order_type.value,
            "quantity": str(self._round_quantity(order.symbol, order.quantity)),
            "timestamp": str(timestamp),
        }
        if order.order_type == OrderType.LIMIT and order.price is not None:
            normalized_price = self._round_price(order.symbol, order.price)
            order.price = normalized_price
            params["price"] = self._format_number(normalized_price)

        start_time = time.monotonic()
        data = await self._signed_request("POST", "/v3/place_order", params)
        latency_ms = (time.monotonic() - start_time) * 1000

        if data and data.get("Success"):
            order_data = data.get("OrderDetail", {})
            order.order_id = str(order_data.get("OrderID", order.order_id))
            (
                order.status,
                order.filled_quantity,
                order.filled_price,
            ) = self._interpret_order_state(
                order_data,
                requested_quantity=order.quantity,
                fallback_price=order.price,
            )
            if order.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
                order.filled_at = datetime.utcnow()
                if order.status == OrderStatus.PARTIALLY_FILLED:
                    logger.info(
                        "Roostoo order partially filled: %s %s %.6f/%.6f @ %.4f (latency %.0fms)",
                        order.side.value,
                        order.symbol,
                        order.filled_quantity,
                        order.quantity,
                        order.filled_price or 0.0,
                        latency_ms,
                    )
                else:
                    logger.info(
                        "Roostoo order filled: %s %s %.6f @ %.4f (latency %.0fms)",
                        order.side.value,
                        order.symbol,
                        order.filled_quantity,
                        order.filled_price or 0.0,
                        latency_ms,
                    )
            else:
                logger.info(
                    "Roostoo order accepted: %s %s %.6f @ %s status=%s (latency %.0fms)",
                    order.side.value,
                    order.symbol,
                    order.quantity,
                    self._format_number(order.price) if order.price is not None else "MKT",
                    order.status.value,
                    latency_ms,
                )
        elif data and not data.get("Success"):
            order.status = OrderStatus.REJECTED
            err = data.get("ErrMsg", "Unknown error")
            logger.error("Roostoo order rejected: %s — %s", order.symbol, err)
        else:
            order.status = OrderStatus.REJECTED
            logger.error("Roostoo order failed: no response for %s", order.symbol)

        # Log to trade logger
        if self._trade_logger:
            await self._trade_logger.log_order(
                symbol=order.symbol,
                side=order.side.value,
                order_type=order.order_type.value,
                quantity=order.quantity,
                price=order.filled_price or order.price,
                order_id=order.order_id,
                status=order.status.value,
                roostoo_response=data,
                latency_ms=latency_ms,
            )

        return order

    async def cancel(self, order_id: str, symbol: str) -> Order:
        """Cancel an order. POST /v3/cancel_order."""
        timestamp = self._auth.get_timestamp()

        params = {
            "order_id": order_id,
            "timestamp": str(timestamp),
        }
        data = await self._signed_request("POST", "/v3/cancel_order", params)

        status = OrderStatus.CANCELLED
        if data and not data.get("Success"):
            logger.warning(
                "Cancel may have failed for %s: %s", order_id, data.get("ErrMsg")
            )

        return Order(
            order_id=order_id,
            symbol=symbol,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=0,
            status=status,
        )

    async def get_status(self, order_id: str, symbol: str) -> Order:
        """Query order status. POST /v3/query_order."""
        timestamp = self._auth.get_timestamp()

        params = {
            "order_id": order_id,
            "timestamp": str(timestamp),
        }
        data = await self._signed_request("POST", "/v3/query_order", params)

        if data and data.get("Success"):
            matches = data.get("OrderMatched", [])
            if matches:
                order_data = matches[0]
                quantity = self._coerce_float(order_data.get("Quantity"), 0.0)
                status, filled_quantity, filled_price = self._interpret_order_state(
                    order_data,
                    requested_quantity=quantity,
                    fallback_price=self._coerce_optional_float(order_data.get("Price")),
                )
                side_str = order_data.get("Side", "BUY").upper()
                return Order(
                    order_id=order_id,
                    symbol=symbol,
                    side=Side(side_str),
                    order_type=OrderType(order_data.get("Type", "MARKET")),
                    quantity=quantity,
                    filled_quantity=filled_quantity,
                    filled_price=filled_price,
                    status=status,
                )

        return Order(
            order_id=order_id,
            symbol=symbol,
            side=Side.BUY,
            order_type=OrderType.MARKET,
            quantity=0,
            status=OrderStatus.PENDING,
        )

    # ── Balance & Exchange Info ──

    async def get_balance(self) -> Dict[str, float]:
        """Get account balances. GET /v3/balance."""
        timestamp = self._auth.get_timestamp()
        params = {"timestamp": str(timestamp)}
        data = await self._signed_request("GET", "/v3/balance", params)
        logger.debug("Roostoo /v3/balance raw response: %s", data)

        balances: Dict[str, float] = {}
        if data and data.get("Success"):
            # API may return "SpotWallet" or "Wallet" depending on version
            wallet = data.get("SpotWallet") or data.get("Wallet") or {}
            for asset, amounts in wallet.items():
                free = float(amounts.get("Free", 0))
                if free > 0 or asset == "USD":
                    balances[asset] = free
            # Preserve a successful zero-cash snapshot so the runtime does not
            # fall back to paper capital when the account simply has no free USD.
            balances.setdefault("USD", 0.0)
        elif data:
            logger.warning("Roostoo balance failed: %s", data.get("ErrMsg", data))
        return balances

    async def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange info (pair precisions, min orders). GET /v3/exchangeInfo."""
        data = await self._unsigned_request("GET", "/v3/exchangeInfo")
        return data or {}

    async def get_ticker(self, symbol: str) -> Optional[float]:
        """Get latest ticker price for a Roostoo pair. GET /v3/ticker."""
        roostoo_pair = self.to_roostoo_symbol(symbol)
        timestamp = self._auth.get_timestamp()
        params = {"pair": roostoo_pair, "timestamp": str(timestamp)}
        data = await self._signed_request("GET", "/v3/ticker", params)
        if data and data.get("Success"):
            ticker_data = data.get("Data", {})
            pair_data = ticker_data.get(roostoo_pair, {})
            price = pair_data.get("LastPrice", 0)
            if price:
                return float(price)
        return None

    # ── Internal Helpers ──

    async def _load_exchange_info(self) -> None:
        """Cache pair precision and min order info from exchange."""
        data = await self.get_exchange_info()
        if not data:
            logger.warning("Failed to load exchange info")
            return
        trade_pairs = data.get("TradePairs", {})
        if not trade_pairs:
            logger.warning("No TradePairs in exchange info response")
            return
        for symbol, info in trade_pairs.items():
            internal = self.to_internal_symbol(symbol)
            qty_precision = int(info.get("AmountPrecision", 8))
            price_precision = self._extract_optional_int(
                info,
                "PricePrecision",
                "PriceScale",
                "QuotePrecision",
                "PriceDecimal",
            )
            price_step = self._extract_optional_float(
                info,
                "PriceStep",
                "PriceStepSize",
                "TickSize",
                "PriceTick",
                "PriceTickSize",
            )
            self._pair_info[internal] = {
                "min_qty": 10 ** (-qty_precision),
                "qty_precision": qty_precision,
                "min_notional": float(info.get("MiniOrder", 0)),
                "price_precision": price_precision,
                "price_step": price_step,
            }

    def _round_quantity(self, symbol: str, quantity: float) -> float:
        """Round quantity to the pair's precision."""
        info = self._pair_info.get(symbol, {})
        precision = info.get("qty_precision", 8)
        return round(quantity, precision)

    def _interpret_order_state(
        self,
        order_data: Dict[str, Any],
        *,
        requested_quantity: float,
        fallback_price: Optional[float],
    ) -> tuple[OrderStatus, float, Optional[float]]:
        """Map Roostoo payloads to internal status and fill semantics.

        The exchange can acknowledge a resting order with `Status=PENDING` and
        may still echo a non-zero `FilledQuantity` from query responses. Status
        takes priority so open orders remain active instead of being treated as
        filled.
        """
        status = self._map_exchange_status(order_data.get("Status"))
        filled_quantity = self._coerce_float(order_data.get("FilledQuantity"), 0.0)
        avg_price = self._coerce_optional_float(order_data.get("FilledAverPrice"))
        resting_price = self._coerce_optional_float(order_data.get("Price"))
        fill_price = avg_price or resting_price or fallback_price

        if status == OrderStatus.FILLED:
            final_qty = filled_quantity if filled_quantity > 0 else requested_quantity
            return status, final_qty, fill_price

        if status == OrderStatus.PARTIALLY_FILLED:
            if filled_quantity <= 0:
                return OrderStatus.SUBMITTED, 0.0, None
            return status, min(filled_quantity, requested_quantity), fill_price

        if status == OrderStatus.CANCELLED:
            final_qty = min(max(filled_quantity, 0.0), requested_quantity)
            return status, final_qty, fill_price if final_qty > 0 else None

        if status == OrderStatus.SUBMITTED:
            return status, 0.0, None

        if requested_quantity > 0 and filled_quantity >= requested_quantity - 1e-10:
            return OrderStatus.FILLED, requested_quantity, fill_price
        if filled_quantity > 0:
            return (
                OrderStatus.PARTIALLY_FILLED,
                min(filled_quantity, requested_quantity),
                fill_price,
            )
        return OrderStatus.SUBMITTED, 0.0, None

    def _round_price(self, symbol: str, price: float) -> float:
        """Round limit price to the pair's published price step or precision."""
        info = self._pair_info.get(symbol, {})
        price_step = info.get("price_step")
        if price_step is not None and price_step > 0:
            return self._round_to_step(price, price_step)

        precision = info.get("price_precision")
        if precision is not None and precision >= 0:
            return round(price, precision)

        return round(price, 8)

    @staticmethod
    def _round_to_step(value: float, step: float) -> float:
        """Round positive values down to a valid exchange step size."""
        try:
            value_dec = Decimal(str(value))
            step_dec = Decimal(str(step))
        except (InvalidOperation, ValueError):
            return value
        if step_dec <= 0:
            return value
        scaled = (value_dec / step_dec).to_integral_value(rounding=ROUND_DOWN)
        return float(scaled * step_dec)

    @staticmethod
    def _format_number(value: float) -> str:
        """Format numbers for exchange params without scientific notation."""
        return format(value, "f").rstrip("0").rstrip(".") or "0"

    @staticmethod
    def _extract_optional_int(info: Dict[str, Any], *keys: str) -> Optional[int]:
        for key in keys:
            raw = info.get(key)
            if raw in (None, ""):
                continue
            try:
                return int(raw)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _extract_optional_float(info: Dict[str, Any], *keys: str) -> Optional[float]:
        for key in keys:
            raw = info.get(key)
            if raw in (None, "", 0, "0"):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                return value
        return None

    @staticmethod
    def _map_exchange_status(raw_status: Any) -> Optional[OrderStatus]:
        if raw_status is None:
            return None
        status_map = {
            "FILLED": OrderStatus.FILLED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "CANCELLED": OrderStatus.CANCELLED,
            "CANCELED": OrderStatus.CANCELLED,
            "NEW": OrderStatus.SUBMITTED,
            "PENDING": OrderStatus.SUBMITTED,
        }
        return status_map.get(str(raw_status).upper())

    @staticmethod
    def _coerce_float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _coerce_optional_float(cls, value: Any) -> Optional[float]:
        parsed = cls._coerce_float(value, 0.0)
        return parsed or None

    async def _signed_request(
        self, method: str, endpoint: str, params: dict
    ) -> Optional[Dict]:
        """Make a signed request with retry on transient errors."""
        headers, query_string = self._auth.sign(params)
        url = f"{self._base_url}{endpoint}"

        for attempt in range(MAX_RETRIES):
            try:
                if method == "GET":
                    resp = await self._session.get(
                        f"{url}?{query_string}", headers=headers
                    )
                else:
                    resp = await self._session.post(
                        url,
                        data=query_string,
                        headers={
                            **headers,
                            "Content-Type": "application/x-www-form-urlencoded",
                        },
                    )

                if resp.status in (429, 500, 502, 503, 504):
                    wait = RETRY_BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        "Roostoo %s %s returned %d, retrying in %.1fs...",
                        method,
                        endpoint,
                        resp.status,
                        wait,
                    )
                    await asyncio.sleep(wait)
                    continue

                data = await resp.json(content_type=None)

                if self._trade_logger:
                    await self._trade_logger.log_api(
                        endpoint=endpoint,
                        params=params,
                        response_code=resp.status,
                        success=resp.status == 200,
                    )
                return data

            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    wait = RETRY_BACKOFF_BASE * (2**attempt)
                    logger.warning(
                        "Roostoo request error: %s, retrying in %.1fs", e, wait
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        "Roostoo request failed after %d retries: %s", MAX_RETRIES, e
                    )
                    if self._trade_logger:
                        await self._trade_logger.log_api(
                            endpoint=endpoint,
                            params=params,
                            success=False,
                            error_msg=str(e),
                        )
        return None

    async def _unsigned_request(
        self, method: str, endpoint: str, params: Optional[dict] = None
    ) -> Optional[Dict]:
        """Make an unsigned request (public endpoints)."""
        url = f"{self._base_url}{endpoint}"
        try:
            if method == "GET":
                resp = await self._session.get(url, params=params)
            else:
                resp = await self._session.post(url, data=params)
            return await resp.json(content_type=None)
        except Exception as e:
            logger.error("Roostoo unsigned request failed: %s", e)
            return None
