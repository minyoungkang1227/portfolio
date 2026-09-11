"""
09_auto_trading_architecture.py

검증된 신호(z-score 임계값 진입)를 증권사 API 연동 자동매매로 구현하기 위한
아키텍처. 신호 생성과 리스크 관리 규칙을 분리해서, 신호가 나와도 4단계 리스크
규칙 중 하나라도 걸리면 진입/청산을 강제하도록 설계했습니다.

4단계 리스크 관리 규칙:
  1. 손절매(-10%)       : 개별 포지션이 -10% 도달 시 즉시 청산
  2. 시간제한 청산        : 보유기간이 max_holding_days를 넘으면 신호와 무관하게 청산
  3. 서킷브레이커         : 계좌 변동성(최근 N일 손익 표준편차)이 임계치를 넘으면 신규 진입 중단
  4. 일일 손실한도(-25%)  : 당일 누적 손실이 -25%에 도달하면 그날은 신규 진입 전면 중단

실제 증권사 API 클라이언트는 브로커마다 다르므로, 여기서는 인터페이스만
정의하고 place_order/get_positions는 스텁(stub)으로 남겨둡니다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum


class Side(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class Position:
    symbol: str
    side: Side
    entry_price: float
    entry_date: date
    quantity: int


@dataclass
class RiskLimits:
    stop_loss_pct: float = -0.10        # 손절매 -10%
    max_holding_days: int = 10          # 시간제한 청산
    circuit_breaker_vol: float = 0.05   # 최근 N일 손익 표준편차 임계치 (서킷브레이커)
    daily_loss_limit_pct: float = -0.25  # 일일 손실한도 -25%


class BrokerAPI:
    """실제 증권사 API 어댑터 인터페이스 (구현체는 브로커별로 별도 작성)"""

    def get_current_price(self, symbol: str) -> float:
        raise NotImplementedError

    def place_order(self, symbol: str, side: Side, quantity: int) -> str:
        raise NotImplementedError

    def close_position(self, position: Position) -> str:
        raise NotImplementedError


class RiskManagedTrader:
    def __init__(self, broker: BrokerAPI, limits: RiskLimits | None = None):
        self.broker = broker
        self.limits = limits or RiskLimits()
        self.positions: list[Position] = []
        self.daily_pnl_history: list[float] = []
        self._trading_halted_today = False

    # ── 4단계 리스크 규칙 ────────────────────────────────────────────────
    def _check_stop_loss(self, position: Position, current_price: float) -> bool:
        ret = (current_price - position.entry_price) / position.entry_price
        if position.side == Side.SHORT:
            ret = -ret
        return ret <= self.limits.stop_loss_pct

    def _check_time_limit(self, position: Position, today: date) -> bool:
        return (today - position.entry_date).days >= self.limits.max_holding_days

    def _check_circuit_breaker(self) -> bool:
        if len(self.daily_pnl_history) < 5:
            return False
        recent = self.daily_pnl_history[-5:]
        std = (sum((x - sum(recent) / len(recent)) ** 2 for x in recent) / len(recent)) ** 0.5
        return std >= self.limits.circuit_breaker_vol

    def _check_daily_loss_limit(self) -> bool:
        if not self.daily_pnl_history:
            return False
        return self.daily_pnl_history[-1] <= self.limits.daily_loss_limit_pct

    def can_open_new_position(self) -> bool:
        if self._trading_halted_today:
            return False
        if self._check_circuit_breaker():
            return False
        if self._check_daily_loss_limit():
            return False
        return True

    # ── 신호 처리 ────────────────────────────────────────────────────────
    def evaluate_signal(self, symbol: str, z_score: float, threshold: float, today: date) -> None:
        """
        05/06에서 검증한 z-score 임계값 로직을 그대로 사용.
        진입 전 반드시 can_open_new_position()으로 리스크 규칙을 먼저 확인.
        """
        if not self.can_open_new_position():
            return

        if z_score <= -threshold:
            self._enter(symbol, Side.LONG, today)
        elif z_score >= threshold:
            self._enter(symbol, Side.SHORT, today)

    def _enter(self, symbol: str, side: Side, today: date) -> None:
        price = self.broker.get_current_price(symbol)
        quantity = 1  # 실제로는 포지션 사이징 로직 적용
        self.broker.place_order(symbol, side, quantity)
        self.positions.append(Position(symbol, side, price, today, quantity))

    def manage_open_positions(self, today: date) -> None:
        """매 틱/매일 호출: 손절매·시간제한 규칙으로 기존 포지션을 청산."""
        remaining = []
        for pos in self.positions:
            current_price = self.broker.get_current_price(pos.symbol)
            if self._check_stop_loss(pos, current_price) or self._check_time_limit(pos, today):
                self.broker.close_position(pos)
                continue
            remaining.append(pos)
        self.positions = remaining

    def record_daily_pnl(self, pnl_pct: float) -> None:
        self.daily_pnl_history.append(pnl_pct)
        self._trading_halted_today = pnl_pct <= self.limits.daily_loss_limit_pct

    def reset_day(self) -> None:
        self._trading_halted_today = False


if __name__ == "__main__":
    class DummyBroker(BrokerAPI):
        def get_current_price(self, symbol: str) -> float:
            return 100.0

        def place_order(self, symbol: str, side: Side, quantity: int) -> str:
            print(f"[ORDER] {side.value} {quantity} {symbol}")
            return "order-id-stub"

        def close_position(self, position: Position) -> str:
            print(f"[CLOSE] {position.side.value} {position.symbol}")
            return "close-id-stub"

    trader = RiskManagedTrader(DummyBroker())
    today = date.today()
    trader.evaluate_signal("KRX_001", z_score=-1.9, threshold=1.6, today=today)
    trader.manage_open_positions(today + timedelta(days=1))
