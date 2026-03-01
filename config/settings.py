import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY", "")

# Claude model
CLAUDE_MODEL = "claude-opus-4-6"

# Default tickers for demo
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]

# Data defaults
DEFAULT_PERIOD = "1y"
DEFAULT_INTERVAL = "1d"

# Backtesting defaults
INITIAL_CAPITAL = 100_000.0
COMMISSION_RATE = 0.001  # 0.1% per trade

# Risk management
MAX_POSITION_SIZE = 0.20   # max 20% of portfolio per position
STOP_LOSS_PCT = 0.07       # 7% stop loss
TAKE_PROFIT_PCT = 0.20     # 20% take profit

# Monte Carlo
MC_SIMULATIONS = 1000
MC_DAYS = 252  # 1 trading year
