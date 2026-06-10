# AlphaMetrics

**Librería pura de análisis técnico y finanzas cuantitativas** — 36 módulos organizados en 7
subpaquetes, implementados sobre `numpy` / `pandas` / `scipy` con tipos de retorno inmutables
(frozen dataclasses), validación centralizada y corrección numérica validada contra TA-Lib donde
aplica.

> **Librería pura — sin estado global, sin I/O.** Ningún módulo lee `.env`, base de datos, red ni
> disco. Las funciones reciben `pd.Series` / `pd.DataFrame` y retornan nuevas estructuras
> (`pd.Series`, frozen dataclasses, `float`). El consumidor es responsable del I/O y de los umbrales.

Extraído del módulo `alphametrics` del proyecto NarrativeAlpha como repositorio independiente.

## Estructura

```
src/
├── core/        types, validation, precision, smoothing   # primitivas compartidas
├── trend/       sma, ema, rsi, macd, stochastic, ichimoku  # tendencia / momentum
├── volatility/  bollinger, atr, adx, vwap, obv, adl        # volatilidad y volumen
├── risk/        sharpe, sortino, drawdown, var, cvar        # métricas de riesgo
├── stats/       returns, volatility_models, correlation, regression, timeseries
├── quant/       black_scholes, portfolio, capm, monte_carlo  # finanzas cuantitativas
└── systems/     costs, position_sizing, stop_loss, order_types, walk_forward, backtest
tests/           # suite espejo (pytest), con fixtures en conftest.py
docs/AlphaMetrics.md   # referencia completa del módulo
```

`src/` es el *Sources Root*: cada subpaquete (`core`, `trend`, `volatility`, `risk`, `stats`,
`quant`, `systems`) es importable como paquete top-level.

## Instalación

Requiere **Python ≥ 3.12**.

```bash
python -m venv .venv
# Windows:  .venv\Scripts\Activate.ps1
# Unix:     source .venv/bin/activate
pip install -e ".[dev,stats]"
```

Dependencias core: `numpy`, `pandas`, `scipy`. El extra `stats` añade `statsmodels`, `arch` y
`pmdarima` (requeridos solo por `stats.timeseries`); el extra `sklearn` añade `scikit-learn`. Ambos
están protegidos con guards de `ImportError`, así que la librería funciona sin ellos.

Alternativamente, hay archivos de dependencias con versiones fijadas (lock) para instalaciones
reproducibles: `pip install -r requirements.txt` (runtime, incluye el stack `stats`) y
`pip install -r requirements-dev.txt` (añade el tooling de desarrollo).

## Uso

```python
import pandas as pd
from trend.sma import sma
from trend.rsi import rsi
from volatility.bollinger import bollinger_bands
from risk.sharpe import sharpe_ratio
from stats.returns import simple_returns
from core.validation import validate_ohlcv

close = pd.Series([...])           # serie de precios de cierre
sma_20 = sma(close, period=20)
rsi_14 = rsi(close, period=14)
bb = bollinger_bands(close, period=20)   # frozen dataclass: bb.upper / bb.mid / bb.lower

returns = simple_returns(close)
sharpe = sharpe_ratio(returns)
```

Indicadores multi-output (`macd`, `bollinger_bands`, `stochastic`, `ichimoku`, `adx`, `max_drawdown`)
retornan frozen dataclasses definidas en `core.types`.

## Tests

```bash
pytest
```

La suite cubre los 7 subpaquetes (`test_core`, `test_trend`, `test_volatility`, `test_risk`,
`test_stats`, `test_quant`, `test_systems`).

## Documentación

Referencia completa (fórmulas, edge cases, validación contra TA-Lib) en
[`docs/AlphaMetrics.md`](docs/AlphaMetrics.md).
