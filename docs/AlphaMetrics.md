# AlphaMetrics — Documentacion del modulo

**Libreria de analisis tecnico y finanzas cuantitativas** con 36 modulos organizados en 7 subpaquetes. Implementaciones puras en numpy/pandas con tipos de retorno inmutables (frozen dataclasses), validacion centralizada y correccion numerica validada contra TA-Lib donde aplica.

**Librería pura — sin estado global, sin I/O, sin dependencias de `shared.config`.** Ningún archivo de `alphametrics/` lee de `.env`, base de datos, red ni disco. Las funciones reciben `pd.Series`/`pd.DataFrame` y retornan nuevas estructuras (`pd.Series`, frozen dataclasses, `float`). Los consumidores (`market_engine`, `scoring`, `outcome_tracker`) son responsables del I/O y de leer umbrales (RSI/ADX/BB) desde `settings`.

**Dependencias core:** `numpy`, `pandas`, `scipy`.
**Dependencias opcionales** (con guards de `ImportError`): `statsmodels`, `arch`, `pmdarima`, `scikit-learn`.

---

## Estructura del paquete

```
src/                         # Sources Root — cada subpaquete es importable top-level
    core/
        types.py             # Frozen dataclasses para indicadores multi-output
        validation.py        # Validacion OHLCV, series y periodos
        precision.py         # Coercion a float64
        smoothing.py         # Wilder smoothing vs EMA estandar
    trend/
        sma.py, ema.py, rsi.py, macd.py, stochastic.py, ichimoku.py
    volatility/
        bollinger.py, atr.py, adx.py, vwap.py, obv.py, adl.py
    risk/
        sharpe.py, sortino.py, drawdown.py, var.py, cvar.py
    stats/
        returns.py, volatility_models.py, correlation.py, regression.py, timeseries.py
    quant/
        black_scholes.py, portfolio.py, capm.py, monte_carlo.py
    systems/
        costs.py, position_sizing.py, stop_loss.py, order_types.py, walk_forward.py, backtest.py
```

### Inventario de modulos por subpaquete

Referencia rapida: archivo, simbolos publicos y que calcula cada uno. Los detalles (formulas, edge cases, validacion contra TA-Lib) estan en las secciones posteriores.

**`core/` (4 archivos) — primitivas compartidas:**

| Archivo | Simbolos | Que hace |
|---|---|---|
| `types.py` | `MACDResult`, `BollingerResult`, `StochasticResult`, `IchimokuResult`, `ADXResult`, `DrawdownResult` | Frozen dataclasses para indicadores multi-output (habilita type safety + IDE) |
| `validation.py` | `validate_ohlcv`, `validate_series`, `validate_period` | Validacion centralizada: case-insensitive, integridad `High>=Low/Close/Open`, float64, sort index, guard de NaN |
| `precision.py` | `ensure_float64` | Coercion a float64 para compatibilidad TA-Lib |
| `smoothing.py` | `wilder_smooth` (alpha=1/N), `ema_smooth` (alpha=2/(N+1), `sma_seed`) | **Distincion Wilder vs EMA estandar** — ver seccion dedicada |

**`trend/` (6 archivos) — indicadores de tendencia/momentum:**

| Archivo | Funcion publica | Que calcula |
|---|---|---|
| `sma.py` | `sma(close, period=20)` | Media movil simple (ventanas 20/50/100/200) |
| `ema.py` | `ema(close, period=20, sma_seed=True)` | Media movil exponencial (alpha=2/(N+1), TA-Lib compat con `sma_seed`) |
| `rsi.py` | `rsi(close, period=14)` | Relative Strength Index con **Wilder smoothing** |
| `macd.py` | `macd(close, 12, 26, 9)` | MACD line/signal/histogram, retorna `MACDResult` |
| `stochastic.py` | `stochastic`, `stochastic_fast` | Estocastico lento/rapido con guard de `High==Low`, retorna `StochasticResult` |
| `ichimoku.py` | `ichimoku(high, low, close, ...)` | Kinko Hyo (tenkan/kijun/senkou A,B/chikou) con shifts +-26, retorna `IchimokuResult` |

**`volatility/` (6 archivos) — volatilidad y volumen:**

| Archivo | Funcion publica | Que calcula |
|---|---|---|
| `atr.py` | `true_range(h,l,c)`, `atr(h,l,c,period=14)` | True Range (captura gaps overnight) y ATR con **Wilder smoothing** (ver `volatility/atr.py:40-64`) |
| `bollinger.py` | `bollinger_bands(close, 20, 2.0)` | Upper/middle/lower + `%B` + bandwidth con `ddof=0` (TA-Lib compat) |
| `adx.py` | `adx(h,l,c,period=14)` | +DI/-DI/ADX via cadena Wilder de 6 pasos, retorna `ADXResult` |
| `vwap.py` | `vwap(h,l,c,volume)` | VWAP acumulativo con reset diario si el indice es `DatetimeIndex` |
| `obv.py` | `obv(close, volume)` | On Balance Volume (acumulativo binario, sin lookback) |
| `adl.py` | `adl(h,l,c,volume)` | Accumulation/Distribution con Money Flow Multiplier |

**`stats/` (5 archivos) — estadistica y series temporales:**

| Archivo | Simbolos | Que calcula |
|---|---|---|
| `returns.py` | `simple_returns`, `log_returns` | Retornos aritmeticos (cross-asset additive) y logaritmicos (time-additive) |
| `volatility_models.py` | `rolling_volatility`, `ewma_volatility`, `parkinson_volatility` | Volatilidad rolling (ddof=1), EWMA (RiskMetrics lambda=0.94), estimador Parkinson (H/L) |
| `correlation.py` | `correlation_matrix`, `rolling_correlation`, `covariance_shrinkage`, `is_psd` | Pearson/Spearman/Kendall, Ledoit-Wolf (requires sklearn), check PSD via eigenvalores |
| `regression.py` | `simple_ols` (+ `RegressionResult`), `ols_statsmodels` (+HAC Newey-West), `rolling_ols` | Regresion via `scipy.stats.linregress` + statsmodels opcional |
| `timeseries.py` | `fit_arima` (+`ARIMAResult`), `auto_arima` (pmdarima), `fit_garch` (+`GARCHResult`) | ARIMA(p,d,q), Hyndman-Khandakar stepwise, GARCH(p,q) con distribucion `t`/`normal`/`skewt`. **GARCH es el modelo de volatilidad avanzada** de este subpaquete (ademas de los 3 de `volatility_models.py`). |

**`risk/` (5 archivos) — metricas de riesgo/rendimiento:**

| Archivo | Simbolos | Que calcula |
|---|---|---|
| `sharpe.py` | `sharpe_ratio(returns, rf=0, periods_per_year=252)` | Sharpe con `ddof=1`, rf per-period via compounding, NaN si std<1e-12 |
| `sortino.py` | `sortino_ratio(returns, rf=0, target=None, ...)` | Sortino con downside deviation sobre **todas las obs** (denom=n) |
| `drawdown.py` | `max_drawdown(returns)` | Retorna `DrawdownResult` con peak/trough/recovery dates y duration |
| `var.py` | `var_historical`, `var_parametric`, `var_monte_carlo` | VaR por percentil/gaussiana/MC (ddof=0, signo positivo = perdida) |
| `cvar.py` | `cvar_historical`, `cvar_parametric` | Expected Shortfall (Basel III/IV), NaN si no hay tail observations |

**`quant/` (4 archivos) — finanzas cuantitativas:**

| Archivo | Simbolos | Que calcula |
|---|---|---|
| `black_scholes.py` | `bs_price`, `bs_greeks` (+`BSResult`), `implied_vol` | Pricing + griegas (theta/365, vega/100), IV via Brent en `[1e-6, 10]` |
| `portfolio.py` | `max_sharpe_portfolio`, `min_variance_portfolio` (+`PortfolioResult`) | Markowitz mean-variance via SLSQP, bounds `[0,1]`, sum=1 |
| `capm.py` | `capm_beta` (+`BetaResult`), `rolling_beta` | CAPM beta/alpha/R2 + Bloomberg adjusted beta (2/3 raw + 1/3 * 1.0) |
| `monte_carlo.py` | `gbm_paths`, `bootstrap_paths` | GBM via log-scheme (evita precios negativos), bootstrap no-parametrico |

**`systems/` (6 archivos) — diseno de sistemas de trading:**

| Archivo | Simbolos | Que hace |
|---|---|---|
| `costs.py` | `TransactionCostModel` (dataclass), `slippage_percentage`, `market_impact_sqroot` | Comision + spread + slippage + modelo Almgren sqrt |
| `position_sizing.py` | `kelly_discrete`, `kelly_continuous`, `position_size` | Kelly (discreto/continuo) + wrapper half-Kelly con cap al 25% |
| `stop_loss.py` | `fixed_stop`, `trailing_stop` (+`StopResult`), `atr_stop` | Stops fijos/trailing/ATR-based |
| `order_types.py` | `OrderSimulator.fill_market/limit/stop/stop_limit` (+`FillResult`) | Simulacion de fills con guard de gaps overnight |
| `backtest.py` | `backtest_signals(...)` (+`BacktestResult`) | Motor long-only con anti-lookahead (`position.shift(1)`) y aplicacion bar-a-bar de costos |
| `walk_forward.py` | `rolling_walk_forward`, `expanding_walk_forward` | Generadores `(train, test)` con gap configurable para evitar data leakage |

---

### Core: tipos de retorno

Los indicadores multi-output retornan frozen dataclasses en lugar de DataFrames, lo que habilita autocompletado en IDEs y type safety:

```python
from core.types import (
    MACDResult, BollingerResult, StochasticResult,
    IchimokuResult, ADXResult, DrawdownResult,
)

result = macd(close)
result.macd       # pd.Series
result.signal     # pd.Series
result.histogram  # pd.Series
```

### Core: validacion

Toda entrada cruza por una de las tres funciones de `core/validation.py` antes del calculo. Esto evita que indicadores silenciosamente produzcan NaN o crashes confusos.

`validate_ohlcv(df, require_volume=True)` (`core/validation.py:10-68`) — acepta columnas case-insensitive (`Open`/`OPEN`/`open`), rechaza NaN en columnas `high`/`low`/`close` con `ValueError` (no chequea `open`), valida integridad de datos (`High >= Low`, `High >= Close`, `High >= Open`), rechaza volumen negativo, convierte a float64 via `ensure_float64`, ordena indice no monotono creciente, retorna **copia** limpia (no muta la entrada).

`validate_series(series, name, min_length)` (`core/validation.py:71-99`) — valida tipo `pd.Series` (raise `TypeError`), longitud minima (`ValueError`), retorna copia float64. Usado por todos los indicadores single-input (RSI, ATR, EMA, etc.).

`validate_period(period, name)` (`core/validation.py:102-105`) — valida entero positivo, rechaza `0`, negativos, floats y bools.

**Contrato de error:** Los indicadores lanzan `ValueError`/`TypeError` en validacion, NO retornan `NaN`. Los NaN que aparecen en la salida provienen del warmup natural del indicador (ej. primeros `period-1` bars de un SMA), no de datos invalidos.

### Core: smoothing — Wilder vs EMA estandar (distincion CRITICA)

Confundir Wilder con EMA es el error de implementacion mas comun en analisis tecnico. La distincion esta centralizada en `core/smoothing.py` y se usa consistentemente en los indicadores:

| Formula | Archivo:linea | alpha(period=14) | Usado en |
|---|---|---|---|
| **Wilder** `avg_t = (avg_{t-1} * (N-1) + x_t) / N` | `core/smoothing.py:14-49` | `1/14 = 0.0714` | `trend/rsi.py:47-49` (RSI), `volatility/atr.py:61-62` (ATR), `volatility/adx.py` (cadena +DM/-DM/DX/ADX) |
| **EMA estandar** `EMA_t = alpha * x_t + (1-alpha) * EMA_{t-1}` | `core/smoothing.py:52-85` | `2/15 = 0.1333` | `trend/ema.py` (EMA), `trend/macd.py` (via `pandas.ewm`), `volatility/bollinger.py` |

**Seeds:** Wilder siempre inicializa con SMA de los primeros N valores (TA-Lib compat). `ema_smooth(sma_seed=True)` replica ese comportamiento; `sma_seed=False` usa el primer valor como seed (equivalente a `pandas.ewm(adjust=False)`).

**Impacto numerico:** Para una serie de 30 dias con periodo 14, Wilder converge ~30% mas lento que EMA estandar, pero es mas estable frente a outliers. Swap accidental entre ambos inflate o deflate los valores de salida hasta un ~5% en los primeros 10*N bars.

### Patron de imports

```python
# Layout plano: src/ es Sources Root, cada subpaquete se importa directamente
from trend.sma import sma
from trend.ema import ema
from trend.rsi import rsi
from trend.macd import macd
from volatility.bollinger import bollinger_bands
from volatility.atr import atr
from volatility.adx import adx
from risk.sharpe import sharpe_ratio
from risk.sortino import sortino_ratio
from risk.drawdown import max_drawdown
from stats.returns import simple_returns, log_returns

# Subpaquetes con dependencias opcionales
from stats.timeseries import fit_arima, fit_garch
from quant.black_scholes import bs_price, bs_greeks, implied_vol
from systems.backtest import backtest_signals
```

### Dependencias opcionales

Funciones que requieren librerias opcionales lanzan `ImportError` con instrucciones de instalacion:

```python
# stats/timeseries.py: fit_arima, auto_arima -> requieren statsmodels / pmdarima
# stats/timeseries.py: fit_garch -> requiere arch
# stats/correlation.py: covariance_shrinkage -> requiere scikit-learn
# stats/regression.py: ols_statsmodels -> requiere statsmodels
```

---

## Indicadores de tendencia y momentum

### SMA (Simple Moving Average)

```python
def sma(close: pd.Series, period: int = 20) -> pd.Series
```

Valores comunes de periodo: **20, 50, 100, 200**. Los primeros `period - 1` valores son NaN.

**Edge cases:** Cuando `len(data) < period`, toda la salida es NaN. Un solo NaN en la serie de entrada se propaga por la ventana rolling.

Numericameente equivalente a `talib.SMA(close, timeperiod=N)`.

---

### EMA (Exponential Moving Average)

```python
def ema(close: pd.Series, period: int = 20, sma_seed: bool = True) -> pd.Series
```

Factor de suavizado: **k = 2/(n+1)**. Para periodo 12, k ~ 0.1538; para periodo 26, k ~ 0.0741. Formula recursiva: `EMA_t = k * close_t + (1-k) * EMA_{t-1}`.

El parametro `sma_seed` controla la inicializacion:
- `sma_seed=True` (default): inicializa con SMA de primeros N valores. Compatible con TA-Lib.
- `sma_seed=False`: inicializa con el primer valor. Equivalente a pandas `ewm(adjust=False)`.

Para series largas ambos convergen; para series cortas la divergencia es significativa.

**Advertencia de convergencia:** Los primeros ~10*N periodos tienen errores de precision de hasta ~5%.

---

### RSI (Relative Strength Index)

```python
def rsi(close: pd.Series, period: int = 14) -> pd.Series
```

Usa **Wilder smoothing (alpha = 1/N)**, NO EMA estandar (alpha = 2/(N+1)). Para periodo 14: Wilder alpha = 1/14 ~ 0.0714, no 2/15 ~ 0.1333. Rango de salida: 0-100. Sobrecompra >= 70, sobreventa <= 30.

```python
# Algoritmo interno:
# 1. SMA seed sobre primeros `period` cambios (indices 1..period)
# 2. Wilder smoothing: avg_t = (avg_{t-1} * (period-1) + current) / period
```

**Edge cases:** Todas ganancias -> RSI = 100 (avg_loss = 0, RS = inf). Todas perdidas -> RSI = 0. La division por cero se maneja naturalmente: 100/(1+inf) = 0.

**Advertencia de convergencia:** Los primeros ~10*N periodos tienen errores de precision de hasta ~5%. Usar al menos 10*N puntos de datos antes de la ventana de analisis.

Validado contra `talib.RSI(close, timeperiod=N)`.

---

### MACD (Moving Average Convergence Divergence)

```python
def macd(
    close: pd.Series,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> MACDResult
```

**Retorna `MACDResult`** con atributos `.macd`, `.signal`, `.histogram` (cada uno `pd.Series`).

```python
result = macd(close)
macd_line = result.macd        # EMA(fast) - EMA(slow)
signal_line = result.signal    # EMA(MACD, signal_period)
hist = result.histogram        # MACD - Signal
```

**Nota de implementacion:** Usa `pandas.ewm(adjust=False)` para las EMAs subyacentes, NO inicializacion SMA como TA-Lib. Esto produce diferencias menores en series cortas vs `talib.MACD()`, pero convergen en series largas.

**Deteccion de senales:** Cruce alcista cuando MACD cruza por encima de Signal; cruce de linea cero cuando MACD cruza por encima de cero. Periodos alternativos comunes: (8,17,9) y (5,35,5).

---

### Oscilador Estocastico (%K, %D)

```python
# Estocastico lento
def stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series,
    fastk_period: int = 14, slowk_period: int = 3, slowd_period: int = 3,
) -> StochasticResult

# Estocastico rapido
def stochastic_fast(
    high: pd.Series, low: pd.Series, close: pd.Series,
    fastk_period: int = 14, fastd_period: int = 3,
) -> StochasticResult
```

**Retorna `StochasticResult`** con atributos `.k` y `.d`.

**Formula:** Fast %K = (Close - Lowest_Low_N) / (Highest_High_N - Lowest_Low_N) * 100. Slow %K = SMA(Fast %K, 3). Slow %D = SMA(Slow %K, 3). Rango: 0-100; sobrecompra >= 80, sobreventa <= 20.

**Edge case critico:** Cuando High == Low (rango cero), retorna 0 (convencion TA-Lib). El default `fastk_period=14` sigue la practica comun; TA-Lib usa 5 por defecto.

---

### Ichimoku Kinko Hyo

```python
def ichimoku(
    high: pd.Series, low: pd.Series, close: pd.Series,
    tenkan_period: int = 9, kijun_period: int = 26,
    senkou_b_period: int = 52, displacement: int = 26,
) -> IchimokuResult
```

**Retorna `IchimokuResult`** con atributos `.tenkan`, `.kijun`, `.senkou_a`, `.senkou_b`, `.chikou`.

| Componente | Formula | Periodo |
|------------|---------|---------|
| Tenkan-sen | (9-period High + Low) / 2 | 9 |
| Kijun-sen | (26-period High + Low) / 2 | 26 |
| Senkou Span A | (Tenkan + Kijun) / 2, **shift +26** | — |
| Senkou Span B | (52-period High + Low) / 2, **shift +26** | 52 |
| Chikou Span | Close, **shift -26** | — |

**Manejo de bordes:** `.shift(26)` causa que los ultimos 26 valores sean NaN. Para graficos, extender el indice 26 periodos hacia el futuro. Requerimiento minimo de datos: **78 periodos** (52 + 26).

No disponible en TA-Lib estandar.

---

## Indicadores de volatilidad y volumen

### Bollinger Bands

```python
def bollinger_bands(
    close: pd.Series, period: int = 20, num_std: float = 2.0,
) -> BollingerResult
```

**Retorna `BollingerResult`** con atributos `.upper`, `.middle`, `.lower`, `.pct_b`, `.bandwidth`.

```python
bb = bollinger_bands(close)
bb.upper      # Middle + num_std * std
bb.lower      # Middle - num_std * std
bb.pct_b      # (Close - Lower) / (Upper - Lower)
bb.bandwidth  # (Upper - Lower) / Middle
```

**Pitfall critico: usa desviacion estandar poblacional (ddof=0)**, compatible con TA-Lib. Pandas por defecto usa `ddof=1`. La diferencia es significativa con ventanas pequenas.

**Deteccion de squeeze:** Bandwidth < ~0.02 (2%) senala expansion de volatilidad inminente. **%B** normalmente entre 0-1; valores fuera de ese rango indican precio fuera de las bandas.

---

### True Range / ATR

```python
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series
def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series
```

True Range captura gaps overnight: `TR = max(H-L, |H-PrevClose|, |L-PrevClose|)`. La primera barra usa `H-L` (sin close previo).

**ATR usa Wilder smoothing (alpha = 1/period), NO SMA.** El primer ATR = promedio simple de los primeros N true ranges, luego suavizado recursivo. Los primeros `period` valores son NaN.

---

### ADX (Average Directional Index)

```python
def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14,
) -> ADXResult
```

**Retorna `ADXResult`** con atributos `.adx`, `.plus_di`, `.minus_di`.

El indicador estandar mas complejo, con una **cadena de 6 pasos** con Wilder smoothing:

1. **+DM / -DM:** Movimiento direccional desde altos/bajos consecutivos
2. **Wilder smooth** TR, +DM, -DM (suma primeros N, luego `prev - prev/period + current`)
3. **+DI / -DI:** `100 * smoothed_DM / smoothed_TR`
4. **DX:** `100 * |+DI - -DI| / (+DI + -DI)`
5. **ADX:** Wilder smoothing de DX (primer ADX = promedio de primeros N valores DX)

**ADX mide fuerza de tendencia, no direccion.** ADX > 25 = tendencia fuerte, < 20 = tendencia debil/inexistente. **Lookback es ~2*period**. Edge case: guard `+DI + -DI = 0` para division de DX.

---

### VWAP (Volume Weighted Average Price)

```python
def vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
) -> pd.Series
```

Precio tipico = (H + L + C) / 3, ponderado acumulativamente por volumen.

**Reset automatico de sesion:** Si el indice tiene atributo `.date` (DatetimeIndex), agrupa por fecha para reset diario. Si no, calcula acumulativo continuo. NaN donde el volumen acumulado es cero.

No disponible en TA-Lib estandar.

---

### OBV (On Balance Volume)

```python
def obv(close: pd.Series, volume: pd.Series) -> pd.Series
```

Indicador acumulativo: close > prev_close suma volumen; close < prev_close resta volumen. Cuando `close == prev_close`, el volumen NO se agrega ni sustrae. Sin lookback — inicia desde barra 0.

Usa `float64` para evitar overflow con instrumentos de alto volumen en periodos largos.

---

### ADL (Accumulation/Distribution Line)

```python
def adl(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
) -> pd.Series
```

**Money Flow Multiplier** = ((Close - Low) - (High - Close)) / (High - Low). Rango: -1 (close en low) a +1 (close en high). A diferencia de OBV (enfoque binario), ADL pondera volumen proporcionalmente. Edge case: High == Low → MFM = 0 (sin informacion de rango).

---

## Metricas de riesgo y rendimiento

### Sharpe Ratio

```python
def sharpe_ratio(
    returns: np.ndarray, rf: float = 0.0, periods_per_year: int = 252,
) -> float
```

**Formula:** `(Rp - Rf) / sigma_p * sqrt(N)` donde N = 252 (diario), 52 (semanal), 12 (mensual).

**Detalles clave:**
- Usa `ddof=1` (desv. estandar muestral) consistentemente.
- Tasa libre de riesgo convertida a per-period: `rf_daily = (1 + rf_annual)^(1/252) - 1`, no `rf/252`.
- Retorna `np.nan` cuando `std < 1e-12` o `len(returns) < 2`.
- Filtra NaN automaticamente.

**Benchmarks:** >1 bueno, >2 muy bueno, >3 excelente.

---

### Sortino Ratio

```python
def sortino_ratio(
    returns: np.ndarray, rf: float = 0.0, target: float | None = None,
    periods_per_year: int = 252,
) -> float
```

**Formula:** `(Rp - Rf) / sigma_downside * sqrt(N)`

**Detalle critico:** La desviacion downside usa **TODAS las observaciones** en el denominador (n), no solo los retornos negativos. Retornos por encima del target contribuyen 0 a la suma. Este es un error de implementacion comun.

Retorna `np.inf` cuando no hay riesgo downside (dd < 1e-12). Si `target=None`, usa la tasa libre de riesgo per-period.

---

### Maximum Drawdown

```python
def max_drawdown(returns: pd.Series) -> DrawdownResult
```

**Retorna `DrawdownResult`** con tracking completo:

```python
result = max_drawdown(returns)
result.drawdown        # pd.Series — serie completa de drawdown
result.max_drawdown    # float — drawdown maximo (valor negativo)
result.peak_date       # indice del pico previo al drawdown
result.trough_date     # indice del punto mas bajo
result.recovery_date   # indice de recuperacion (None si no recupero)
result.duration        # periodos de pico a recuperacion (None si no recupero)
```

Calmar Ratio = Annual Return / |MaxDrawdown|. Edge case: serie monotonicamente creciente produce drawdown cero.

---

### VaR (Value at Risk) — tres metodos

```python
def var_historical(returns: np.ndarray, confidence: float = 0.95) -> float
def var_parametric(returns: np.ndarray, confidence: float = 0.95) -> float
def var_monte_carlo(
    returns: np.ndarray, confidence: float = 0.95,
    n_sims: int = 10_000, seed: int = 42,
) -> float
```

**Convencion de signo: retorna magnitud POSITIVA de perdida.** Por ejemplo, 0.05 significa 5% de perdida potencial.

- **Historico:** Percentil empirico.
- **Parametrico:** Asume distribucion normal. Usa `ddof=0`.
- **Monte Carlo:** Simula con `np.random.default_rng(seed)`.

| Confianza | z-score | Uso |
|-----------|---------|-----|
| 95% | 1.645 | Reporte estandar |
| 99% | 2.326 | Regulatorio Basel |
| 99.9% | 3.090 | Estres extremo |

**Escalamiento temporal:** `VaR_T = VaR_1 * sqrt(T)` (solo bajo supuesto i.i.d. normal).

---

### Expected Shortfall (CVaR)

```python
def cvar_historical(returns: np.ndarray, confidence: float = 0.95) -> float
def cvar_parametric(returns: np.ndarray, confidence: float = 0.95) -> float
```

Media de perdidas mas alla del VaR — **medida de riesgo coherente** (subaditiva, a diferencia de VaR). Requerida por Basel III/IV para riesgo de mercado desde 2019.

- **Parametrico:** `ES = -mu + sigma * phi(Phi^{-1}(alpha)) / alpha`
- **Edge case:** Con 250 observaciones al 99% de confianza, solo ~2-3 observaciones caen en la cola. Retorna `np.nan` si no hay observaciones en la cola. ES siempre >= VaR para el mismo nivel de confianza.

---

## Fundamentos estadisticos y matematicos

### Retornos logaritmicos vs simples

```python
def simple_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame
def log_returns(prices: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame
```

**Retornos log son aditivos en el tiempo** (multi-periodo = suma de single-periodo). **Retornos simples son aditivos entre activos** (retorno portfolio = suma ponderada). Usar retornos log para analisis de series temporales y Black-Scholes; retornos simples para calculos cross-section de portfolio.

Ambas funciones aceptan Series o DataFrames y retornan el mismo tipo. Primer valor siempre NaN.

**Edge cases:** Precios cero o negativos hacen los retornos log indefinidos — validar `prices > 0`.

---

### Modelos de volatilidad

```python
def rolling_volatility(returns: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series
def ewma_volatility(returns: pd.Series, span: int = 30, annualize: bool = True) -> pd.Series
def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 21, annualize: bool = True) -> pd.Series
```

- **Rolling:** Desviacion estandar rolling con `ddof=1`.
- **EWMA:** Modelo RiskMetrics (lambda=0.94 -> span ~ 32.33).
- **Parkinson:** Estimador basado en rango (H/L). 5x mas eficiente que close-to-close para procesos GBM, pero pierde gaps overnight.

**Anualizacion:** diario * sqrt(252), semanal * sqrt(52), mensual * sqrt(12), horario * sqrt(252*6.5). Todas usan `ddof=1` y `annualize=True` por defecto.

---

### Matriz de correlacion

```python
def correlation_matrix(returns: pd.DataFrame, method: str = "pearson") -> pd.DataFrame
def rolling_correlation(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame
def covariance_shrinkage(returns: pd.DataFrame) -> tuple[np.ndarray, float]
def is_psd(matrix: np.ndarray, tol: float = 1e-10) -> bool
```

- `correlation_matrix`: Soporta pearson, spearman, kendall.
- `covariance_shrinkage`: Ledoit-Wolf shrinkage. Siempre PSD, bien condicionada. **Requiere scikit-learn** (ImportError guard). Retorna `(cov_matrix, shrinkage_coefficient)`.
- `is_psd`: Verifica positive semi-definiteness via eigenvalores.

**Edge cases:** Serie constante produce correlacion NaN. Cuando n_samples < n_features, covarianza muestral es singular — Ledoit-Wolf es esencial. Correlaciones rolling pueden perder PSD — verificar con `is_psd()`.

---

### Regresion lineal (OLS)

```python
def simple_ols(x: pd.Series, y: pd.Series) -> RegressionResult
def ols_statsmodels(y: pd.Series, X: pd.DataFrame, hac_lags: int | None = None)
def rolling_ols(y: pd.Series, x: pd.Series, window: int = 60) -> pd.DataFrame
```

- `simple_ols`: Usa `scipy.stats.linregress`. Retorna `RegressionResult` (slope, intercept, r_squared, p_value, stderr). Filtra NaN automaticamente.
- `ols_statsmodels`: Agrega constante automaticamente (no olvidar `sm.add_constant()` ya no es un pitfall). Con `hac_lags` usa errores estandar HAC (Newey-West). **Requiere statsmodels** (ImportError guard).
- `rolling_ols`: Vectorizado via covarianza rolling. Retorna DataFrame con columnas: beta, alpha, r_squared.

Para series temporales: verificar heterocedasticidad (test Breusch-Pagan), autocorrelacion (Durbin-Watson, Ljung-Box), y multicolinealidad (VIF > 10).

---

### Modelos ARIMA/GARCH

```python
def fit_arima(series: pd.Series, order: tuple[int,int,int] = (1,0,1)) -> ARIMAResult
def auto_arima(series: pd.Series, max_p: int = 5, max_q: int = 5, seasonal: bool = False)
def fit_garch(returns: pd.Series, p: int = 1, q: int = 1, dist: str = "t", mean: str = "Constant") -> GARCHResult
```

**Tipos de retorno:**
- `ARIMAResult`: `.model`, `.aic`, `.bic`, `.params` (dict)
- `GARCHResult`: `.model`, `.params` (dict), `.conditional_volatility` (pd.Series), `.converged` (bool)

**Dependencias opcionales:** `fit_arima` requiere statsmodels, `auto_arima` requiere pmdarima, `fit_garch` requiere arch. Todos con ImportError guard.

**ARIMA:** Nunca aplicar a precios crudos (no estacionarios). Usar retornos o diferencias.

**GARCH(1,1):** `sigma^2_t = omega + alpha * epsilon^2_{t-1} + beta * sigma^2_{t-1}`. Restriccion de estacionariedad: alpha + beta < 1. **Convencion arch:** p = GARCH lags (sigma^2), q = ARCH lags (epsilon^2) — invertido respecto a algunos libros. Retornos porcentuales (*100) convergen mejor que retornos decimales. Forecasts multi-step revierten hacia la varianza incondicional.

**Flujo de diagnostico:** Test ADF para estacionariedad -> Ljung-Box en residuos al cuadrado para efectos ARCH -> fit GARCH -> verificar `converged == True`.

---

## Modulos de diseno de sistemas

### Motor de backtesting

```python
def backtest_signals(
    close: pd.Series, entries: pd.Series, exits: pd.Series,
    init_cash: float = 10_000.0, cost_model: TransactionCostModel | None = None,
) -> BacktestResult
```

Motor de backtesting basado en senales (long-only). **No depende de vectorbt ni backtrader.** El tracking de posición se hace con un loop explícito sobre las barras (ver `systems/backtest.py:58-66`); los retornos aplicados al `position.shift(1)` sí son vectorizados. Los costos de transacción se aplican bar-a-bar sobre los eventos de entry/exit.

**Retorna `BacktestResult`:**
- `.equity_curve` (pd.Series) — curva de equity
- `.returns` (pd.Series) — retornos de la estrategia
- `.total_return` (float) — retorno total
- `.annual_return` (float) — retorno anualizado
- `.n_trades` (int) — numero de trades completados
- `.win_rate` (float) — tasa de acierto

```python
from systems.backtest import backtest_signals
from systems.costs import TransactionCostModel

cost = TransactionCostModel(commission_per_share=0.005, half_spread_bps=2.5)
result = backtest_signals(close, entries, exits, init_cash=10_000, cost_model=cost)
```

**Anti lookahead bias:** La posicion se aplica a los retornos de la barra siguiente (`position.shift(1) * market_returns`).

**Pitfalls criticos:** Lookahead bias (usar datos futuros), survivorship bias, y overfitting via miles de combinaciones de parametros sin validacion out-of-sample.

---

### Walk-forward validation

```python
# Ventana fija (rolling)
def rolling_walk_forward(
    data, train_size: int = 252, test_size: int = 63,
    step_size: int = 21, gap: int = 0,
) -> Generator[tuple, None, None]

# Ventana expandible
def expanding_walk_forward(
    data, min_train_size: int = 252, test_size: int = 63,
    step_size: int = 21, gap: int = 0,
) -> Generator[tuple, None, None]
```

Ambos son generadores que producen tuplas `(train, test)`. Defaults: train_size = 252 (1 anio), test_size = 63 (1 trimestre), step_size = 21 (1 mes).

**Parametro gap** previene data leakage por features/labels que se solapan. Ventanas rolling se adaptan mas rapido a cambios de regimen que ventanas expandibles.

---

### Position sizing con Kelly Criterion

```python
def kelly_discrete(win_rate: float, avg_win: float, avg_loss: float) -> float
def kelly_continuous(returns: np.ndarray) -> float
def position_size(kelly_fraction: float, fraction: float = 0.5, max_pct: float = 0.25) -> float
```

- **Discreto:** `f* = (b*p - q) / b` donde b = payoff ratio.
- **Continuo:** `f* = mu / sigma^2`.
- **`position_size`:** Wrapper practico. Aplica half-Kelly por defecto (`fraction=0.5`) y cap al 25% (`max_pct=0.25`). Retorna 0 si Kelly es negativo (expectativa negativa).

**Half-Kelly es la practica estandar** — reduce varianza 50% sacrificando solo ~25% de crecimiento. El error de estimacion de parametros hace que full Kelly sea peligroso; Kelly fraccional conservador (0.25-0.5) es estandar de produccion.

---

### Implementaciones de stop loss

```python
def fixed_stop(entry_price: float, prices: pd.Series, stop_pct: float = 0.02) -> StopResult
def trailing_stop(prices: pd.Series, trail_pct: float = 0.03) -> StopResult
def atr_stop(entry_price: float, atr_values: pd.Series, multiplier: float = 3.0) -> pd.Series
```

`fixed_stop` y `trailing_stop` retornan `StopResult` con:
- `.stop_level` (pd.Series) — nivel de stop a lo largo del tiempo
- `.triggered_date` — fecha de activacion (None si no se activo)
- `.triggered_price` — precio de activacion (None si no se activo)

`atr_stop` retorna `pd.Series`: `entry_price - multiplier * ATR`. Multiplier tipico: 2-3.

**Edge case:** Gaps a traves del stop significan fill al precio de apertura, no al stop. Stops basados en ATR se adaptan a volatilidad automaticamente.

---

### Simulacion de tipos de orden

```python
class OrderSimulator:
    def fill_market(self, bar: dict) -> FillResult
    def fill_limit(self, bar: dict, limit_price: float, side: str = "buy") -> FillResult
    def fill_stop(self, bar: dict, stop_price: float, side: str = "sell") -> FillResult
    def fill_stop_limit(self, bar: dict, stop_price: float, limit_price: float, side: str = "sell") -> FillResult
```

Cada metodo retorna `FillResult(filled: bool, fill_price: float | None)`. El `bar` es un dict con keys: open, high, low, close.

- **Market:** Fill al open de la siguiente barra.
- **Limit:** Buy si low <= limit; sell si high >= limit. Fill al min/max(open, limit).
- **Stop:** Logica inversa al limit.
- **Stop-limit:** Logica de dos etapas — primero verifica si el stop se activa, luego si el limit es alcanzable.

**Edge case principal:** Gap de apertura mas alla del precio limit/stop — la orden puede no ejecutarse o ejecutarse a peor precio.

---

### Modelos de costos de transaccion

```python
@dataclass
class TransactionCostModel:
    commission_per_share: float = 0.005
    half_spread_bps: float = 2.5
    min_commission: float = 1.0

    def total_cost(self, price: float, shares: float) -> float

def slippage_percentage(price: float, slippage_pct: float, side: str = "buy") -> float
def market_impact_sqroot(sigma: float, order_qty: float, market_volume: float, eta: float = 0.1) -> float
```

- `total_cost`: Comision + spread. Comision minima aplicada.
- `slippage_percentage`: `fill = price * (1 + sign * slippage_pct)`.
- `market_impact_sqroot`: Modelo Almgren. `MI = eta * sigma * sqrt(V_order / V_market)`.

**Costos tipicos:** Ida y vuelta en large-cap liquido = **5-15 bps** (comision + spread + impacto). Small-cap = 50-100+ bps. IB Fixed: $0.005/accion, min $1.00. Brokers sin comision igual tienen spread de ~5 bps.

---

## Modulos avanzados de finanzas cuantitativas

### Black-Scholes: pricing y Greeks

```python
def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> float
def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "call") -> BSResult
def implied_vol(market_price: float, S: float, K: float, T: float, r: float, option_type: str = "call") -> float
```

`bs_greeks` retorna `BSResult` con: `.price`, `.delta`, `.gamma`, `.theta`, `.vega`, `.rho`.

- **Theta:** Por dia calendario (/365).
- **Vega:** Por 1% de movimiento de volatilidad (/100).
- **Gamma:** Igual para call y put.

**Edge cases:** T <= 0 retorna valor intrinseco. sigma <= 0 retorna valor descontado. `implied_vol` usa metodo de Brent (`brentq`) en rango [1e-6, 10.0]; retorna NaN si no encuentra solucion.

Al vencimiento (T→0): delta salta a 0 o 1, gamma pica y colapsa, vega → 0.

---

### Optimizacion de portfolio (Markowitz mean-variance)

```python
def max_sharpe_portfolio(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0) -> PortfolioResult
def min_variance_portfolio(cov: np.ndarray) -> PortfolioResult
```

**Retorna `PortfolioResult`** con `.weights` (np.ndarray), `.expected_return`, `.volatility`, `.sharpe`.

Usa `scipy.optimize.minimize` con metodo SLSQP, bounds [0,1] y restriccion de suma = 1.

**Edge cases:** Covarianza singular → usar `covariance_shrinkage()` o agregar `1e-6 * I`. Soluciones de esquina (100% en un activo) → agregar regularizacion L2.

---

### CAPM beta

```python
def capm_beta(stock_returns: pd.Series, market_returns: pd.Series) -> BetaResult
def rolling_beta(stock_returns: pd.Series, market_returns: pd.Series, window: int = 60) -> pd.Series
```

`capm_beta` retorna `BetaResult` con `.beta`, `.alpha`, `.r_squared`, `.adjusted_beta`.

**Bloomberg adjusted beta** se calcula automaticamente: `(2/3) * raw_beta + (1/3) * 1.0` (reversion a la media hacia 1.0).

`rolling_beta` usa covarianza rolling vectorizada. Filtra NaN automaticamente via `linregress`.

**La frecuencia importa:** Retornos diarios vs semanales producen betas diferentes.

---

### Simulacion Monte Carlo

```python
def gbm_paths(S0: float, mu: float, sigma: float, T: float, n_steps: int, n_paths: int, seed: int | None = None) -> np.ndarray
def bootstrap_paths(S0: float, hist_log_returns: np.ndarray, n_days: int, n_paths: int, seed: int | None = None) -> np.ndarray
```

- **GBM:** `S_t = S_0 * exp((mu - sigma^2/2)*t + sigma*sqrt(t)*Z)`. Retorna shape `(n_steps + 1, n_paths)`.
- **Bootstrap:** No parametrico, preserva fat tails. Muestrea con reemplazo de retornos log historicos. Retorna shape `(n_days + 1, n_paths)`.

**Siempre usar el esquema log** (`exp(cumsum)`) — la discretizacion de Euler puede producir precios negativos. MC converge a **O(1/sqrt(N))**, entonces 4x paths da 2x precision. Memoria: 1M paths * 252 steps * 8 bytes ~ 2GB — procesar en lotes. Siempre usar `np.random.default_rng(seed)`, no el legacy `np.random.seed()`.

---

## Patrones transversales de implementacion

### Reglas de precision numerica

1. **Siempre float64** — forzado por `ensure_float64()`. float32 introduce errores acumulativos sutiles.
2. **ddof explicito siempre** — Bollinger y VaR usan `ddof=0` (poblacional). Sharpe y modelos de volatilidad usan `ddof=1` (muestral).
3. **Wilder vs EMA centralizado** — La distincion esta en `core/smoothing.py`. RSI, ATR, ADX usan Wilder (alpha=1/N). MACD, EMA usan estandar (alpha=2/(N+1)).
4. **Manejo de NaN** — `validate_ohlcv` rechaza NaN en columnas de precio. Cada indicador documenta su comportamiento de NaN en la salida.
5. **Indicadores acumulativos** (OBV, ADL, VWAP) acumulan drift de punto flotante — verificar periodicamente contra valores conocidos.

### Bloque de imports esencial

```python
import numpy as np
import pandas as pd
from trend.sma import sma
from trend.ema import ema
from trend.rsi import rsi
from trend.macd import macd
from volatility.bollinger import bollinger_bands
from volatility.atr import atr
from volatility.adx import adx
from risk.sharpe import sharpe_ratio
from risk.sortino import sortino_ratio
from risk.drawdown import max_drawdown
from risk.var import var_historical, var_parametric, var_monte_carlo
from core.types import MACDResult, BollingerResult, ADXResult

# Opcionales (guarded con ImportError):
# from stats.timeseries import fit_arima, fit_garch, auto_arima
# from stats.correlation import covariance_shrinkage
# from quant.black_scholes import bs_price, bs_greeks, implied_vol
# from systems.backtest import backtest_signals
```

---

## Conclusion

Tres decisiones arquitectonicas definen el modulo. Primero, **frozen dataclasses como tipos de retorno** para indicadores multi-output — habilitan type safety e IDE support sin perder la expresividad de Series. Segundo, **validacion y smoothing centralizados en `core/`** — eliminan duplicacion y garantizan consistencia entre los 36 modulos. Tercero, **aislamiento de dependencias opcionales** con ImportError guards — el core funciona con solo numpy/pandas/scipy, mientras que ARIMA, GARCH, Ledoit-Wolf y auto_arima se activan al instalar sus librerias.

**Wilder smoothing (alpha = 1/N) vs EMA estandar (alpha = 2/(N+1))** sigue siendo la distincion numerica mas importante — confundir ambos es el error de implementacion mas comun en analisis tecnico. La centralizacion en `core/smoothing.py` previene este error.

La estrategia de **zero dependencias externas para trading systems** (motor de backtest propio en lugar de vectorbt/backtrader) prioriza control total sobre la ejecucion, a costa de funcionalidades avanzadas como live trading o parameter sweeps masivos. Half-Kelly con cap del 25% (`position_size()`) y costos de transaccion de **5-15 bps round-trip** para US equities liquidos son los defaults de produccion.
