---
name: Panel finance validator
description: Checa consistência do painel IBRA (Economatica → ffill → indicadores) com definições de finanças e mercado; roda validate_panel e interpreta resultados.
---

# Quando usar

- O usuário pedir para **validar contas**, **conferir se os números batem**, **auditar o pipeline** ou **verificar indicadores** do projeto Quant (painel B3 / IBRA).
- Após mudanças em `src/indicators/`, `src/derived.py`, `src/fundamentals_ffill.py` ou `src/io_excel.py`.

# O que fazer

1. Executar a partir da raiz do projeto `Quant` (rotina padrão após cada `build_panel.py`):
   ```bash
   python scripts/validate_panel.py
   ```
   Para foco em ativos específicos (recomendado para debug):
   ```bash
   python scripts/validate_panel.py --tickers VALE3,PETR4
   ```
   Para checagem rápida de identidades contábeis em amostra:
   ```bash
   python scripts/validate_panel.py --sample 80000
   ```
   Exit code **0** = todos os checks passaram; **1** = inconsistência; **2** = Parquet ausente.
   Observação: com `--sample`, o script usa amostra para checks de identidade e série completa para `r_daily_close` (evita falso positivo por quebra de sequência temporal).

2. Ler a lógica dos checks em `src/finance_validation.py` para explicar ao usuário **o que** cada teste garante (identidades contábeis, P/L vs níveis, EV/EBITDA×EBITDA, retorno vs preço, etc.).

3. Se algo falhar:
   - Comparar com `indicator_definitions.csv` e com as fórmulas em `src/indicators/compute_indicators.py`.
   - Conferir se a falha é de definição da fonte (Economatica pode usar janelas/reporting diferentes).
   - Distinguir erro real de dado vs tolerância numérica (`rtol`/`atol`) antes de propor mudanças.

4. Excel de inspeção humana:
   - Pós-ffill (sem indicadores agregados): `python scripts/export_ffill_intermediate_excel.py`
   - Indicadores (55 colunas) + raw: `python scripts/export_tickers_excel.py` (aba `indicators`)

# Escopo

- Não substitui auditoria contábil nem validação legal dos dados da Economatica.
- Checks são **sanity** alinhados a finanças corporativas e mercado de capitais (razões, identidades EV/EBIT, retornos simples).
