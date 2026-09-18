# CLAUDE.md — NFL line model
Sibling of /docker/nrl-line-prediction-model. Same principles: no leakage, respect
the market baseline, beating the line != profit, walk-forward evaluation.
- `Date` is the US-Eastern game date (matches the workbook); `Kickoff` is AU local.
- Home handicap `*_point` is the book's number (-3.5 = home favoured); `bookie_margin`
  = -point (a margin). `au_vs_sharp` > 0 = sharps like home more than the AU book.
- ESPN endpoints 403 a browser User-Agent from this host; keep the plain UA.
- Outs: QB starter -> backup_qb flag (learned coef); others -> points (stacked, capped).
- SQLite `data/nfl.db` is canonical for outs/bets/snapshots; the xlsx re-seeds `matches` on change.
