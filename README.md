# pySPOQ (Python SPOQ toolbox port)

Short project note:

- Core SPOQ penalty + gradient + metric implemented
- Prox / inner solver implemented
- Warm start (`pds`) implemented
- Full outer solver modes (including trust-region mode) implemented
- Simulated toolbox workflow implemented
- Local Streamlit app implemented (`webapp/`)

Validation/proof is summarized here:

- `docs/VALIDATION_SUMMARY.md`

Run the app locally:

```bash
python3 -m streamlit run webapp/app.py
```
