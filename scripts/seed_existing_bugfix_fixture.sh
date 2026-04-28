#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE="${1:-$PROJECT_DIR/workspaces/existing-bugfix-demo}"

rm -rf "$WORKSPACE"
mkdir -p "$WORKSPACE/invoice_calc" "$WORKSPACE/tests"

cat > "$WORKSPACE/invoice_calc/__init__.py" <<'PY'
"""Small invoice calculator package used by the agentic bug-fix demo."""

from .calculator import calculate_invoice

__all__ = ["calculate_invoice"]
PY

cat > "$WORKSPACE/tests/__init__.py" <<'PY'
"""Test package for unittest discovery."""
PY

cat > "$WORKSPACE/invoice_calc/calculator.py" <<'PY'
from __future__ import annotations

from .discounts import volume_discount
from .tax import sales_tax


def calculate_invoice(items: list[dict[str, float]], tax_rate: float) -> float:
    """Return final invoice total after volume discount and sales tax.

    `tax_rate` is a decimal fraction: 0.2 means 20%.
    """
    subtotal = sum(item["quantity"] * item["unit_price"] for item in items)
    discount = volume_discount(subtotal, len(items))
    taxable = subtotal - discount
    return round(taxable + sales_tax(taxable, tax_rate), 2)
PY

cat > "$WORKSPACE/invoice_calc/discounts.py" <<'PY'
from __future__ import annotations


def volume_discount(subtotal: float, item_count: int) -> float
    """Return a 10% discount when an invoice contains at least ten items."""
    if item_count >= 10:
        return round(subtotal * 0.10, 2)
    return 0.0
PY

cat > "$WORKSPACE/invoice_calc/tax.py" <<'PY'
from __future__ import annotations


def sales_tax(amount: float, tax_rate: float) -> float:
    """Return sales tax for a decimal tax rate such as 0.2 for 20%."""
    return round(amount * (tax_rate / 100), 2)
PY

cat > "$WORKSPACE/tests/test_invoice_calc.py" <<'PY'
from __future__ import annotations

import unittest

from invoice_calc import calculate_invoice
from invoice_calc.discounts import volume_discount
from invoice_calc.tax import sales_tax


class InvoiceCalculatorTests(unittest.TestCase):
    def test_small_invoice_has_no_discount_and_decimal_tax(self) -> None:
        items = [
            {"name": "coffee", "quantity": 2, "unit_price": 3.50},
            {"name": "cake", "quantity": 1, "unit_price": 4.00},
        ]
        self.assertEqual(calculate_invoice(items, 0.20), 13.20)

    def test_volume_discount_applies_before_tax(self) -> None:
        items = [{"name": f"item-{idx}", "quantity": 1, "unit_price": 10.00} for idx in range(10)]
        self.assertEqual(volume_discount(100.0, len(items)), 10.0)
        self.assertEqual(calculate_invoice(items, 0.20), 108.00)

    def test_sales_tax_uses_decimal_fraction_not_percent_number(self) -> None:
        self.assertEqual(sales_tax(90.0, 0.20), 18.0)


if __name__ == "__main__":
    unittest.main()
PY

cat > "$WORKSPACE/README.md" <<'MD'
# Invoice Calculator Bug-Fix Fixture

This is a deliberately broken existing project for testing whether the harness
can investigate and repair code instead of creating a new project from scratch.

Run the current test suite:

```bash
python -m unittest discover -v
```

Known intent:

- `tax_rate` is a decimal fraction, so `0.20` means 20%.
- invoices with at least ten line items get a 10% volume discount before tax.
MD

git -C "$WORKSPACE" init >/dev/null
git -C "$WORKSPACE" config user.name "agenticFeedbackCoding-fixture"
git -C "$WORKSPACE" config user.email "fixture@example.local"
git -C "$WORKSPACE" add -A
git -C "$WORKSPACE" commit -m "fixture: planted invoice calculator bugs" >/dev/null

echo "Seeded existing-project bug-fix fixture at $WORKSPACE"
echo "Expected initial failure: python -m unittest discover -v"
