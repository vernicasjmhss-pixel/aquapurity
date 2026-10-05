"""
constraints.py
--------------
Stage 3b: Exclusion screening for candidate recharge sites.

Simple rule-based filter applied BEFORE TOPSIS ranking.
Each rule is documented so it can be explained to reviewers.

Flags (from sites.csv):
  constraint_protected  : 1 if site falls in a forest / protected area
  constraint_built_up   : 1 if site is in a settlement / urban zone
  constraint_saline     : 1 if site has saline groundwater risk

These flags are set synthetically in generate_data.py; in a real deployment
they would come from Bhuvan / CGWB spatial layers.
"""

import os
import sys

import pandas as pd

# Force UTF-8 stdout: prints use an arrow character that cp1252 consoles reject.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# Each rule returns True if the site SHOULD BE EXCLUDED
EXCLUSION_RULES = {
    "Protected area":
        # Rule: Recharge structures cannot be built in forests or wildlife sanctuaries.
        # Legal basis: Forest Conservation Act 1980 and Wildlife Protection Act 1972.
        lambda row: row["constraint_protected"] == 1,

    "Built-up / Settlement":
        # Rule: Urban or dense settlement areas are unsuitable due to land acquisition
        # difficulty and surface-sealing that prevents infiltration.
        lambda row: row["constraint_built_up"] == 1,

    "Saline groundwater risk":
        # Rule: Sites where the aquifer is known / suspected to be saline are excluded.
        # Injecting water into saline zones can worsen quality in adjacent wells.
        lambda row: row["constraint_saline"] == 1,
}


def screen_sites(sites: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Apply all exclusion rules and add an 'is_constrained' column.
    Sites that fail ANY rule are marked constrained (eligible=False).

    Parameters
    ----------
    sites   : DataFrame with constraint_* columns
    verbose : print a summary

    Returns
    -------
    sites with additional columns:
      is_constrained     : bool (True = excluded)
      constraint_reason  : comma-separated list of triggered rules
    """
    reasons = [""] * len(sites)
    constrained = [False] * len(sites)

    for rule_name, rule_fn in EXCLUSION_RULES.items():
        mask = sites.apply(rule_fn, axis=1)
        for i, flag in enumerate(mask):
            if flag:
                constrained[i] = True
                reasons[i] = (reasons[i] + ", " + rule_name).lstrip(", ")

    sites = sites.copy()
    sites["is_constrained"] = constrained
    sites["constraint_reason"] = reasons

    if verbose:
        n_total = len(sites)
        n_const = sum(constrained)
        n_elig  = n_total - n_const
        print(f"[Constraints] {n_total} sites → "
              f"{n_elig} eligible, {n_const} constrained")
        if n_const > 0:
            # count by rule
            for rule_name in EXCLUSION_RULES:
                count = sites["constraint_reason"].str.contains(rule_name, regex=False).sum()
                print(f"  {rule_name}: {count} sites")

    return sites


# ── CLI demo ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    sites = pd.read_csv(os.path.join(DATA_DIR, "sites.csv"))
    screened = screen_sites(sites)
    print(screened[["site_id", "district", "is_constrained", "constraint_reason"]].head(20))
