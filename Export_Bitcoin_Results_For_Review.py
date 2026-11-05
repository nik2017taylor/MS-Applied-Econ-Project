# -*- coding: utf-8 -*-
"""Package an existing Bitcoin-control run for review. No download or estimation.

Run in the Spyder session holding crypto_fitted_models to also include the
coefficient covariance matrices. Saved result tables can be packaged after a
kernel restart. The original run directory is read-only.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

# This is the exact run identified by the output you supplied.
RUN_FOLDER = (
    Path.home() / "data_center_study_data" / "LMP_single_file"
    / "outputs" / "bitcoin_controls_v1" / "20260929T030053973839Z"
)

TABLES = (
    "crypto_run_summary_df", "crypto_key_results_df",
    "crypto_specification_comparison_df", "crypto_all_coefficients_df",
    "crypto_control_audit_df", "crypto_centering_df", "crypto_coverage_df",
    "crypto_common_sample_check_df", "crypto_analysis_errors_df",
    "crypto_variance_ratios_df", "crypto_zero_moment_df",
    "crypto_sensitivity_df", "crypto_event_study_df", "crypto_placebo_df",
    "crypto_horizon_effects_df", "crypto_daily_primary_df",
    "crypto_mining_economics_daily_df", "crypto_raw_mean_variance_changes_df",
    "crypto_raw_prepost_summary_df", "crypto_source_status_df",
    "crypto_input_provenance_df",
)


def package_results(run_folder: Path, workspace: dict | None = None) -> dict:
    """Zip whitelisted saved tables; include live covariance matrices if matched."""
    run_folder = Path(run_folder).expanduser().resolve()
    if not run_folder.is_dir():
        raise FileNotFoundError(
            f"Saved run not found: {run_folder}\n"
            "Set RUN_FOLDER to the 'New results saved separately' directory. "
            "Do not rerun the analysis just to make this archive."
        )
    required = ("manifest.json", "crypto_all_coefficients_df.csv.gz",
                "crypto_run_summary_df.csv.gz")
    missing_required = [name for name in required if not (run_folder / name).is_file()]
    if missing_required:
        raise FileNotFoundError(f"Not a complete saved result directory: {missing_required}")

    files = [run_folder / (name + ".csv.gz") for name in TABLES]
    missing = [p.name for p in files if not p.is_file()]
    files = [p for p in files if p.is_file()] + [run_folder / "manifest.json"]
    workspace = {} if workspace is None else workspace
    live_folder = workspace.get("crypto_results_folder")
    same_run = bool(live_folder) and Path(str(live_folder)).expanduser().resolve() == run_folder
    fitted = workspace.get("crypto_fitted_models") if same_run else None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive = run_folder.parent / f"Bitcoin_Results_For_Review_{run_folder.name}_{stamp}.zip"
    report = {
        "source_run": run_folder.name,
        "saved_files_included": [p.name for p in files],
        "missing_optional_files": missing,
        "covariance_matrices_included": 0,
        "covariance_errors": [],
        "live_workspace_matched_saved_run": same_run,
        "hourly_panel_included": False,
        "no_models_rerun": True,
        "no_network_requests": True,
        "source_files_modified": False,
    }
    try:
        with ZipFile(archive, "x", compression=ZIP_DEFLATED, compresslevel=6) as bundle:
            for path in files:
                bundle.write(path, arcname=path.name)
            if isinstance(fitted, dict):
                import pandas as pd
                for key, models in fitted.items():
                    if not isinstance(key, tuple) or len(key) != 3 or not isinstance(models, dict):
                        continue
                    site, node, specification = key
                    if specification not in ("hashprice_interaction", "btc_price_interaction"):
                        continue
                    for outcome, model in models.items():
                        try:
                            names = list(model.model.exog_names)
                            cov = pd.DataFrame(model.cov_params(), index=names, columns=names)
                            stem = "__".join(str(x) for x in (site, node, specification, outcome))
                            stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
                            bundle.writestr(
                                f"coefficient_covariances/{stem}.csv",
                                cov.to_csv(index_label="term", float_format="%.17g"),
                            )
                            report["covariance_matrices_included"] += 1
                        except Exception as error:
                            report["covariance_errors"].append(
                                {"model": repr((key, outcome)), "error_type": type(error).__name__}
                            )
            bundle.writestr("REVIEW_EXPORT_INVENTORY.json", json.dumps(report, indent=2))
        with ZipFile(archive, "r") as bundle:
            bad = bundle.testzip()
            if bad:
                raise OSError(f"Archive verification failed at {bad}")
    except Exception:
        if archive.exists():
            archive.unlink()
        raise
    report["archive_path"] = str(archive)
    report["archive_mib"] = archive.stat().st_size / (1024 ** 2)
    return report


if __name__ == "__main__":
    crypto_review_export = package_results(RUN_FOLDER, globals())
    print("\nRESULTS PACKAGED -- no analysis was rerun")
    print("Upload this ZIP:\n" + crypto_review_export["archive_path"])
    print(f"Size: {crypto_review_export['archive_mib']:.2f} MiB")
    print("Covariance matrices:", crypto_review_export["covariance_matrices_included"])
    if not crypto_review_export["covariance_matrices_included"]:
        print("Saved tables are included. Live covariance matrices were unavailable "
              "or belonged to a different run; do not rerun just for this export.")
    if crypto_review_export["missing_optional_files"]:
        print("Missing optional files:", crypto_review_export["missing_optional_files"])
    print("Original inputs, credentials, fitted-model pickles, and the large hourly panel "
          "were not selected for this archive. Local paths may appear in provenance tables.")
