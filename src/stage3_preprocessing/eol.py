from pathlib import Path
import pandas as pd


def create_eol_anchors(output_csv: Path):

    data = [
        {
            "experiment_id": "strawberry_experiment",
            "fruit_id": "F01",
            "eol_timestamp": "2026-03-26 08:00:00",
            "eol_basis": "visual",
            "proposed_by": "cong",
            "reviewed_by": "",
            "approved_by": "",
            "status": "approved",
            "notes": "End of life"
        },
        {
            "experiment_id": "strawberry_experiment",
            "fruit_id": "F02",
            "eol_timestamp": "2026-03-28 08:00:00",
            "eol_basis": "visual",
            "proposed_by": "cong",
            "reviewed_by": "",
            "approved_by": "",
            "status": "approved",
            "notes": "End of life"
        },
        {
            "experiment_id": "strawberry_experiment",
            "fruit_id": "F03",
            "eol_timestamp": "2026-03-26 08:00:00",
            "eol_basis": "visual",
            "proposed_by": "cong",
            "reviewed_by": "",
            "approved_by": "",
            "status": "approved",
            "notes": "End of life"
        },
        {
            "experiment_id": "strawberry_experiment",
            "fruit_id": "F04",
            "eol_timestamp": "2026-03-26 08:00:00",
            "eol_basis": "visual",
            "proposed_by": "cong",
            "reviewed_by": "",
            "approved_by": "",
            "status": "approved",
            "notes": "End of life"
        },
        {
            "experiment_id": "strawberry_experiment",
            "fruit_id": "F05",
            "eol_timestamp": "2026-03-28 08:00:00",
            "eol_basis": "visual",
            "proposed_by": "cong",
            "reviewed_by": "",
            "approved_by": "",
            "status": "approved",
            "notes": "End of life"
        },
        {
            "experiment_id": "strawberry_experiment",
            "fruit_id": "F06",
            "eol_timestamp": "2026-03-26 08:00:00",
            "eol_basis": "visual",
            "proposed_by": "cong",
            "reviewed_by": "",
            "approved_by": "",
            "status": "approved",
            "notes": "End of life"
        }
    ]

    df = pd.DataFrame(data)

    output_csv.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        output_csv,
        index=False
    )

    print(f"Saved: {output_csv}")
    print(df)


def main():

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    output_csv = (
        PROJECT_ROOT
        / "data"
        / "02_processed"
        / "manifests"
        / "eol_anchors.csv"
    )

    create_eol_anchors(output_csv)


if __name__ == "__main__":
    main()