# lab_uw/data_io/converters/tsv_to_hdf5.py
from pathlib import Path
from lab_uw.data_io.readers.tsv_reader import load_tsv
from lab_uw.data_io.writers.hdf5_writer import write_hdf5

def convert_tsv_to_hdf5(
    tsv_path: Path,
    out_h5: Path,
    run_name: str | None = None,
    *,
    dataset: str = "active",
    channel_count: int = 1,
    add_timestamps: bool = False,
) -> None:
    frame = load_tsv(tsv_path)
    if run_name is None:
        run_name = tsv_path.stem
    write_hdf5(
        frame=frame,
        out_path=out_h5,
        run_name=run_name,
        channel_count=channel_count,
        dataset=dataset,
        add_timestamps=add_timestamps,
    )
