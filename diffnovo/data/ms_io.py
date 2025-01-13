"""Mass spectrometry file type input/output operations."""
import collections
import csv
import os
import re
from pathlib import Path
from typing import Any, Dict


class MztabWriter:
    """
    Export spectrum identifications to an mzTab file.

    Parameters
    ----------
    filename : str
        The name of the mzTab file.
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.psms = []

    def save(self) -> None:
        """
        Export the spectrum identifications to the mzTab file.
        """
        with open(self.filename, "w") as f:
            writer = csv.writer(f, delimiter="\t", lineterminator=os.linesep)
            writer.writerow(
                [
                    "True_Peptide",
                    "Post_correction_peptide",
                    "Peptide_score",
                ]
            )
            for psm in self.psms:
                writer.writerow(
                    [
                        #"PSM",
                        psm[0],  # true peptide sequence
                        psm[1],  # beam search alg peptide sequence
                        psm[2], # peptide score
                    ]
                )
