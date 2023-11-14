from typing import List, Tuple
import numpy as np


class PcSaftParameters:
    @staticmethod
    def from_json(
        parameters: List[str],
        pure_path: str,
        binary_path: str = None,
        identifier_option: IdentifierOption = IdentifierOption.Name
    ) -> 'PcSaftParameters':
        """
        Creates parameters from json files.

        Parameters
        ----------
        substances : List[str]
            The substances to search.
        pure_path : str
            Path to file containing pure substance parameters.
        binary_path : str, optional
            Path to file containing binary substance parameters.
        identifier_option : IdentifierOption, optional, defaults to IdentifierOption.Name
            Identifier that is used to search substance.
        """
    @staticmethod
    def from_multiple_json(
        input: List[Tuple[List[str], str]],
        binary_path: str = None,
        identifier_option: IdentifierOption = IdentifierOption.Name
    ) -> 'PcSaftParameters':
        """
        Creates parameters from json files.

        Parameters
        ----------
        input : List[Tuple[List[str], str]]
            The substances to search and their respective parameter files.
            E.g. [(["methane", "propane"], "parameters/alkanes.json"), (["methanol"], "parameters/alcohols.json")]
        binary_path : str, optional
            Path to file containing binary substance parameters.
        identifier_option : IdentifierOption, optional, defaults to IdentifierOption.Name
            Identifier that is used to search substance.
        """
    @property
    def pure_records(self) -> List[PureRecord]: ...
    @property
    def binary_records(self) -> Option[List[np.ndarray]]: ...
