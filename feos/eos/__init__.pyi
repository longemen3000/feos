class EquationOfState:
    """Test"""

    @staticmethod
    def pcsaft(
        parameters: PcSaftParameters,
        max_eta: float = 0.5,
        max_iter_cross_assoc: int = 50,
        tol_cross_assoc: float = 1e-10,
        dq_variant: DQVariants = DQVariants.DQ35
    ) -> 'EquationOfState':
        """
        PC-SAFT equation of state.

        Parameters
        ----------
        parameters : PcSaftParameters
            The parameters of the PC-SAFT equation of state to use.
        max_eta : float, optional
            Maximum packing fraction. Defaults to 0.5.
        max_iter_cross_assoc : unsigned integer, optional
            Maximum number of iterations for cross association. Defaults to 50.
        tol_cross_assoc : float
            Tolerance for convergence of cross association. Defaults to 1e-10.
        dq_variant : DQVariants, optional
            Combination rule used in the dipole/quadrupole term. Defaults to 'DQVariants.DQ35'

        Returns
        -------
        EquationOfState
            The PC-SAFT equation of state that can be used to compute thermodynamic
            states.
        """
    @staticmethod
    def gc_pcsaft(
        parameters: GcPcSaftEosParameters,
        max_eta: float = 0.5,
        max_iter_cross_assoc: int = 50,
        tol_cross_assoc: float = 1e-10,
    ) -> 'EquationOfState':
        """
        (heterosegmented) group contribution PC-SAFT equation of state.

        Parameters
        ----------
        parameters : GcPcSaftEosParameters
            The parameters of the PC-SAFT equation of state to use.
        max_eta : float, optional
            Maximum packing fraction. Defaults to 0.5.
        max_iter_cross_assoc : unsigned integer, optional
            Maximum number of iterations for cross association. Defaults to 50.
        tol_cross_assoc : float
            Tolerance for convergence of cross association. Defaults to 1e-10.

        Returns
        -------
        EquationOfState
            The gc-PC-SAFT equation of state that can be used to compute thermodynamic
            states.
        """
