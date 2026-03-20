import re
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem.rdMolDescriptors import CalcMolFormula as rdCalcMolFormula

from synkit.IO.debug import setup_logging


logger = setup_logging("INFO")


def fix_phospho_charge(smiles: str, use_atom_map: bool = False) -> str:
    """
    Attempt to parse a SMILES string with possibly over-charged phosphorus,
    using one of two patterns:

      - Atom-map aware (`use_atom_map=True`): matches `[P+N:map]`, `[P:map]`, etc.
      - Normal (`use_atom_map=False`): matches `[P+N]`, `[P+]`, etc., ignoring any maps.

    Steps:
      1. Optionally strip or preserve `:map` labels according to `use_atom_map`.
      2. Try RDKit parsing as-is.
      3. On failure, repeatedly decrement the first matched P-charge token by one:
         - `[P+N]→[P+N-1]` (or drop the `+` if it hits zero)
         - (and preserve `:map` if `use_atom_map=True`)
      4. Return the first version that parses, or the final string if none parse.

    :param smiles: SMILES string with possible P charges.
    :param use_atom_map: If True, use the atom-map-aware pattern; otherwise ignore maps.
    :returns: A valid-parsing SMILES or the last attempt.
    """
    # Choose pattern based on use_atom_map
    if use_atom_map:
        # group(1)=digits, group(2)=:map
        pattern = re.compile(r"\[P\+?(\d*)?(:\d+)\]")
    else:
        # group(1)=digits (no map)
        pattern = re.compile(r"\[P\+?(\d*)?\]")

    def _decrement_once(smi: str) -> (str, bool):
        def _repl(m: re.Match) -> str:
            charge_str = m.group(1) or ""
            map_str = m.group(2) or ""  # will be '' when use_atom_map=False
            current = int(charge_str) if charge_str.isdigit() else 1
            new_charge = current - 1
            if new_charge <= 0:
                return f"[P{map_str}]"
            return f"[P+{new_charge}{map_str}]"

        new_smi, n = pattern.subn(_repl, smi, count=1)
        return new_smi, n > 0

    # Quick parse attempt
    if Chem.MolFromSmiles(smiles) is not None:
        return smiles

    modified = smiles
    while True:
        modified, changed = _decrement_once(modified)
        if not changed:
            break
        if Chem.MolFromSmiles(modified) is not None:
            logger.warning(
                "fix_phospho_charge: parsed after decrementing → %s", modified
            )
            return modified

    logger.error("fix_phospho_charge: unable to parse after adjustments: %s", smiles)
    return modified


def calc_mol_formula(mol):
    """
    Calculate the molecular formula of an RDKit molecule.

    :param mol: The molecule, provided as an RDKit Mol object or a SMILES string.
    :type mol: rdkit.Chem.Mol or str
    :return: The molecular formula.
    :rtype: str
    :raises ValueError: If mol is None or the SMILES string is invalid.
    :raises TypeError: If mol is of an unsupported type.
    """
    if mol is None:
        raise ValueError("Input molecule is None")

    # Convert SMILES string to RDKit Mol
    if isinstance(mol, str):
        mol_obj = Chem.MolFromSmiles(mol)
        if mol_obj is None:
            raise ValueError(f"Invalid SMILES string: {mol}")
    elif isinstance(mol, Mol):
        mol_obj = mol
    else:
        raise TypeError(f"Unsupported type for mol: {type(mol)}")

    # Compute formula and return
    return rdCalcMolFormula(mol_obj)