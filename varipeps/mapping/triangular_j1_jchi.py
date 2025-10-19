import collections.abc
from dataclasses import dataclass
from functools import partial, reduce
from os import PathLike

import jax.numpy as jnp
import numpy as np
from jax import jit
import jax.util
import h5py
import jax

from varipeps import varipeps_config
import varipeps.config
from varipeps.peps import PEPS_Tensor, PEPS_Unit_Cell
from varipeps.contractions import apply_contraction, apply_contraction_jitted
from varipeps.expectation.model import Expectation_Model
from varipeps.expectation.three_sites import (
    calc_three_sites_triangle_without_bottom_right_multiple_gates,
    calc_three_sites_triangle_without_top_left_multiple_gates,
    calc_three_sites_triangle_without_bottom_left_multiple_gates,
    calc_three_sites_triangle_without_top_right_multiple_gates,
)
from varipeps.expectation.spiral_helpers import apply_unitary
from varipeps.typing import Tensor
from varipeps.utils.random import PEPS_Random_Number_Generator
from varipeps.mapping import Map_To_PEPS_Model

from varipeps.utils.debug_print import debug_print

from typing import (
    Sequence,
    Union,
    List,
    Callable,
    TypeVar,
    Optional,
    Tuple,
    Type,
    Dict,
    Any,
)



@dataclass
class Triangular_j1_jchi_model(Expectation_Model):
    r"""
    Expectation model for the triangular-lattice J1-Jchi Hamiltonian.
        --> y
       | +---+
       v |\  | < down triangle
       x | \ |
         +---+
            ^ up triangle

    Args:
    down_gates : Sequence[jnp.ndarray]
        Sequence of gates to be applied on down-pointing triangles
        The gate is applied in the order [top-left, top-right, bottom-right].
    
    up_gates : Sequence[jnp.ndarray]
        Sequence of gates to be applied on up-pointing triangles
        The gate is applied in the order [top-left, bottom-left, bottom-right].
    j1 : float
        Heisenberg exchange coupling.
    jchi : float
        Chiral three-spin coupling. In the provided constructor (from_j1_jchi)
        the sign of the chiral term is opposite for up- and down-pointing triangles.
    normalization_factor : int, optional
        Factor used to normalize the returned expectation values (e.g., when
        multiple physical sites are mapped to a single PEPS site). Default is 1.
    is_spiral_peps : bool, optional
        If True, the model treats tensors as spiral iPEPS and requires a
        spiral_unitary_operator to construct the spiral rotations.
    spiral_unitary_operator : Optional[jnp.ndarray], optional
        Operator used to generate the spiral rotations when is_spiral_peps is True.


    - The returned dtype will be real when all gates are Hermitian and complex
      otherwise.
    """

    up_gates: Sequence[jnp.ndarray]
    down_gates: Sequence[jnp.ndarray]
    j1 : float
    jchi : float
    normalization_factor: int = 1
    is_spiral_peps: bool = False
    real_d: int = 2
    spiral_unitary_operator: Optional[jnp.ndarray] = None

    def __post_init__(self) -> None:
        if isinstance(self.up_gates, jnp.ndarray):
            self.up_gates = (self.up_gates,)
        else:
            self.up_gates = tuple(self.up_gates)

        if isinstance(self.down_gates, jnp.ndarray):
            self.down_gates = (self.down_gates,)
        else:
            self.down_gates = tuple(self.down_gates)

        self._result_type = (
            jnp.float64
            if all(
                jnp.allclose(g, g.T.conj())
                for g in self.up_gates
                + self.down_gates
            )
            else jnp.complex128
        )

        if self.is_spiral_peps:
            self._spiral_D, self._spiral_sigma = jnp.linalg.eigh(
                self.spiral_unitary_operator
            )

    def __call__(
        self,
        peps_tensors: Sequence[jnp.ndarray],
        unitcell: PEPS_Unit_Cell,
        spiral_vectors: Optional[Union[jnp.ndarray, Sequence[jnp.ndarray]]] = None,
        *,
        normalize_by_size: bool = True,
        only_unique: bool = True,
        return_single_gate_results: bool = False,
    ) -> Union[jnp.ndarray, List[jnp.ndarray]]:
        result = [
            jnp.array(0, dtype=self._result_type)
            for _ in range(len(self.up_gates))
        ]

        if self.is_spiral_peps:
            if spiral_vectors is None:
                raise ValueError(
                    "When using spiral iPEPS, spiral_vectors must be provided."
                )
            # if not isinstance(spiral_vectors, collections.abc.Sequence):
            #     spiral_vectors = (spiral_vectors,) * 3


            #[top-left, top-right, bottom-right]
            working_down_gates = tuple(
                apply_unitary(
                    h,
                    tuple(jnp.array(ri) for ri in ((0,0), (0,1), (1,1))),
                    spiral_vectors,
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    3,
                    (0, 1, 2),
                    varipeps_config.spiral_wavevector_type,
                )
                for h in self.up_gates
            )
            #[top-left, bottom-left, bottom-right]
            working_up_gates = tuple(
                apply_unitary(
                    h,
                    tuple(jnp.array(ri) for ri in ((0,0), (1,0), (1,1))),
                    spiral_vectors,
                    self._spiral_D,
                    self._spiral_sigma,
                    self.real_d,
                    3,
                    (0, 1, 2),
                    varipeps_config.spiral_wavevector_type,
                )
                for h in self.down_gates
            )
        else:
            working_up_gates = self.up_gates
            working_down_gates = self.down_gates

        for x, iter_rows in unitcell.iter_all_rows(only_unique=only_unique):
            for y, view in iter_rows:
                # Get all 4 tensors in the 2x2 view
                tensors_i = view.get_indices(
                    (slice(0, 2, None), slice(0, 2, None))
                )
                tensors = [
                    peps_tensors[i] for j in tensors_i for i in j
                ]
                tensor_objs = [t for tl in view[:2, :2] for t in tl]

                # remat/checkpoint for memory; gates treated static via static_argnums
                step_result_down = (
                    calc_three_sites_triangle_without_bottom_left_multiple_gates(
                        tensors,
                        tensor_objs,
                        working_down_gates,
                    )
                )

                step_result_up = (
                    calc_three_sites_triangle_without_top_right_multiple_gates(
                        tensors,
                        tensor_objs,
                        working_up_gates,
                    )
                )

                for sr_i, (sr_down, sr_up) in enumerate(
                    zip(step_result_down, step_result_up, strict=True)
                ):
                    result[sr_i] += sr_down + sr_up

        if normalize_by_size:
            size = unitcell.get_len_unique_tensors() if only_unique else (unitcell.get_size()[0] * unitcell.get_size()[1])
            size = size * self.normalization_factor
            result = [r / size for r in result]

        if len(result) == 1:
            return result[0]
        else:
            return result

    @classmethod
    def from_j1_jchi(
        cls,
        j1: float,
        jchi: float,
        p: int = 2,
        normalization_factor: int = 1,
        is_spiral_peps: bool = False,
        spiral_unitary_operator: Optional[jnp.ndarray] = None,
    ):
        """
        Alternative constructor to build the Hamiltonian gates from J1 and Jchi values.
        """
        Id = jnp.eye(p)
        if p == 2:
            Sx = jnp.array([[0, 1], [1, 0]]) / 2
            Sy = jnp.array([[0, -1j], [1j, 0]]) / 2
            Sz = jnp.array([[1, 0], [0, -1]]) / 2
        else:
            raise NotImplementedError(f"Spin matrices for physical dimension p={p} not defined.")
        
        Svec = jnp.array([Sx, Sy, Sz])

        levicivit3 = jnp.zeros((3, 3, 3))
        levicivit3 = levicivit3.at[0, 1, 2].set(1).at[1, 2, 0].set(1).at[2, 0, 1].set(1)
        levicivit3 = levicivit3.at[0, 2, 1].set(-1).at[2, 1, 0].set(-1).at[1, 0, 2].set(-1)

        def kron(*args):
            return reduce(jnp.kron, args)

        SS_12 = kron(Sx, Sx, Id) + kron(Sy, Sy, Id) + kron(Sz, Sz, Id)
        SS_23 = kron(Id, Sx, Sx) + kron(Id, Sy, Sy) + kron(Id, Sz, Sz)
        SS_31 = kron(Sx, Id, Sx) + kron(Sy, Id, Sy) + kron(Sz, Id, Sz)

        SSS_chi = jnp.einsum('abc,aij,bkl,cmn->ikmjln', levicivit3, Svec, Svec, Svec).reshape(p**3, p**3)

        H_J1 = 1/2 * j1 * (SS_12 + SS_23 + SS_31)

        trgl_down = (H_J1 + jchi * SSS_chi)

        trgl_up = H_J1 + jchi * SSS_chi

        return cls(
            up_gates=(trgl_up,),
            down_gates=(trgl_down,),
            normalization_factor=normalization_factor,
            is_spiral_peps=is_spiral_peps,
            spiral_unitary_operator=spiral_unitary_operator,
            j1=j1,
            jchi=jchi,
            real_d=p,
        )

    def save_to_group(self, grp: h5py.Group):
        cls = type(self)
        grp.attrs["class"] = f"{cls.__module__}.{cls.__qualname__}"

        grp_gates = grp.create_group("gates", track_order=True)
        # Save up and down gates consistently with attributes
        grp_gates.attrs["len_up"] = len(self.up_gates)
        for i, g in enumerate(self.up_gates):
            grp_gates.create_dataset(
                f"up_gate_{i:d}", data=g, compression="gzip", compression_opts=6
            )

        grp_gates.attrs["len_down"] = len(self.down_gates)
        for i, g in enumerate(self.down_gates):
            grp_gates.create_dataset(
                f"down_gate_{i:d}", data=g, compression="gzip", compression_opts=6
            )

        grp.attrs["normalization_factor"] = self.normalization_factor
        grp.attrs["is_spiral_peps"] = self.is_spiral_peps
        grp.attrs["j1"] = self.j1
        grp.attrs["jchi"] = self.jchi

        if self.is_spiral_peps:
            grp.create_dataset(
                "spiral_unitary_operator",
                data=self.spiral_unitary_operator,
                compression="gzip",
                compression_opts=6,
            )

    @classmethod
    def load_from_group(cls, grp: h5py.Group):
        if not grp.attrs["class"] == f"{cls.__module__}.{cls.__qualname__}":
            raise ValueError(
                "The HDF5 group suggests that this is not the right class to load data from it."
            )

        # Prefer new names; fall back to old ones if present
        if "len_up" in grp["gates"].attrs:
            up_gates = tuple(
                jnp.asarray(grp["gates"][f"up_gate_{i:d}"])
                for i in range(grp["gates"].attrs["len_up"])
            )
            down_gates = tuple(
                jnp.asarray(grp["gates"][f"down_gate_{i:d}"])
                for i in range(grp["gates"].attrs["len_down"])
            )
        else:
            # Backward compatibility with previous naming (if any)
            up_gates = tuple(
                jnp.asarray(grp["gates"][f"triangle_without_top_left_gate_{i:d}"])
                for i in range(grp["gates"].attrs.get("len_top_left", 0))
            )
            down_gates = tuple(
                jnp.asarray(grp["gates"][f"triangle_without_bottom_right_gate_{i:d}"])
                for i in range(grp["gates"].attrs.get("len_bottom_right", 0))
            )

        is_spiral_peps = grp.attrs["is_spiral_peps"]
        spiral_unitary_operator = (
            jnp.asarray(grp["spiral_unitary_operator"])
            if is_spiral_peps
            else None
        )

        return cls(
            up_gates=up_gates,
            down_gates=down_gates,
            normalization_factor=grp.attrs["normalization_factor"],
            is_spiral_peps=is_spiral_peps,
            spiral_unitary_operator=spiral_unitary_operator,
            j1=grp.attrs["j1"],
            jchi=grp.attrs["jchi"],
        )