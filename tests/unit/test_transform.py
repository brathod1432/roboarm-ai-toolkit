"""Unit tests for homogeneous transformation utilities."""

from __future__ import annotations

import math

import numpy as np
import pytest

from roboarm.core.transform import (
    chain_transforms,
    dh_transform,
    extract_position,
    inverse_transform,
    is_valid_transform,
    mdh_transform,
)


class TestDHTransform:
    """Tests for the standard DH transform builder."""

    def test_identity_at_zero(self):
        """DH transform with a=0, d=0, theta=0, alpha=0 should be identity."""
        T = dh_transform(alpha=0.0, a=0.0, theta=0.0, d=0.0)
        np.testing.assert_array_almost_equal(T, np.eye(4))

    def test_pure_translation(self):
        """DH transform with only link length a should translate along x."""
        T = dh_transform(alpha=0.0, a=1.5, theta=0.0, d=0.0)
        assert T[0, 3] == pytest.approx(1.5)
        assert T[1, 3] == pytest.approx(0.0)
        assert T[2, 3] == pytest.approx(0.0)

    def test_rotation_90_deg(self):
        """DH transform with theta=pi/2 should rotate about Z."""
        T = dh_transform(alpha=0.0, a=1.0, theta=math.pi / 2, d=0.0)
        # After 90-deg rotation about Z, a=1.0 => translation in Y
        assert T[0, 3] == pytest.approx(0.0, abs=1e-10)
        assert T[1, 3] == pytest.approx(1.0)
        assert is_valid_transform(T)

    def test_d_offset(self):
        """DH transform with d offset should translate along Z."""
        T = dh_transform(alpha=0.0, a=0.0, theta=0.0, d=2.0)
        assert T[2, 3] == pytest.approx(2.0)

    def test_valid_se3(self):
        """All DH transforms should be valid SE(3) matrices."""
        T = dh_transform(
            alpha=math.pi / 4, a=0.8, theta=math.pi / 6, d=0.3,
        )
        assert is_valid_transform(T)


class TestMDHTransform:
    """Tests for the modified DH (Craig) transform builder."""

    def test_identity_at_zero(self):
        """MDH transform with all-zero parameters should be identity."""
        T = mdh_transform(alpha=0.0, a=0.0, theta=0.0, d=0.0)
        np.testing.assert_array_almost_equal(T, np.eye(4))

    def test_valid_se3(self):
        """MDH transforms should always be valid SE(3) matrices."""
        T = mdh_transform(
            alpha=math.pi / 2, a=0.5, theta=math.pi / 3, d=0.2,
        )
        assert is_valid_transform(T)

    def test_pure_a_translation(self):
        """MDH with only a should translate along x."""
        T = mdh_transform(alpha=0.0, a=1.0, theta=0.0, d=0.0)
        assert T[0, 3] == pytest.approx(1.0)
        assert T[1, 3] == pytest.approx(0.0)
        assert T[2, 3] == pytest.approx(0.0)


class TestChainTransforms:
    """Tests for chaining multiple transforms."""

    def test_single_transform(self):
        """Chaining a single transform should return that transform."""
        T = dh_transform(alpha=0.0, a=1.0, theta=0.0, d=0.0)
        result = chain_transforms([T])
        np.testing.assert_array_almost_equal(result, T)

    def test_two_translations(self):
        """Chaining two pure translations should add their offsets."""
        T1 = dh_transform(alpha=0.0, a=1.0, theta=0.0, d=0.0)
        T2 = dh_transform(alpha=0.0, a=0.5, theta=0.0, d=0.0)
        result = chain_transforms([T1, T2])
        assert result[0, 3] == pytest.approx(1.5)

    def test_empty_chain_is_identity(self):
        """Chaining zero transforms should return identity."""
        result = chain_transforms([])
        np.testing.assert_array_almost_equal(result, np.eye(4))


class TestInverseTransform:
    """Tests for efficient SE(3) inverse."""

    def test_inverse_of_identity(self):
        """Inverse of identity should be identity."""
        T = np.eye(4)
        T_inv = inverse_transform(T)
        np.testing.assert_array_almost_equal(T_inv, np.eye(4))

    def test_inverse_roundtrip(self):
        """T @ inv(T) should equal identity."""
        T = dh_transform(
            alpha=math.pi / 6, a=1.2, theta=math.pi / 4, d=0.5,
        )
        T_inv = inverse_transform(T)
        product = T @ T_inv
        np.testing.assert_array_almost_equal(product, np.eye(4), decimal=10)

    def test_inverse_roundtrip_complex(self):
        """Verify inverse roundtrip with more complex parameters."""
        T = dh_transform(
            alpha=-math.pi / 3, a=2.5, theta=-math.pi / 5, d=1.0,
        )
        T_inv = inverse_transform(T)
        np.testing.assert_array_almost_equal(
            T_inv @ T, np.eye(4), decimal=10,
        )


class TestExtractPosition:
    """Tests for position extraction from transforms."""

    def test_extract_from_identity(self):
        """Position from identity transform should be the origin."""
        pos = extract_position(np.eye(4))
        np.testing.assert_array_almost_equal(pos, [0.0, 0.0, 0.0])

    def test_extract_from_dh(self):
        """Position should match the last column of the transform."""
        T = dh_transform(alpha=0.0, a=3.0, theta=0.0, d=1.0)
        pos = extract_position(T)
        assert pos[0] == pytest.approx(3.0)
        assert pos[2] == pytest.approx(1.0)


class TestIsValidTransform:
    """Tests for SE(3) validation."""

    def test_identity_is_valid(self):
        """The 4x4 identity matrix is a valid SE(3) transform."""
        assert is_valid_transform(np.eye(4)) is True

    def test_wrong_shape_is_invalid(self):
        """A 3x3 matrix should not pass SE(3) validation."""
        assert is_valid_transform(np.eye(3)) is False

    def test_non_orthogonal_is_invalid(self):
        """A matrix with non-orthogonal rotation block should fail."""
        T = np.eye(4)
        T[0, 0] = 2.0  # break orthogonality
        assert is_valid_transform(T) is False

    def test_wrong_bottom_row_is_invalid(self):
        """A matrix with incorrect bottom row should fail."""
        T = np.eye(4)
        T[3, 0] = 0.1
        assert is_valid_transform(T) is False
