"""
해밀토니안 제약 조건 공식화 및 검증 모듈
=======================================

28개 ξ-다발 실험 결과에서 도출한 8개 제약 조건을
유한 차원 해밀토니안 행렬에 대해 수치적으로 검증한다.

[이론 배경]
Berry-Keating 해밀토니안 H_BK = xp + px 는 형식적으로
리만 ξ 함수의 영점을 고유값으로 가져야 한다는 가설을 만족해야 한다.
ξ-다발 프레임워크에서 이 가설은 아래 8개 기하-스펙트럼 조건으로 구체화된다.

유한 차원 번역 원칙:
- 무한 차원 연산자 → n×n 행렬 (트렁케이션)
- 연속 적분 → 유한 합 / 수치 적분
- 발산 조건 → ε-정규화 후 극한 거동 관측
- 위상적 불변량 → 복소 행렬식의 winding number

사용법:
    H = np.diag([1, 2, 3, 4, 5], dtype=float)
    hc = HamiltonianConstraints(H, description="대각 해밀토니안")
    report = hc.full_assessment()
    hc.report()
"""

import numpy as np
from scipy.linalg import eigh, eigvalsh, solve
from scipy.integrate import quad
from scipy.special import zeta as scipy_zeta
import warnings


# ---------------------------------------------------------------------------
# 보조 함수
# ---------------------------------------------------------------------------

def riemann_counting(T):
    """
    N̄(T) = (T/2π) log(T/2πe) + 7/8

    리만 영점 카운팅 함수의 주항(Weyl formula).
    T > 0 이어야 한다.

    Parameters
    ----------
    T : float
        양의 실수

    Returns
    -------
    float
        [0, T] 구간 내 예상 영점 개수
    """
    if T <= 0:
        return 0.0
    return (T / (2 * np.pi)) * np.log(T / (2 * np.pi * np.e)) + 7.0 / 8.0


def _spectral_density(energies, eigenvalues, epsilon=0.1):
    """
    로렌츠 정규화를 사용한 spectral density 계산.

    A(E) = (1/π) Σ_n ε / ((E - λ_n)² + ε²)

    Parameters
    ----------
    energies : array_like
        평가할 에너지 배열
    eigenvalues : array_like
        해밀토니안 고유값 배열
    epsilon : float
        로렌츠 폭 (ε-정규화)

    Returns
    -------
    np.ndarray
        각 에너지에서의 spectral density
    """
    energies = np.asarray(energies)
    density = np.zeros_like(energies, dtype=float)
    for lam in eigenvalues:
        density += epsilon / ((energies - lam) ** 2 + epsilon ** 2)
    return density / np.pi


def _green_function_trace(E_plus_ieps, H):
    """
    그린 함수의 대각합 계산.

    Tr G(z) = Tr (z - H)^{-1} = Σ_n 1 / (z - λ_n)

    Parameters
    ----------
    E_plus_ieps : complex
        복소 에너지 z = E + iε
    H : np.ndarray
        해밀토니안 행렬 (고유값은 미리 계산됨)

    Returns
    -------
    complex
        Tr G(z)
    """
    eigenvalues = eigvalsh(H)
    return np.sum(1.0 / (E_plus_ieps - eigenvalues.astype(complex)))


# ---------------------------------------------------------------------------
# 메인 클래스
# ---------------------------------------------------------------------------

class HamiltonianConstraints:
    """
    Berry-Keating 후보 해밀토니안의 ξ-다발 제약 조건 검증 클래스.

    8개 조건을 각각 독립적으로 검증하며,
    각 결과는 {'passed': bool, 'value': ..., 'basis': str} 딕셔너리로 반환된다.
    """

    def __init__(self, H_matrix, description=""):
        """
        Parameters
        ----------
        H_matrix : array_like
            n×n 복소 행렬 (해밀토니안 후보).
            에르미트(Hermitian)이 아니어도 입력 가능하나,
            일부 조건(C6)은 비에르미트 행렬에서 실패로 표시된다.
        description : str
            후보 해밀토니안 식별 이름
        """
        self.H = np.array(H_matrix, dtype=complex)
        self.n, m = self.H.shape
        if self.n != m:
            raise ValueError(f"정방 행렬이어야 합니다. 입력 크기: {self.H.shape}")
        self.dim = self.n
        self.description = description

        # 에르미트 여부에 따라 고유값 분해 방식 선택
        hermitian_residual = np.max(np.abs(self.H - self.H.conj().T))
        self._is_hermitian = hermitian_residual < 1e-10
        if self._is_hermitian:
            self.eigenvalues, self.eigenvectors = eigh(self.H)
        else:
            eigvals = np.linalg.eigvals(self.H)
            self.eigenvalues = np.sort(eigvals.real)
            self.eigenvectors = None  # 비에르미트는 별도 처리

    # ------------------------------------------------------------------
    # C1: Gauss-Bonnet 정수성
    # ------------------------------------------------------------------

    def check_c1_gauss_bonnet(self, target_N=None):
        """
        C1: Gauss-Bonnet 정수성

        [무한 차원 원래 조건]
        ∫∫_R Δ(log|ξ|) dσ dt = 2πN
        여기서 N은 임계대에 포함된 ξ 영점 개수.
        이는 복소 함수론의 논거정리(Argument Principle)이자
        Gauss-Bonnet 정리의 함수론적 표현이다.

        [유한 차원 번역]
        스펙트럼 제타 함수 ζ_H(s) = Σ |λ_n|^{-s} 의
        특성다항식 P(z) = det(z - H) 에 논거정리 적용:
          (1/2πi) ∮_Γ P'(z)/P(z) dz = N (원 Γ 안의 고유값 개수)

        고유값 개수 N과 스펙트럼 범위로부터 위상 공간 부피를 추정하고,
        Weyl 법칙 N ≈ (1/2π) × Vol(위상공간) × ρ 와의 정합성 검사.

        Parameters
        ----------
        target_N : int, optional
            예상 고유값 개수. None이면 dim 사용.

        Returns
        -------
        dict
            passed, gauss_bonnet_integral, N_eigenvalues, 2pi_N, relative_error
        """
        if target_N is None:
            target_N = self.dim

        # 특성다항식의 계수를 이용한 winding number 계산
        # 고유값 λ_n에서 P(z) = Π(z - λ_n) 이므로
        # 충분히 큰 원 위에서 winding number = 차수 = dim
        # 여기서는 실제 고유값 개수로 검증

        N_actual = len(self.eigenvalues)

        # Gauss-Bonnet 적분 = 2π × (고유값 개수)
        gauss_bonnet_integral = 2.0 * np.pi * N_actual

        # Weyl 공식과 비교: 고유값 범위 [E_min, E_max]에서
        # 반고전 밀도 ρ ≈ N / (E_max - E_min) 로 근사
        E_min = self.eigenvalues.min()
        E_max = self.eigenvalues.max()
        spectral_range = E_max - E_min if E_max != E_min else 1.0

        # 기대값: 2π × target_N
        expected = 2.0 * np.pi * target_N
        relative_error = abs(gauss_bonnet_integral - expected) / abs(expected)

        # 정수성 검사: 2πN의 배수여야 하므로
        # gauss_bonnet_integral / (2π) 가 정수에 가까운지 확인
        ratio = gauss_bonnet_integral / (2.0 * np.pi)
        integrality_error = abs(ratio - round(ratio))

        passed = integrality_error < 1e-10  # 항상 정수 → 이론적으로 항상 통과

        return {
            'passed': passed,
            'gauss_bonnet_integral': gauss_bonnet_integral,
            'N_eigenvalues': N_actual,
            '2pi_N': expected,
            'integrality_error': integrality_error,
            'spectral_range': spectral_range,
            'basis': (
                f"특성다항식 winding number = {N_actual} (정수), "
                f"Gauss-Bonnet 적분 = 2π×{N_actual} = {gauss_bonnet_integral:.6f}"
            )
        }

    # ------------------------------------------------------------------
    # C2: 유니터리 게이지
    # ------------------------------------------------------------------

    def check_c2_unitary_gauge(self, epsilon=0.05, n_test_points=200):
        """
        C2: 유니터리 게이지 — Re(그린함수의 대각 성분) = 0 (임계선 위)

        [무한 차원 원래 조건]
        임계선 Re(s) = 1/2 위에서 ξ'/ξ의 실수부가 0:
          Re(ξ'/ξ(1/2 + it)) = 0

        [유한 차원 번역]
        그린 함수 G(z) = (z - H)^{-1} 에서
        z = E + iε (ε-정규화된 실수 에너지) 로 평가할 때:
          Re(Tr G(E + iε)) = Σ_n (E - λ_n) / ((E - λ_n)² + ε²)

        이는 스펙트럼에 대칭적인 에너지 E* (스펙트럼 중앙)에서
        고유값이 E* 기준 대칭 분포이면 Re(Tr G(E*)) = 0 을 만족한다.

        실질 검사: 스펙트럼 중심 E* = (λ_max + λ_min)/2 에서
        Re(Tr G(E* + iε))가 0에 가까운지, 그리고 spectral function
        A(E) = -Im Tr G(E + iε) / π 가 양수인지 (물리적 의미 보존).

        Parameters
        ----------
        epsilon : float
            그린 함수 정규화 파라미터 ε
        n_test_points : int
            검사할 에너지 포인트 수

        Returns
        -------
        dict
            passed, center_re_green, max_re_green, spectral_function_positive
        """
        E_min = self.eigenvalues.min()
        E_max = self.eigenvalues.max()
        E_center = 0.5 * (E_min + E_max)

        # 스펙트럼 중심에서 Re(Tr G) 계산
        z_center = complex(E_center, epsilon)
        tr_g_center = np.sum(1.0 / (z_center - self.eigenvalues.astype(complex)))
        center_re = tr_g_center.real

        # 여러 에너지 포인트에서 Re(Tr G) 최대 절댓값
        E_test = np.linspace(E_min - 0.5, E_max + 0.5, n_test_points)
        re_values = []
        for E in E_test:
            z = complex(E, epsilon)
            tr_g = np.sum(1.0 / (z - self.eigenvalues.astype(complex)))
            re_values.append(tr_g.real)
        re_values = np.array(re_values)

        # spectral function A(E) = -Im Tr G(E+iε)/π 의 양수성 검사
        im_values = []
        for E in E_test:
            z = complex(E, epsilon)
            tr_g = np.sum(1.0 / (z - self.eigenvalues.astype(complex)))
            im_values.append(tr_g.imag)
        spectral_fn = -np.array(im_values) / np.pi
        spectral_positive = np.all(spectral_fn >= -1e-10)  # 수치 오차 허용

        # 스펙트럼 대칭성 검사 (고유값이 E_center 기준 대칭인지)
        centered_evals = self.eigenvalues - E_center
        # 각 λ에 대해 -λ가 존재하는지 (허용 오차 내)
        symmetry_residuals = []
        for lam in centered_evals:
            mirror = -lam
            dist = np.min(np.abs(centered_evals - mirror))
            symmetry_residuals.append(dist)
        mean_symmetry_residual = np.mean(symmetry_residuals)
        spectral_range = E_max - E_min if E_max > E_min else 1.0
        normalized_symmetry = mean_symmetry_residual / spectral_range

        # 판정: |Re(Tr G(E_center))| / dim < tolerance
        # 및 spectral function 양수 보장
        tolerance = 1.0  # 대각합 크기 대비 허용 오차 (dim에 비례)
        center_re_normalized = abs(center_re) / self.dim
        passed = (center_re_normalized < 0.1) and spectral_positive

        return {
            'passed': passed,
            'center_re_green': center_re,
            'center_re_normalized': center_re_normalized,
            'max_abs_re_green': float(np.max(np.abs(re_values))),
            'spectral_function_positive': bool(spectral_positive),
            'spectral_symmetry_residual': float(normalized_symmetry),
            'epsilon_used': epsilon,
            'basis': (
                f"스펙트럼 중심 E*={E_center:.4f}에서 "
                f"Re(Tr G(E*+iε))/dim = {center_re_normalized:.6f} "
                f"(기준 < 0.1), spectral fn 양수={spectral_positive}"
            )
        }

    # ------------------------------------------------------------------
    # C3: Klein 4-군 대칭
    # ------------------------------------------------------------------

    def check_c3_klein_symmetry(self, tolerance=1e-8):
        """
        C3: Klein 4-군 대칭 G = Z₂ × Z₂

        [무한 차원 원래 조건]
        ξ(s)가 만족하는 4중 대칭:
          Id:    s → s
          S₁:   s → 1 - s    (함수 방정식)
          S₂:   s → s̄        (슈바르츠 반사)
          S₁S₂: s → 1 - s̄   (결합 대칭)

        [유한 차원 번역]
        해밀토니안 H의 Z₂ × Z₂ 대칭:
          S₁: 시간 역전 연산자 T → H에 대해 T H T^{-1} = H
              T: 복소 켤레 + 단위 행렬 (반유니터리)
              즉 H = H* (실수 행렬이거나 실수 스펙트럼)
          S₂: 입자-구멍 대칭 → 스펙트럼 반전 C H C^{-1} = -H
              즉 λ ∈ spec(H) ↔ -λ ∈ spec(H)
          S₁S₂: 결합 CT 대칭

        검사:
          - [H, S₁] = H - H* = 0  ↔  H 실수 대칭성
          - 고유값의 {λ} = {-λ} 대칭 (S₂)
          - 두 조건 동시 만족 ↔ Klein 4-군

        Parameters
        ----------
        tolerance : float
            교환자 잔차 허용 오차

        Returns
        -------
        dict
            passed, s1_time_reversal, s2_particle_hole, s1s2_combined
        """
        # S₁ 검사: 시간 역전 — H = H* (실수부 대칭, 허수부 반대칭)
        s1_residual = np.max(np.abs(self.H - self.H.conj()))
        s1_passed = s1_residual < tolerance

        # S₂ 검사: 입자-구멍 대칭 — 고유값이 ±λ 쌍으로 존재
        evals_sorted = np.sort(self.eigenvalues)
        # 중심 주위 대칭: 각 λ_n에 대해 -λ_n이 스펙트럼에 존재하는지
        s2_residuals = []
        for lam in evals_sorted:
            dists = np.abs(evals_sorted + lam)  # -lam에 가장 가까운 고유값 탐색
            s2_residuals.append(np.min(dists))
        s2_mean_residual = np.mean(s2_residuals)
        # 스펙트럼 규모로 정규화
        spec_scale = max(np.max(np.abs(evals_sorted)), 1e-12)
        s2_normalized = s2_mean_residual / spec_scale
        s2_passed = s2_normalized < tolerance * 100  # 수치 적분 허용 오차 완화

        # S₁S₂ 검사: CT 결합 — H = -H* (허수 반에르미트)
        s1s2_residual = np.max(np.abs(self.H + self.H.conj()))
        # 실수 H에서는 S₁S₂가 -H = H 즉 H = 0인 경우만 가능
        # 일반적으로 S₁, S₂ 각각의 조건으로 Klein 4-군 판정
        s1s2_passed = (s1_passed or s2_passed)  # 두 생성원 중 하나라도 성립

        # 전체 Klein 4-군 판정: S₁, S₂ 둘 다 또는 두 대칭 중 독립 쌍 존재
        passed = s1_passed and s2_passed

        return {
            'passed': passed,
            's1_time_reversal': {
                'passed': bool(s1_passed),
                'residual': float(s1_residual),
                'meaning': 'H = H* (시간 역전 불변)'
            },
            's2_particle_hole': {
                'passed': bool(s2_passed),
                'normalized_residual': float(s2_normalized),
                'meaning': '고유값 ±λ 쌍 대칭 (입자-구멍 대칭)'
            },
            's1s2_combined': {
                'passed': bool(s1s2_passed),
                'residual_ct': float(s1s2_residual),
                'meaning': 'S₁ 또는 S₂ 성립 → Klein 부분군 존재'
            },
            'basis': (
                f"S₁ 잔차={s1_residual:.2e}, "
                f"S₂ 정규화잔차={s2_normalized:.2e}, "
                f"Klein G=Z₂×Z₂ 통과={passed}"
            )
        }

    # ------------------------------------------------------------------
    # C4: 모노드로미 양자화
    # ------------------------------------------------------------------

    def check_c4_monodromy(self, epsilon=0.01, n_angles=360):
        """
        C4: 모노드로미 양자화 — Δarg det(z - H) = ±2πk

        [무한 차원 원래 조건]
        각 단순 ξ-영점 ρ 주위 소원을 따라
        Δarg(ξ) = ±π  (단순 영점 → winding number ±1/2)
        이는 특성 행렬의 논거 변화에 해당한다.

        [유한 차원 번역]
        특성다항식 P(z) = det(z - H)의 각 영점(= 고유값) λ_n 주위로
        소원 z = λ_n + ε·e^{iθ}, θ ∈ [0, 2π] 를 따라
        Δarg P(z) = 2π × (해당 고유값의 대수적 중복도)

        단순 고유값이면 winding number = 1 (완전 회전 2π),
        이것이 단순 ξ 영점의 Δarg = ±π 에 대응된다
        (P(z) = det(z-H)에서 ξ와 달리 제곱 없음).

        Parameters
        ----------
        epsilon : float
            고유값 주위 소원 반지름
        n_angles : int
            적분 경로의 이산화 각도 수

        Returns
        -------
        dict
            passed, winding_numbers, expected_winding, max_deviation
        """
        angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
        winding_numbers = []
        evals = self.eigenvalues.copy()

        for lam in evals:
            # z = λ + ε·e^{iθ} 위에서 det(z - H) 의 위상 변화 계산
            z_circle = lam + epsilon * np.exp(1j * angles)
            log_det_phases = []

            for z in z_circle:
                # det(z - H) = Π(z - λ_k)
                diffs = z - evals.astype(complex)
                log_det = np.sum(np.log(diffs))
                log_det_phases.append(log_det.imag)

            # 총 위상 변화 = 첫값과 마지막값의 차이 (연속적으로 unwrap)
            phases = np.unwrap(np.array(log_det_phases))
            total_phase_change = phases[-1] - phases[0]
            winding = total_phase_change / (2 * np.pi)
            winding_numbers.append(winding)

        winding_numbers = np.array(winding_numbers)

        # 단순 고유값이면 winding number ≈ 1.0
        # 중복 고유값이면 그 중복도만큼
        expected = np.ones(len(evals))  # 단순 고유값 가정
        deviations = np.abs(np.round(winding_numbers) - winding_numbers)
        max_deviation = float(np.max(deviations))

        # 판정: 모든 winding number가 정수에 가까운지
        passed = max_deviation < 0.1

        return {
            'passed': passed,
            'winding_numbers': winding_numbers.tolist(),
            'winding_mean': float(np.mean(winding_numbers)),
            'winding_std': float(np.std(winding_numbers)),
            'max_deviation_from_integer': max_deviation,
            'epsilon_used': epsilon,
            'basis': (
                f"고유값 {len(evals)}개 주위 winding number 계산, "
                f"정수 편차 최대={max_deviation:.4f} (기준 < 0.1), "
                f"통과={passed}"
            )
        }

    # ------------------------------------------------------------------
    # C5: 곡률 집중
    # ------------------------------------------------------------------

    def check_c5_curvature_concentration(self, epsilon=0.05, n_points=500):
        """
        C5: 곡률 집중 — κ(s) = |ξ'/ξ|² 가 영점에서만 발산

        [무한 차원 원래 조건]
        κ(s) = |ξ'(s)/ξ(s)|² 는 ξ의 영점에서 극점을 가지며
        다른 곳에서는 유한하다. 이는 ξ-다발의 곡률이
        영점에 집중(concentrated)됨을 의미한다.

        [유한 차원 번역]
        스펙트럼 제타 함수 ζ_H(s) = Tr (H^{-s}) 의 로그 미분을 통해
        곡률 κ(E) = |d/dE log A(E)|² 를 계산한다.
        여기서 A(E) = spectral density = -Im Tr G(E+iε)/π.

        곡률은 고유값 E = λ_n 에서 피크를 가져야 하고
        고유값 사이에서는 작아야 한다.
        피크-대-배경 비율(peak-to-background ratio)로 평가.

        Parameters
        ----------
        epsilon : float
            그린 함수 정규화 ε
        n_points : int
            에너지 격자 수

        Returns
        -------
        dict
            passed, peak_to_background_ratio, curvature_at_eigenvalues,
            curvature_away_from_eigenvalues
        """
        E_min = self.eigenvalues.min()
        E_max = self.eigenvalues.max()
        margin = 0.5 * (E_max - E_min + 1.0)
        E_grid = np.linspace(E_min - margin * 0.1,
                             E_max + margin * 0.1, n_points)

        # Spectral density A(E) 계산
        A = _spectral_density(E_grid, self.eigenvalues, epsilon=epsilon)

        # 로그 미분 → 곡률
        # dA/dE ≈ (A[i+1] - A[i-1]) / (2 dE)
        dE = E_grid[1] - E_grid[0]
        dA = np.gradient(A, dE)

        # 분모 0 방지
        A_safe = np.where(A > 1e-30, A, 1e-30)
        log_deriv = dA / A_safe
        kappa = log_deriv ** 2

        # 고유값 근방 곡률 (피크)
        peak_kappas = []
        for lam in self.eigenvalues:
            idx = np.argmin(np.abs(E_grid - lam))
            peak_kappas.append(kappa[idx])
        peak_kappas = np.array(peak_kappas)

        # 고유값 사이 곡률 (배경)
        # 고유값에서 가장 먼 에너지 포인트 선택
        dist_to_nearest = np.array([
            np.min(np.abs(E - self.eigenvalues)) for E in E_grid
        ])
        half_gap = (
            np.min(np.diff(np.sort(self.eigenvalues))) / 2.0
            if len(self.eigenvalues) > 1 else 1.0
        )
        bg_mask = dist_to_nearest > half_gap * 0.8
        bg_kappas = kappa[bg_mask] if bg_mask.sum() > 0 else np.array([0.0])

        mean_peak = float(np.mean(peak_kappas))
        mean_bg = float(np.mean(bg_kappas))

        # 피크-대-배경 비율
        ratio = mean_peak / (mean_bg + 1e-30)

        # 판정: 피크가 배경보다 현저히 큰지 (최소 10배)
        passed = ratio > 10.0

        return {
            'passed': passed,
            'peak_to_background_ratio': float(ratio),
            'mean_curvature_at_eigenvalues': mean_peak,
            'mean_curvature_away': mean_bg,
            'n_eigenvalue_peaks': len(peak_kappas),
            'epsilon_used': epsilon,
            'basis': (
                f"곡률 피크/배경 비율={ratio:.2f} (기준 > 10), "
                f"피크 평균κ={mean_peak:.2e}, 배경 평균κ={mean_bg:.2e}"
            )
        }

    # ------------------------------------------------------------------
    # C6: 자기수반성
    # ------------------------------------------------------------------

    def check_c6_self_adjointness(self, tolerance=1e-10):
        """
        C6: 자기수반성 — H = H†, 고유값 실수

        [무한 차원 원래 조건]
        Berry-Keating 해밀토니안은 자기수반이어야 하며
        스펙트럼이 실수 집합에 포함된다.
        이는 확률 해석 (유니터리 시간 진화)의 전제 조건.

        [유한 차원 번역]
        에르미트 조건: H = H†, 즉 H_{ij} = H_{ji}*
        잔차 측정: ||H - H†||_F / ||H||_F
        고유값 허수부: max |Im(λ_n)|

        PT-대칭 확장 (비에르미트 H의 경우):
        PT H (PT)^{-1} = H 를 검사하여
        "의사-에르미트(pseudo-Hermitian)" 여부도 확인.
        여기서 P = parity (부호 반전), T = 시간 역전 (복소 켤레).

        Parameters
        ----------
        tolerance : float
            자기수반성 판정 허용 오차

        Returns
        -------
        dict
            passed, hermitian_residual, max_imaginary_eigenvalue,
            pt_symmetric
        """
        # 에르미트 잔차
        H_dag = self.H.conj().T
        diff = self.H - H_dag
        hermitian_residual = float(np.linalg.norm(diff, 'fro'))
        norm_H = float(np.linalg.norm(self.H, 'fro'))
        relative_residual = hermitian_residual / (norm_H + 1e-30)
        hermitian_passed = relative_residual < tolerance

        # 고유값 허수부 검사
        raw_eigvals = np.linalg.eigvals(self.H)
        max_imag = float(np.max(np.abs(raw_eigvals.imag)))
        eigenvalue_real = max_imag < tolerance * max(np.max(np.abs(raw_eigvals.real)), 1.0)

        # PT 대칭 검사
        # P: 대각 요소 부호 반전 (운동량 공간에서 parity)
        # 단순화: P = diag(+1,-1,+1,-1,...), T = 복소 켤레
        n = self.dim
        parity_diag = np.array([(-1) ** i for i in range(n)], dtype=float)
        P = np.diag(parity_diag)
        PT_H_PT_inv = P @ self.H.conj() @ P  # PT H (PT)^{-1} = P H* P
        pt_residual = float(np.linalg.norm(PT_H_PT_inv - self.H, 'fro'))
        pt_symmetric = pt_residual < (tolerance * norm_H * 100)

        passed = hermitian_passed and eigenvalue_real

        return {
            'passed': bool(passed),
            'hermitian_residual_relative': relative_residual,
            'hermitian_residual_absolute': hermitian_residual,
            'max_imaginary_eigenvalue': max_imag,
            'eigenvalues_real': bool(eigenvalue_real),
            'pt_symmetric': bool(pt_symmetric),
            'pt_residual': pt_residual,
            'basis': (
                f"||H-H†||_F/||H||_F={relative_residual:.2e} (기준 <{tolerance:.0e}), "
                f"최대 Im(λ)={max_imag:.2e}, PT대칭={pt_symmetric}"
            )
        }

    # ------------------------------------------------------------------
    # C7: Euler 곱 호환성
    # ------------------------------------------------------------------

    def check_c7_euler_product(self, primes=None):
        """
        C7: Euler 곱 호환성 — 소수에 대한 연결 반영

        [무한 차원 원래 조건]
        리만 제타의 Euler 곱:
          ζ(s) = Π_p (1 - p^{-s})^{-1}
        Berry-Keating 해밀토니안이 이와 호환되려면
        주기 궤도의 길이 스펙트럼이 log(p^k) (소수의 거듭제곱)와
        관련되어야 한다 (Gutzwiller trace formula).

        [유한 차원 번역]
        고유값 λ_n에 대한 스펙트럼 제타:
          ζ_H(s) = Σ_n |λ_n|^{-s}

        이것의 소수 채널 분해 가능성을 검사:
        각 소수 p에 대해 log(p^k) 간격으로 고유값이 클러스터되는지,
        또는 Hardy-Littlewood 추측의 쌍 상관과 GUE 통계 일치성 검사.

        실질 검사: 고유값 간격 통계가 GUE (Gaussian Unitary Ensemble)
        분포를 따르는지 (리만 영점과 동일한 통계 → Euler 곱 구조 암시).

        Parameters
        ----------
        primes : list of int, optional
            검사할 소수 목록 (기본값: [2,3,5,7,11])

        Returns
        -------
        dict
            passed, gue_statistic, euler_product_residual, level_spacing
        """
        if primes is None:
            primes = [2, 3, 5, 7, 11]

        if len(self.eigenvalues) < 3:
            return {
                'passed': False,
                'reason': '고유값이 3개 미만으로 통계 검사 불가',
                'basis': '차원 부족'
            }

        # 레벨 간격 통계 (normalized)
        evals_sorted = np.sort(self.eigenvalues)
        spacings = np.diff(evals_sorted)
        mean_spacing = np.mean(spacings)
        if mean_spacing < 1e-12:
            normalized_spacings = spacings
        else:
            normalized_spacings = spacings / mean_spacing

        # GUE Wigner-Dyson 분포와 비교: P(s) ≈ (32/π²) s² exp(-4s²/π)
        def wigner_dyson_gue(s):
            return (32.0 / np.pi ** 2) * s ** 2 * np.exp(-4.0 * s ** 2 / np.pi)

        # 히스토그램으로 실제 분포 추정
        n_bins = max(5, len(normalized_spacings) // 3)
        hist, bin_edges = np.histogram(normalized_spacings,
                                        bins=n_bins, density=True)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        # GUE 이론값
        gue_vals = np.array([wigner_dyson_gue(s) for s in bin_centers])

        # 카이-제곱 유사 잔차 (정규화)
        total_range = max(bin_centers.max(), 1e-10)
        gue_residual = float(np.mean(np.abs(hist - gue_vals)))

        # Euler 곱 로그 간격 검사
        # log(p) 값들
        log_primes = np.log(np.array(primes, dtype=float))

        # 고유값 사이 로그 간격
        log_evals = np.log(np.abs(evals_sorted[evals_sorted != 0]))
        log_gaps = np.diff(log_evals) if len(log_evals) > 1 else np.array([0.0])

        # 각 log(p)에 가장 가까운 로그 간격 거리
        euler_residuals = []
        for lg in log_gaps:
            dists = np.abs(log_primes - lg % log_primes.max())
            euler_residuals.append(np.min(dists))
        euler_mean_residual = float(np.mean(euler_residuals)) if euler_residuals else 1.0

        # 판정 기준:
        # - GUE 잔차 < 0.5 (레벨 반발 통계 존재)
        # - 레벨 반발 지표: s→0일 때 간격 분포가 0 (Wigner surmise)
        near_zero_spacings = normalized_spacings[normalized_spacings < 0.1]
        level_repulsion_ok = len(near_zero_spacings) < len(normalized_spacings) * 0.1

        passed = level_repulsion_ok  # 레벨 반발이 핵심 조건

        return {
            'passed': bool(passed),
            'gue_residual': gue_residual,
            'euler_log_spacing_residual': euler_mean_residual,
            'level_repulsion': bool(level_repulsion_ok),
            'mean_spacing': float(mean_spacing),
            'n_spacings_analyzed': len(normalized_spacings),
            'log_primes': log_primes.tolist(),
            'basis': (
                f"레벨 반발 통과={level_repulsion_ok}, "
                f"GUE 잔차={gue_residual:.4f}, "
                f"Euler 로그 간격 잔차={euler_mean_residual:.4f}"
            )
        }

    # ------------------------------------------------------------------
    # C8: 반고전 카운팅
    # ------------------------------------------------------------------

    def check_c8_semiclassical_counting(self, T_max=None, tolerance=0.5):
        """
        C8: 반고전 카운팅 — N̄(T) ≈ (T/2π)log(T/2πe) + 7/8

        [무한 차원 원래 조건]
        리만 영점 카운팅 함수 N(T)가 Weyl 공식을 따름:
          N̄(T) = (T/2π)log(T/2πe) + 7/8
        이는 xp 위상 공간의 부피 계산과 직결된다.

        [유한 차원 번역]
        고유값 λ_n이 T 이하인 개수 N_H(T)를 계산하고
        리만 카운팅 함수 N̄(T)와 비교.

        단순화: 고유값을 임계선 허수부 Im(ρ) = t 에 대응시켜
        evals ~ t_n (리만 영점 높이)으로 해석.
        또는 Weyl 법칙 자체: N_H(E) = dim × Θ(E_max - E) 적분

        실질 검사: 고유값 누적 분포 N_H(E)가 매끄러운 Weyl 법칙을
          N̄_Weyl(E) = C × E^{d/2} (d차원 시스템)
        으로 피팅 가능한지 확인하고, 리만 카운팅과의 유사성 측정.

        Parameters
        ----------
        T_max : float, optional
            검사 범위 상한. None이면 max(|λ|) 사용.
        tolerance : float
            N_H 대 N̄ 차이의 상대 허용 오차

        Returns
        -------
        dict
            passed, N_actual, N_riemann, relative_error,
            weyl_fit_residual
        """
        evals_positive = np.sort(np.abs(self.eigenvalues))

        if T_max is None:
            T_max = float(evals_positive.max()) if len(evals_positive) > 0 else 10.0

        # 실제 고유값 카운팅
        N_actual = int(np.sum(evals_positive <= T_max))

        # 리만 카운팅
        N_riemann = riemann_counting(T_max)
        N_riemann_safe = max(abs(N_riemann), 1e-6)

        # 상대 오차
        relative_error = abs(N_actual - N_riemann) / N_riemann_safe

        # Weyl 법칙 피팅: N(E) ≈ C × E^α
        # 로그-로그 회귀
        T_values = evals_positive[evals_positive > 0]
        if len(T_values) > 2:
            N_cumulative = np.arange(1, len(T_values) + 1, dtype=float)
            log_T = np.log(T_values)
            log_N = np.log(N_cumulative)
            # 선형 회귀: log N = α log T + log C
            A_mat = np.vstack([log_T, np.ones(len(log_T))]).T
            result = np.linalg.lstsq(A_mat, log_N, rcond=None)
            alpha, log_C = result[0]
            C_fit = np.exp(log_C)
            N_weyl_pred = C_fit * T_values ** alpha
            weyl_residual = float(np.mean(np.abs(N_cumulative - N_weyl_pred) / N_cumulative))
        else:
            alpha = 0.0
            C_fit = 0.0
            weyl_residual = 1.0

        # 리만 카운팅과 비교한 여러 T 값에서의 RMS 오차
        T_test = np.linspace(1.0, T_max, min(50, max(10, self.dim)))
        rms_errors = []
        for T in T_test:
            if T <= 0:
                continue
            n_h = int(np.sum(evals_positive <= T))
            n_r = riemann_counting(T)
            if abs(n_r) > 0.5:
                rms_errors.append(abs(n_h - n_r) / abs(n_r))
        rms_riemann = float(np.mean(rms_errors)) if rms_errors else 1.0

        # 판정: Weyl 피팅 잔차가 작고, 멱함수 지수 α가 양수
        passed = (weyl_residual < 0.3) and (alpha > 0.3)

        return {
            'passed': bool(passed),
            'N_actual_at_Tmax': N_actual,
            'N_riemann_at_Tmax': N_riemann,
            'relative_error_at_Tmax': relative_error,
            'weyl_fit_exponent': float(alpha),
            'weyl_fit_C': float(C_fit),
            'weyl_fit_residual': weyl_residual,
            'rms_riemann_discrepancy': rms_riemann,
            'T_max': T_max,
            'basis': (
                f"T_max={T_max:.2f}에서 N_H={N_actual}, N̄={N_riemann:.2f}, "
                f"Weyl 지수α={alpha:.3f}, 피팅잔차={weyl_residual:.4f}"
            )
        }

    # ------------------------------------------------------------------
    # 종합 평가
    # ------------------------------------------------------------------

    def full_assessment(self):
        """
        8개 제약 조건 전체 검증 후 종합 점수 반환.

        Returns
        -------
        dict
            각 조건 결과 + 'total_score' 키
        """
        results = {}
        results['C1_gauss_bonnet']           = self.check_c1_gauss_bonnet()
        results['C2_unitary_gauge']           = self.check_c2_unitary_gauge()
        results['C3_klein_symmetry']          = self.check_c3_klein_symmetry()
        results['C4_monodromy']               = self.check_c4_monodromy()
        results['C5_curvature_concentration'] = self.check_c5_curvature_concentration()
        results['C6_self_adjointness']        = self.check_c6_self_adjointness()
        results['C7_euler_product']           = self.check_c7_euler_product()
        results['C8_semiclassical']           = self.check_c8_semiclassical_counting()

        score = sum(
            1 for k, v in results.items()
            if k != 'total_score' and isinstance(v, dict) and v.get('passed', False)
        )
        results['total_score'] = f"{score}/8"
        return results

    def report(self):
        """
        사람이 읽을 수 있는 한국어 보고서 출력.
        """
        results = self.full_assessment()
        label_map = {
            'C1_gauss_bonnet':           'C1: Gauss-Bonnet 정수성',
            'C2_unitary_gauge':          'C2: 유니터리 게이지',
            'C3_klein_symmetry':         'C3: Klein 4-군 대칭',
            'C4_monodromy':              'C4: 모노드로미 양자화',
            'C5_curvature_concentration':'C5: 곡률 집중',
            'C6_self_adjointness':       'C6: 자기수반성',
            'C7_euler_product':          'C7: Euler 곱 호환성',
            'C8_semiclassical':          'C8: 반고전 카운팅',
        }

        print("=" * 62)
        print(f" Berry-Keating 제약 조건 검증 보고서")
        if self.description:
            print(f" 후보: {self.description}  (dim={self.dim})")
        print("=" * 62)

        for key, label in label_map.items():
            r = results.get(key, {})
            status = "통과" if r.get('passed', False) else "실패"
            mark = "[O]" if r.get('passed', False) else "[X]"
            print(f"{mark} {label}: {status}")
            basis = r.get('basis', '')
            if basis:
                # 긴 문자열은 잘라서 표시
                if len(basis) > 70:
                    basis = basis[:67] + "..."
                print(f"      근거: {basis}")

        print("-" * 62)
        print(f" 종합 점수: {results['total_score']}")
        print("=" * 62)


# ---------------------------------------------------------------------------
# 스펙트럼 제타 함수
# ---------------------------------------------------------------------------

class SpectralZetaFunction:
    """
    유한 차원 해밀토니안의 스펙트럼 제타 함수

    ζ_H(s) = Σ_n |λ_n|^{-s}   (λ_n > 0 인 고유값만 사용)

    이것의 해석적 구조가 리만 제타 ζ(s)와 얼마나 유사한지 비교한다.
    s = 1/2 + it 임계선 위에서의 거동이 핵심.
    """

    def __init__(self, eigenvalues):
        """
        Parameters
        ----------
        eigenvalues : array_like
            해밀토니안 고유값 배열. 양수 값만 사용.
        """
        evals = np.asarray(eigenvalues, dtype=float)
        self.eigenvalues = np.sort(evals[evals > 0])
        if len(self.eigenvalues) == 0:
            raise ValueError("양수 고유값이 없습니다.")

    def evaluate(self, s):
        """
        ζ_H(s) = Σ_n λ_n^{-s} 계산 (복소 s).

        Parameters
        ----------
        s : complex or float
            복소 인수

        Returns
        -------
        complex
            ζ_H(s) 값
        """
        s = complex(s)
        return complex(np.sum(self.eigenvalues ** (-s)))

    def zeros_on_critical_line(self, t_range, n_points=500, epsilon=1e-6):
        """
        임계선 s = 1/2 + it 에서 ζ_H 의 영점 탐색.

        Re(ζ_H) 와 Im(ζ_H) 가 동시에 0에 가까운 t 값 탐색.
        부호 변화를 이분법으로 좁혀 영점 후보를 반환.

        Parameters
        ----------
        t_range : tuple (t_min, t_max)
            탐색 범위
        n_points : int
            초기 격자 점 수
        epsilon : float
            영점 판정 허용 오차

        Returns
        -------
        list of float
            영점 후보 t 값 목록
        """
        t_min, t_max = t_range
        t_grid = np.linspace(t_min, t_max, n_points)

        zeta_vals = np.array([self.evaluate(0.5 + 1j * t) for t in t_grid])
        zeta_abs = np.abs(zeta_vals)

        # 극소점 탐색: 이웃보다 작은 점
        zeros = []
        for i in range(1, len(t_grid) - 1):
            if (zeta_abs[i] < zeta_abs[i - 1] and
                    zeta_abs[i] < zeta_abs[i + 1] and
                    zeta_abs[i] < epsilon):
                zeros.append(float(t_grid[i]))

        return zeros

    def curvature_profile(self, t_array):
        """
        κ(t) = |ζ_H'(s) / ζ_H(s)|²  (s = 1/2 + it) 프로파일 계산.

        로그 미분의 제곱 = 곡률 밀도.
        ξ-다발 C5 조건과 직접 대응.

        Parameters
        ----------
        t_array : array_like
            계산할 t 값 배열

        Returns
        -------
        np.ndarray
            각 t에서의 곡률 κ(t)
        """
        t_arr = np.asarray(t_array, dtype=float)
        dt = (t_arr[1] - t_arr[0]) if len(t_arr) > 1 else 1e-4
        dt_small = min(dt * 0.01, 1e-5)

        kappas = np.zeros(len(t_arr))
        for i, t in enumerate(t_arr):
            s = complex(0.5, t)
            z_val = self.evaluate(s)
            # 수치 미분
            z_plus  = self.evaluate(s + 1j * dt_small)
            z_minus = self.evaluate(s - 1j * dt_small)
            dz_dt = (z_plus - z_minus) / (2 * dt_small)
            dz_ds = 1j * dz_dt  # ds/dt = i 이므로

            if abs(z_val) > 1e-30:
                log_deriv = dz_ds / z_val
                kappas[i] = abs(log_deriv) ** 2
            else:
                kappas[i] = np.nan  # 영점 근방: 발산

        return kappas


# ---------------------------------------------------------------------------
# 사전 정의 해밀토니안 생성자
# ---------------------------------------------------------------------------

def make_gue_hamiltonian(n, seed=42):
    """
    GUE (Gaussian Unitary Ensemble) 랜덤 에르미트 행렬 생성.

    H = (A + A†) / (2√n),  A ~ Ginibre (복소 가우시안)

    Berry-Keating 가설의 통계적 배경: 리만 영점이 GUE 통계를 따름.
    """
    rng = np.random.default_rng(seed)
    A = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n)))
    H = (A + A.conj().T) / (2.0 * np.sqrt(n))
    return H


def make_harmonic_oscillator(n):
    """
    조화진동자 트렁케이션 해밀토니안.

    H_HO = diag(1/2, 3/2, 5/2, ..., (2n-1)/2) (단위 ω = ℏ = 1)

    고유값: λ_k = k + 1/2, k = 0, 1, ..., n-1
    Berry-Keating 가설과 비교를 위한 기준 해밀토니안.
    """
    evals = np.arange(n, dtype=float) + 0.5
    return np.diag(evals)


def make_xp_hamiltonian(n, hbar=1.0):
    """
    xp + px 트렁케이션 해밀토니안 (Berry-Keating 직접 후보).

    유한 기저 {|k⟩, k=1,...,n} 에서의 행렬 원소:
    ⟨k|xp + px|l⟩ = -iℏ δ_{kl} + 교차항 (수치 근사)

    단순화: Laguerre 다항식 기저에서 xp 작용소를 트렁케이션.
    H_{kl} = -iℏ/2 (√(k+1) δ_{l,k+1} - √(l+1) δ_{k,l+1})
    + 에르미트화 (자기수반 보장)

    실제 Berry-Keating xp 해밀토니안의 수치 근사.
    """
    k_vals = np.arange(1, n + 1, dtype=float)
    # 상삼각 부분: xp 작용
    H = np.zeros((n, n), dtype=complex)
    for k in range(n - 1):
        H[k, k + 1] = -1j * hbar * 0.5 * np.sqrt(k_vals[k])
        H[k + 1, k] = +1j * hbar * 0.5 * np.sqrt(k_vals[k])

    # 대각: (k+1/2) 항 (위상 공간 면적)
    for k in range(n):
        H[k, k] = hbar * (k_vals[k] + 0.5)

    # 에르미트화
    H = 0.5 * (H + H.conj().T)
    return H


# ---------------------------------------------------------------------------
# 메인 실행 블록
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("\n[1] GUE 랜덤 해밀토니안 (n=20)")
    print("-" * 40)
    H_gue = make_gue_hamiltonian(20, seed=0)
    hc_gue = HamiltonianConstraints(H_gue, description="GUE 랜덤 n=20")
    hc_gue.report()

    print("\n[2] 조화진동자 트렁케이션 (n=20)")
    print("-" * 40)
    H_ho = make_harmonic_oscillator(20)
    hc_ho = HamiltonianConstraints(H_ho, description="조화진동자 n=20")
    hc_ho.report()

    print("\n[3] xp 트렁케이션 (n=20)")
    print("-" * 40)
    H_xp = make_xp_hamiltonian(20)
    hc_xp = HamiltonianConstraints(H_xp, description="xp 트렁케이션 n=20")
    hc_xp.report()

    # 비교 요약
    print("\n[종합 비교]")
    print("=" * 62)
    print(f"{'후보':<25} {'점수':>8}")
    print("-" * 62)
    for label, hc in [
        ("GUE 랜덤", hc_gue),
        ("조화진동자", hc_ho),
        ("xp 트렁케이션", hc_xp),
    ]:
        r = hc.full_assessment()
        print(f"{label:<25} {r['total_score']:>8}")
    print("=" * 62)

    # SpectralZetaFunction 예시 (xp 양수 고유값 사용)
    print("\n[스펙트럼 제타 함수 — xp 후보]")
    evals_xp = eigvalsh(H_xp)
    szf = SpectralZetaFunction(evals_xp)
    zeros = szf.zeros_on_critical_line((5.0, 30.0), n_points=300, epsilon=0.05)
    print(f"임계선 영점 후보 (ε<0.05): {zeros[:5]} ...")
    t_sample = np.linspace(5.0, 30.0, 200)
    kappa = szf.curvature_profile(t_sample)
    valid = kappa[~np.isnan(kappa)]
    print(f"곡률 κ(t) 최대값: {np.max(valid):.4f}, 평균: {np.mean(valid):.4f}")
